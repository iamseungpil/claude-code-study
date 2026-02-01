import { Hono } from 'hono';
import type { Env, AppVariables, Challenge, PersonalTimer, Submission } from '../types';
import { requireAuth } from '../middleware/auth';
import { parseWeek, isValidWeek, WEEK_VALIDATION_ERROR, checkChallengeActive } from '../lib/validation';
import { triggerSubmissionCollection } from '../lib/github';

const GITHUB_URL_RE = /^https:\/\/github\.com\/[a-zA-Z0-9_-]+\/[a-zA-Z0-9_.-]+\/?$/;

const app = new Hono<{ Bindings: Env; Variables: AppVariables }>();

// ---------- POST /api/submissions/submit ----------
app.post('/api/submissions/submit', async (c) => {
  const user = requireAuth(c);
  const body = await c.req.json<{ week: number; github_url: string }>();

  if (!body.week || !isValidWeek(body.week)) {
    return c.json({ detail: WEEK_VALIDATION_ERROR }, 400);
  }
  if (!GITHUB_URL_RE.test(body.github_url)) {
    return c.json({ detail: 'Invalid GitHub URL format' }, 400);
  }

  // Check challenge status
  const ch = await c.env.DB.prepare('SELECT * FROM challenges WHERE week = ?')
    .bind(body.week)
    .first<Challenge>();

  const error = checkChallengeActive(ch, 'Challenge has ended. No more submissions allowed.');
  if (error) return c.json({ detail: error.detail }, error.status);
  // TypeScript narrowing: checkChallengeActive guarantees ch is non-null here
  if (!ch) return c.json({ detail: 'Challenge not found' }, 404);

  // Check personal timer
  const timer = await c.env.DB.prepare(
    'SELECT * FROM personal_timers WHERE user_id = ? AND week = ?',
  )
    .bind(user.user_id, body.week)
    .first<PersonalTimer>();

  if (!timer) {
    return c.json({ detail: 'You must start your timer first before submitting.' }, 400);
  }

  // Calculate elapsed time
  const submissionTime = new Date();
  const personalStart = new Date(timer.started_at);
  const elapsedSeconds = (submissionTime.getTime() - personalStart.getTime()) / 1000;
  const elapsedMinutes = elapsedSeconds / 60;

  // Get next submission number
  const countRow = await c.env.DB.prepare(
    'SELECT MAX(submission_number) as max_num FROM submissions WHERE user_id = ? AND week = ?',
  )
    .bind(user.user_id, body.week)
    .first<{ max_num: number | null }>();

  const submissionNumber = (countRow?.max_num ?? 0) + 1;

  // Insert submission
  await c.env.DB.prepare(
    `INSERT INTO submissions (user_id, week, submission_number, github_url, submitted_at, elapsed_seconds, elapsed_minutes, personal_start_time, global_start_time)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
     ON CONFLICT(user_id, week, submission_number) DO UPDATE SET
       github_url = excluded.github_url,
       submitted_at = excluded.submitted_at,
       elapsed_seconds = excluded.elapsed_seconds,
       elapsed_minutes = excluded.elapsed_minutes`,
  )
    .bind(
      user.user_id,
      body.week,
      submissionNumber,
      body.github_url,
      submissionTime.toISOString(),
      Math.round(elapsedSeconds * 10) / 10,
      Math.round(elapsedMinutes * 10) / 10,
      timer.started_at,
      ch.start_time,
    )
    .run();

  // Update personal timer status
  await c.env.DB.prepare(
    'UPDATE personal_timers SET status = ? WHERE user_id = ? AND week = ?',
  )
    .bind('submitted', user.user_id, body.week)
    .run();

  // Create pending evaluation if not exists
  await c.env.DB.prepare(
    `INSERT OR IGNORE INTO evaluations (user_id, week, submission_number, rubric_score, time_rank, time_rank_bonus, total_score, status)
     VALUES (?, ?, ?, 0, 0, 0, 0, 'pending_review')`,
  )
    .bind(user.user_id, body.week, submissionNumber)
    .run();

  // Trigger submission collection (fire-and-forget)
  if (c.env.GITHUB_PAT) {
    c.executionCtx.waitUntil(
      triggerSubmissionCollection(c.env.GITHUB_PAT, body.week, user.user_id, body.github_url),
    );
  }

  // Build submission history for response
  const { results: allSubs } = await c.env.DB.prepare(
    'SELECT * FROM submissions WHERE user_id = ? AND week = ? ORDER BY submission_number ASC',
  )
    .bind(user.user_id, body.week)
    .all<Submission>();

  const history = allSubs.map((s) => ({
    submission_number: s.submission_number,
    github_url: s.github_url,
    submitted_at: s.submitted_at,
    elapsed_seconds: s.elapsed_seconds,
    elapsed_minutes: s.elapsed_minutes,
  }));

  return c.json({
    status: 'submitted',
    message: 'Submission received. Pending admin review.',
    participant_id: user.user_id,
    week: body.week,
    elapsed_minutes: Math.round(elapsedMinutes * 10) / 10,
    submission_number: submissionNumber,
    is_resubmission: submissionNumber > 1,
    submission_history: history,
  });
});

// ---------- GET /api/submissions/:week ----------
app.get('/api/submissions/:week', async (c) => {
  const week = parseWeek(c.req.param('week'));
  if (week === null) {
    return c.json({ detail: WEEK_VALIDATION_ERROR }, 400);
  }

  const { results } = await c.env.DB.prepare(
    `SELECT s.*, pt.started_at as personal_start_from_timer
     FROM submissions s
     LEFT JOIN personal_timers pt ON s.user_id = pt.user_id AND s.week = pt.week
     WHERE s.week = ?
     ORDER BY s.user_id, s.submission_number DESC`,
  )
    .bind(week)
    .all();

  // Group by user, take latest submission + build history
  const byUser = new Map<string, { latest: Record<string, unknown>; history: unknown[] }>();
  for (const row of results) {
    const r = row as Record<string, unknown>;
    const uid = r.user_id as string;
    if (!byUser.has(uid)) {
      byUser.set(uid, { latest: r, history: [] });
    }
    byUser.get(uid)!.history.push({
      submission_number: r.submission_number,
      github_url: r.github_url,
      submitted_at: r.submitted_at,
      elapsed_seconds: r.elapsed_seconds,
      elapsed_minutes: r.elapsed_minutes,
    });
  }

  const out = [];
  for (const [uid, data] of byUser) {
    out.push({
      participant_id: uid,
      week,
      github_url: data.latest.github_url,
      submitted_at: data.latest.submitted_at,
      personal_start_time: data.latest.personal_start_time || data.latest.personal_start_from_timer,
      global_start_time: data.latest.global_start_time,
      elapsed_seconds: data.latest.elapsed_seconds,
      elapsed_minutes: data.latest.elapsed_minutes,
      status: 'cloned',
      submission_number: data.latest.submission_number,
      submission_history: [...data.history].reverse(),
    });
  }

  return c.json(out);
});

// ---------- GET /api/submissions/:week/:pid/history ----------
app.get('/api/submissions/:week/:pid/history', async (c) => {
  const week = parseWeek(c.req.param('week'));
  if (week === null) {
    return c.json({ detail: WEEK_VALIDATION_ERROR }, 400);
  }
  const pid = c.req.param('pid');

  const { results: subs } = await c.env.DB.prepare(
    'SELECT * FROM submissions WHERE user_id = ? AND week = ? ORDER BY submission_number ASC',
  )
    .bind(pid, week)
    .all<Submission>();

  if (subs.length === 0) {
    return c.json({ submission_history: [], total_submissions: 0 });
  }

  // Get evaluations for this user/week
  const { results: evals } = await c.env.DB.prepare(
    'SELECT * FROM evaluations WHERE user_id = ? AND week = ? AND status = ?',
  )
    .bind(pid, week, 'completed')
    .all();

  const evalMap = new Map<number, Record<string, unknown>>();
  for (const ev of evals) {
    const e = ev as Record<string, unknown>;
    evalMap.set(e.submission_number as number, e);
  }

  const history = subs.map((s) => {
    const entry: Record<string, unknown> = {
      submission_number: s.submission_number,
      github_url: s.github_url,
      submitted_at: s.submitted_at,
      elapsed_seconds: s.elapsed_seconds,
      elapsed_minutes: s.elapsed_minutes,
    };

    const ev = evalMap.get(s.submission_number);
    if (ev) {
      // Parse JSON fields
      let strengths: unknown = ev.strengths;
      let improvements: unknown = ev.improvements;
      let breakdown: unknown = ev.breakdown;
      try { if (typeof strengths === 'string') strengths = JSON.parse(strengths); } catch { /* keep */ }
      try { if (typeof improvements === 'string') improvements = JSON.parse(improvements); } catch { /* keep */ }
      try { if (typeof breakdown === 'string') breakdown = JSON.parse(breakdown); } catch { /* keep */ }

      entry.evaluation = {
        rubric: ev.rubric_score,
        time_rank: ev.time_rank,
        time_rank_bonus: ev.time_rank_bonus,
        total: ev.total_score,
        status: ev.status,
        evaluated_at: ev.evaluated_at,
        feedback: ev.feedback || '',
        strengths: strengths || [],
        improvements: improvements || [],
        breakdown: breakdown || {},
      };
    }

    return entry;
  });

  return c.json({
    submission_history: history,
    total_submissions: history.length,
    personal_start_time: subs[0].personal_start_time,
  });
});

export default app;
