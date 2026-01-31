import { Hono } from 'hono';
import type { Env, JwtPayload, Challenge, User } from '../types';
import { requireAdmin } from '../middleware/auth';
import { getTimeRank, calculateTimeRankBonus } from '../lib/scoring';

const PARTICIPANT_ID_RE = /^[a-zA-Z0-9_-]{3,30}$/;

const app = new Hono<{ Bindings: Env; Variables: { user: JwtPayload } }>();

// ---------- POST /api/admin/challenge/:week/start ----------
app.post('/api/admin/challenge/:week/start', async (c) => {
  const admin = requireAdmin(c);
  const week = parseInt(c.req.param('week'), 10);
  if (isNaN(week) || week < 1 || week > 5) {
    return c.json({ detail: 'Week must be between 1 and 5' }, 400);
  }

  const ch = await c.env.DB.prepare('SELECT * FROM challenges WHERE week = ?')
    .bind(week)
    .first<Challenge>();

  if (ch && ch.status !== 'not_started') {
    return c.json({ detail: `Challenge already ${ch.status}` }, 400);
  }

  const startTime = new Date().toISOString();
  await c.env.DB.prepare(
    'UPDATE challenges SET status = ?, start_time = ?, end_time = NULL, started_by = ? WHERE week = ?',
  )
    .bind('started', startTime, admin.user_id, week)
    .run();

  return c.json({ status: 'started', week, start_time: startTime, started_by: admin.user_id });
});

// ---------- POST /api/admin/challenge/:week/end ----------
app.post('/api/admin/challenge/:week/end', async (c) => {
  requireAdmin(c);
  const week = parseInt(c.req.param('week'), 10);
  if (isNaN(week) || week < 1 || week > 5) {
    return c.json({ detail: 'Week must be between 1 and 5' }, 400);
  }

  const ch = await c.env.DB.prepare('SELECT * FROM challenges WHERE week = ?')
    .bind(week)
    .first<Challenge>();

  if (!ch || ch.status !== 'started') {
    return c.json({ detail: 'Challenge is not active' }, 400);
  }

  const endTime = new Date().toISOString();
  await c.env.DB.prepare('UPDATE challenges SET status = ?, end_time = ? WHERE week = ?')
    .bind('ended', endTime, week)
    .run();

  return c.json({ status: 'ended', week, end_time: endTime });
});

// ---------- POST /api/admin/challenge/:week/restart ----------
app.post('/api/admin/challenge/:week/restart', async (c) => {
  requireAdmin(c);
  const week = parseInt(c.req.param('week'), 10);
  if (isNaN(week) || week < 1 || week > 5) {
    return c.json({ detail: 'Week must be between 1 and 5' }, 400);
  }

  // Count before deleting
  const subCount = await c.env.DB.prepare(
    'SELECT COUNT(*) as cnt FROM submissions WHERE week = ?',
  )
    .bind(week)
    .first<{ cnt: number }>();

  const evalCount = await c.env.DB.prepare(
    'SELECT COUNT(*) as cnt FROM evaluations WHERE week = ?',
  )
    .bind(week)
    .first<{ cnt: number }>();

  // Delete all related data
  await c.env.DB.batch([
    c.env.DB.prepare('DELETE FROM submissions WHERE week = ?').bind(week),
    c.env.DB.prepare('DELETE FROM evaluations WHERE week = ?').bind(week),
    c.env.DB.prepare('DELETE FROM personal_timers WHERE week = ?').bind(week),
    c.env.DB.prepare(
      'UPDATE challenges SET status = ?, start_time = NULL, end_time = NULL, started_by = NULL WHERE week = ?',
    ).bind('not_started', week),
  ]);

  return c.json({
    status: 'restarted',
    week,
    deleted_submissions: subCount?.cnt ?? 0,
    deleted_evaluations: evalCount?.cnt ?? 0,
    message: `Week ${week} has been reset. All submissions and evaluations cleared.`,
  });
});

// ---------- GET /api/admin/users ----------
app.get('/api/admin/users', async (c) => {
  requireAdmin(c);

  const { results } = await c.env.DB.prepare(
    'SELECT user_id, full_name, first_name, last_name, role, profile_image, registered_at FROM users',
  ).all<User>();

  return c.json(
    results.map((u) => ({
      user_id: u.user_id,
      full_name: u.full_name,
      first_name: u.first_name,
      last_name: u.last_name,
      role: u.role,
      profile_image: u.profile_image,
      registered_at: u.registered_at,
    })),
  );
});

// ---------- DELETE /api/admin/users/:user_id ----------
app.delete('/api/admin/users/:user_id', async (c) => {
  const admin = requireAdmin(c);
  const userId = c.req.param('user_id');

  if (userId.toLowerCase() === admin.user_id.toLowerCase()) {
    return c.json({ detail: 'Cannot delete yourself' }, 400);
  }

  const existing = await c.env.DB.prepare('SELECT user_id FROM users WHERE LOWER(user_id) = LOWER(?)')
    .bind(userId)
    .first();

  if (!existing) {
    return c.json({ detail: 'User not found' }, 404);
  }

  await c.env.DB.prepare('DELETE FROM users WHERE LOWER(user_id) = LOWER(?)').bind(userId).run();

  return c.json({ status: 'deleted', user_id: userId });
});

// ---------- POST /api/admin/evaluations/:week/:pid ----------
app.post('/api/admin/evaluations/:week/:pid', async (c) => {
  const admin = requireAdmin(c);
  const week = parseInt(c.req.param('week'), 10);
  const pid = c.req.param('pid');

  if (isNaN(week) || week < 1 || week > 5) {
    return c.json({ detail: 'Week must be between 1 and 5' }, 400);
  }
  if (!PARTICIPANT_ID_RE.test(pid)) {
    return c.json({ detail: 'Invalid participant ID format' }, 400);
  }

  const body = await c.req.json<{
    rubric_score: number;
    submission_number: number;
    feedback?: string;
    strengths?: string[];
    improvements?: string[];
  }>();

  // Verify submission exists
  const sub = await c.env.DB.prepare(
    'SELECT * FROM submissions WHERE user_id = ? AND week = ? AND submission_number = ?',
  )
    .bind(pid, week, body.submission_number)
    .first();

  if (!sub) {
    return c.json({ detail: 'Submission number not found' }, 404);
  }

  // Calculate time rank
  const timeRank = await getTimeRank(c.env.DB, week, pid);
  const timeRankBonus = calculateTimeRankBonus(timeRank);
  const rubricScore = Math.round(body.rubric_score);
  const totalScore = rubricScore + timeRankBonus;
  const evaluatedAt = new Date().toISOString();

  // Upsert evaluation
  await c.env.DB.prepare(
    `INSERT INTO evaluations (user_id, week, submission_number, rubric_score, time_rank, time_rank_bonus, total_score, feedback, strengths, improvements, status, evaluated_at, evaluated_by)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'completed', ?, ?)
     ON CONFLICT(user_id, week, submission_number) DO UPDATE SET
       rubric_score = excluded.rubric_score,
       time_rank = excluded.time_rank,
       time_rank_bonus = excluded.time_rank_bonus,
       total_score = excluded.total_score,
       feedback = excluded.feedback,
       strengths = excluded.strengths,
       improvements = excluded.improvements,
       status = 'completed',
       evaluated_at = excluded.evaluated_at,
       evaluated_by = excluded.evaluated_by`,
  )
    .bind(
      pid,
      week,
      body.submission_number,
      rubricScore,
      timeRank,
      timeRankBonus,
      totalScore,
      body.feedback || '',
      body.strengths ? JSON.stringify(body.strengths) : null,
      body.improvements ? JSON.stringify(body.improvements) : null,
      evaluatedAt,
      admin.user_id,
    )
    .run();

  return c.json({
    status: 'completed',
    participant_id: pid,
    week,
    submission_number: body.submission_number,
    scores: {
      rubric: rubricScore,
      time_rank: timeRank,
      time_rank_bonus: timeRankBonus,
      total: totalScore,
    },
  });
});

export default app;
