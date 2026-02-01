import { Hono } from 'hono';
import type { Env, AppVariables, Challenge, PersonalTimer } from '../types';
import { requireAuth } from '../middleware/auth';
import { parseWeek, WEEK_VALIDATION_ERROR, checkChallengeActive } from '../lib/validation';

const app = new Hono<{ Bindings: Env; Variables: AppVariables }>();

// ---------- POST /api/challenge/:week/start-personal ----------
app.post('/api/challenge/:week/start-personal', async (c) => {
  const user = requireAuth(c);
  const week = parseWeek(c.req.param('week'));
  if (week === null) {
    return c.json({ detail: WEEK_VALIDATION_ERROR }, 400);
  }

  const ch = await c.env.DB.prepare('SELECT * FROM challenges WHERE week = ?')
    .bind(week)
    .first<Challenge>();

  const error = checkChallengeActive(ch, 'Challenge has ended. Cannot start personal timer.');
  if (error) return c.json({ detail: error.detail }, error.status);

  // Check existing timer
  const existing = await c.env.DB.prepare(
    'SELECT * FROM personal_timers WHERE user_id = ? AND week = ?',
  )
    .bind(user.user_id, week)
    .first<PersonalTimer>();

  if (existing) {
    return c.json({
      status: 'already_started',
      week,
      user_id: user.user_id,
      started_at: existing.started_at,
      message: 'Personal timer was already started',
    });
  }

  const startedAt = new Date().toISOString();
  await c.env.DB.prepare(
    'INSERT INTO personal_timers (user_id, week, started_at, status) VALUES (?, ?, ?, ?)',
  )
    .bind(user.user_id, week, startedAt, 'in_progress')
    .run();

  return c.json({
    status: 'started',
    week,
    user_id: user.user_id,
    started_at: startedAt,
  });
});

// ---------- GET /api/challenge/:week/my-status ----------
app.get('/api/challenge/:week/my-status', async (c) => {
  const user = requireAuth(c);
  const week = parseWeek(c.req.param('week'));
  if (week === null) {
    return c.json({ detail: WEEK_VALIDATION_ERROR }, 400);
  }

  const ch = await c.env.DB.prepare('SELECT * FROM challenges WHERE week = ?')
    .bind(week)
    .first<Challenge>();

  const response: Record<string, unknown> = {
    week,
    challenge_status: ch?.status ?? 'not_started',
    personal_status: 'not_started',
    personal_start_time: null,
    elapsed_seconds: null,
  };

  const timer = await c.env.DB.prepare(
    'SELECT * FROM personal_timers WHERE user_id = ? AND week = ?',
  )
    .bind(user.user_id, week)
    .first<PersonalTimer>();

  if (timer) {
    response.personal_status = timer.status;
    response.personal_start_time = timer.started_at;

    if (timer.status === 'in_progress') {
      const start = new Date(timer.started_at).getTime();
      const now = Date.now();
      response.elapsed_seconds = Math.round((now - start) / 1000);
    }
  }

  return c.json(response);
});

export default app;
