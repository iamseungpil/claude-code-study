import { Hono } from 'hono';
import type { Env, AppVariables, Challenge, PersonalTimer } from '../types';
import { parseWeek, WEEK_VALIDATION_ERROR } from '../lib/validation';

const app = new Hono<{ Bindings: Env; Variables: AppVariables }>();

// ---------- GET /api/challenges/status ----------
app.get('/api/challenges/status', async (c) => {
  const { results: challenges } = await c.env.DB.prepare(
    'SELECT * FROM challenges ORDER BY week',
  ).all<Challenge>();

  const { results: timers } = await c.env.DB.prepare(
    'SELECT * FROM personal_timers',
  ).all<PersonalTimer>();

  // Build the same shape as the old challenges.json
  const out: Record<string, unknown> = {};
  for (const ch of challenges) {
    const weekTimers = timers.filter((t) => t.week === ch.week);
    const personalStarts: Record<string, { started_at: string; status: string }> = {};
    for (const t of weekTimers) {
      personalStarts[t.user_id] = { started_at: t.started_at, status: t.status };
    }
    out[`week${ch.week}`] = {
      week: ch.week,
      status: ch.status,
      start_time: ch.start_time,
      end_time: ch.end_time,
      started_by: ch.started_by,
      personal_starts: personalStarts,
    };
  }

  return c.json(out);
});

// ---------- GET /api/challenge/:week/status ----------
app.get('/api/challenge/:week/status', async (c) => {
  const week = parseWeek(c.req.param('week'));
  if (week === null) {
    return c.json({ detail: WEEK_VALIDATION_ERROR }, 400);
  }

  const ch = await c.env.DB.prepare('SELECT * FROM challenges WHERE week = ?')
    .bind(week)
    .first<Challenge>();

  if (!ch) {
    return c.json({ detail: 'Challenge not found' }, 404);
  }

  return c.json({ week: ch.week, status: ch.status, start_time: ch.start_time, end_time: ch.end_time, started_by: ch.started_by });
});

export default app;
