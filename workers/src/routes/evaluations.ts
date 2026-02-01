import { Hono } from 'hono';
import type { Env, AppVariables, Evaluation } from '../types';
import { parseWeek, WEEK_VALIDATION_ERROR } from '../lib/validation';

const app = new Hono<{ Bindings: Env; Variables: AppVariables }>();

// ---------- GET /api/evaluations/:week/:pid ----------
app.get('/api/evaluations/:week/:pid', async (c) => {
  const week = parseWeek(c.req.param('week'));
  if (week === null) {
    return c.json({ detail: WEEK_VALIDATION_ERROR }, 400);
  }
  const pid = c.req.param('pid');

  // Get the latest completed evaluation for this user/week
  const ev = await c.env.DB.prepare(
    `SELECT * FROM evaluations
     WHERE user_id = ? AND week = ?
     ORDER BY submission_number DESC
     LIMIT 1`,
  )
    .bind(pid, week)
    .first<Evaluation>();

  if (!ev || ev.status === 'pending_review') {
    return c.json({
      participant: pid,
      week,
      status: 'pending_review',
      submission_number: ev ? ev.submission_number : null,
    });
  }

  // Parse JSON fields
  let strengths: unknown = ev.strengths;
  let improvements: unknown = ev.improvements;
  let breakdown: unknown = ev.breakdown;
  try {
    if (typeof strengths === 'string') strengths = JSON.parse(strengths);
  } catch { /* keep as string */ }
  try {
    if (typeof improvements === 'string') improvements = JSON.parse(improvements);
  } catch { /* keep as string */ }
  try {
    if (typeof breakdown === 'string') breakdown = JSON.parse(breakdown);
  } catch { /* keep as string */ }

  return c.json({
    participant: pid,
    week,
    status: ev.status,
    scores: {
      rubric: ev.rubric_score,
      time_rank: ev.time_rank,
      time_rank_bonus: ev.time_rank_bonus,
      total: ev.total_score,
    },
    breakdown: breakdown || {},
    feedback: ev.feedback || '',
    strengths: strengths || [],
    improvements: improvements || [],
    submission_number: ev.submission_number,
    evaluated_at: ev.evaluated_at,
  });
});

export default app;
