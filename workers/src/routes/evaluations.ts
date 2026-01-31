import { Hono } from 'hono';
import type { Env } from '../types';

const app = new Hono<{ Bindings: Env }>();

// ---------- GET /api/evaluations/:week/:pid ----------
app.get('/api/evaluations/:week/:pid', async (c) => {
  const week = parseInt(c.req.param('week'), 10);
  const pid = c.req.param('pid');

  // Get the latest completed evaluation for this user/week
  const ev = await c.env.DB.prepare(
    `SELECT * FROM evaluations
     WHERE user_id = ? AND week = ?
     ORDER BY submission_number DESC
     LIMIT 1`,
  )
    .bind(pid, week)
    .first();

  if (!ev || (ev as Record<string, unknown>).status === 'pending_review') {
    return c.json({
      participant: pid,
      week,
      status: 'pending_review',
      submission_number: ev ? (ev as Record<string, unknown>).submission_number : null,
    });
  }

  const e = ev as Record<string, unknown>;

  // Parse JSON fields
  let strengths: unknown = e.strengths;
  let improvements: unknown = e.improvements;
  try {
    if (typeof strengths === 'string') strengths = JSON.parse(strengths);
  } catch { /* keep as string */ }
  try {
    if (typeof improvements === 'string') improvements = JSON.parse(improvements);
  } catch { /* keep as string */ }

  return c.json({
    participant: pid,
    week,
    status: e.status,
    scores: {
      rubric: e.rubric_score,
      time_rank: e.time_rank,
      time_rank_bonus: e.time_rank_bonus,
      total: e.total_score,
    },
    breakdown: {},
    feedback: e.feedback || '',
    strengths: strengths || [],
    improvements: improvements || [],
    submission_number: e.submission_number,
    evaluated_at: e.evaluated_at,
  });
});

export default app;
