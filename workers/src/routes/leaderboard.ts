import { Hono } from 'hono';
import type { AppVariables, Env, User } from '../types';
import { calculateSeasonPoints } from '../lib/scoring';
import { parseWeek, WEEK_VALIDATION_ERROR } from '../lib/validation';

const app = new Hono<{ Bindings: Env; Variables: AppVariables }>();

// Helper: load user map for enrichment
async function loadUserMap(db: D1Database): Promise<Map<string, { full_name: string; profile_image: string | null }>> {
  const { results } = await db.prepare('SELECT user_id, full_name, profile_image FROM users').all<User>();
  const map = new Map<string, { full_name: string; profile_image: string | null }>();
  for (const u of results) {
    map.set(u.user_id, { full_name: u.full_name, profile_image: u.profile_image });
  }
  return map;
}

// Helper: get week leaderboard data
async function getWeekLeaderboardData(
  db: D1Database,
  week: number,
  existingUserMap?: Map<string, { full_name: string; profile_image: string | null }>,
) {
  // Get the latest completed evaluation for each user in this week
  const { results } = await db
    .prepare(
      `SELECT e.user_id, e.rubric_score, e.time_rank, e.time_rank_bonus, e.total_score, e.evaluated_at
       FROM evaluations e
       WHERE e.week = ?
         AND e.status = 'completed'
         AND e.submission_number = (
           SELECT MAX(e2.submission_number)
           FROM evaluations e2
           WHERE e2.user_id = e.user_id AND e2.week = e.week AND e2.status = 'completed'
         )
       ORDER BY e.total_score DESC`,
    )
    .bind(week)
    .all();

  const userMap = existingUserMap ?? await loadUserMap(db);

  return results.map((row, i) => {
    const r = row as Record<string, unknown>;
    const pid = r.user_id as string;
    const userInfo = userMap.get(pid);
    return {
      rank: i + 1,
      medal: i === 0 ? '\u{1F947}' : i === 1 ? '\u{1F948}' : i === 2 ? '\u{1F949}' : '',
      participant_id: pid,
      full_name: userInfo?.full_name ?? pid,
      profile_image: userInfo?.profile_image ?? null,
      total: r.total_score,
      rubric: r.rubric_score,
      time_rank: r.time_rank,
      time_rank_bonus: r.time_rank_bonus,
      evaluated_at: r.evaluated_at,
    };
  });
}

// NOTE: /api/leaderboard/season MUST be defined BEFORE /api/leaderboard/:week
// to prevent "season" from matching as a week parameter.

// ---------- GET /api/leaderboard/season ----------
app.get('/api/leaderboard/season', async (c) => {
  const userMap = await loadUserMap(c.env.DB);

  const seasonScores = new Map<
    string,
    { participant_id: string; total_points: number; weeks_completed: number; weekly_scores: Record<string, unknown> }
  >();

  for (let week = 1; week <= 5; week++) {
    const leaderboard = await getWeekLeaderboardData(c.env.DB, week, userMap);

    for (const entry of leaderboard) {
      if (!seasonScores.has(entry.participant_id)) {
        seasonScores.set(entry.participant_id, {
          participant_id: entry.participant_id,
          total_points: 0,
          weeks_completed: 0,
          weekly_scores: {},
        });
      }

      const record = seasonScores.get(entry.participant_id)!;
      const rank = entry.rank;
      const points = calculateSeasonPoints(rank);

      record.total_points += points;
      record.weeks_completed += 1;
      record.weekly_scores[`week${week}`] = {
        rank,
        points,
        score: entry.total,
      };
    }
  }

  const results = [...seasonScores.values()].sort((a, b) => b.total_points - a.total_points);

  const enriched = results.map((r, i) => {
    const userInfo = userMap.get(r.participant_id);
    let title = '';
    if (i === 0) title = '\u{1F3B8} Master of Vibe Coding';
    else if (i === 1) title = '\u{1F948} Runner-up';
    else if (i === 2) title = '\u{1F949} 3rd Place';

    return {
      ...r,
      season_rank: i + 1,
      title,
      full_name: userInfo?.full_name ?? r.participant_id,
      profile_image: userInfo?.profile_image ?? null,
    };
  });

  return c.json(enriched);
});

// ---------- GET /api/leaderboard/:week ----------
app.get('/api/leaderboard/:week', async (c) => {
  const week = parseWeek(c.req.param('week'));
  if (week === null) {
    return c.json({ detail: WEEK_VALIDATION_ERROR }, 400);
  }
  const data = await getWeekLeaderboardData(c.env.DB, week);
  return c.json(data);
});

export default app;
