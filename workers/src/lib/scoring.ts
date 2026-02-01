/**
 * Time rank bonus calculation.
 * 1st: +20, 2nd: +17, 3rd: +14, 4th: +11, 5th: +8, 6th+: +5
 */
const RANK_POINTS: Record<number, number> = {
  1: 20,
  2: 17,
  3: 14,
  4: 11,
  5: 8,
};

export function calculateTimeRankBonus(rank: number): number {
  return RANK_POINTS[rank] ?? 5;
}

/** Season points awarded based on weekly leaderboard rank. */
const SEASON_RANK_POINTS: Record<number, number> = {
  1: 10,
  2: 7,
  3: 5,
};
const SEASON_DEFAULT_POINTS = 3;

export function calculateSeasonPoints(rank: number): number {
  return SEASON_RANK_POINTS[rank] ?? SEASON_DEFAULT_POINTS;
}

/** Sentinel value returned when a user has no ranked submission for the week. */
const UNRANKED_POSITION = 999;

/**
 * Get time rank for a user in a given week.
 * Considers only the LATEST submission for each user, ordered by elapsed_minutes ASC.
 */
export async function getTimeRank(db: D1Database, week: number, userId: string): Promise<number> {
  const { results } = await db
    .prepare(
      `SELECT user_id, elapsed_minutes
       FROM submissions s1
       WHERE week = ?
         AND submission_number = (
             SELECT MAX(submission_number)
             FROM submissions s2
             WHERE s2.user_id = s1.user_id AND s2.week = s1.week
         )
         AND elapsed_minutes IS NOT NULL
       ORDER BY elapsed_minutes ASC`,
    )
    .bind(week)
    .all();

  for (let i = 0; i < results.length; i++) {
    if ((results[i] as { user_id: string }).user_id === userId) {
      return i + 1;
    }
  }
  return UNRANKED_POSITION;
}
