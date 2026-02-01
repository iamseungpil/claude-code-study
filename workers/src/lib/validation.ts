import type { Challenge } from '../types';

/** Validates that a week number is between 1 and 5 inclusive. */
export function isValidWeek(week: number): boolean {
  return !isNaN(week) && week >= 1 && week <= 5;
}

/** Parses a week string param and returns the number, or null if invalid. */
export function parseWeek(weekParam: string): number | null {
  const week = parseInt(weekParam, 10);
  return isValidWeek(week) ? week : null;
}

/** Week validation error response body. */
export const WEEK_VALIDATION_ERROR = 'Week must be between 1 and 5';

/** Participant ID pattern: 3-30 chars, letters/numbers/underscore/dash. */
export const PARTICIPANT_ID_RE = /^[a-zA-Z0-9_-]{3,30}$/;

/** Name pattern: letters (Latin + Korean) and spaces, requires at least one letter. */
export const NAME_RE = /^(?=.*[a-zA-Z\uAC00-\uD7A3])[a-zA-Z\uAC00-\uD7A3\s]{1,50}$/;

/**
 * Checks that a challenge is in 'started' state.
 * Returns an error response body + status, or null if challenge is active.
 */
export function checkChallengeActive(
  ch: Challenge | null,
  endedMessage = 'Challenge has ended.',
): { detail: string; status: 400 } | null {
  if (!ch || ch.status === 'not_started') {
    return { detail: 'Challenge has not started yet. Wait for admin to start.', status: 400 };
  }
  if (ch.status === 'ended') {
    return { detail: endedMessage, status: 400 };
  }
  return null;
}
