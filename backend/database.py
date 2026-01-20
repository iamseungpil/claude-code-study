#!/usr/bin/env python3
"""
SQLite Database Module for Claude Code Study

Provides database connection, schema creation, and CRUD operations
for users and submissions.
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

# Database path
BASE_DIR = Path(__file__).parent.parent
DB_PATH = BASE_DIR / "data" / "claude_study.db"


def get_connection() -> sqlite3.Connection:
    """Get a database connection with row factory."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def get_db():
    """Context manager for database connections."""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Initialize database with schema."""
    with get_db() as conn:
        cursor = conn.cursor()

        # Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                first_name TEXT,
                last_name TEXT,
                profile_image TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Submissions table (user/week/try)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS submissions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                week INTEGER NOT NULL,
                submission_number INTEGER NOT NULL,
                github_url TEXT,
                submitted_at TIMESTAMP,
                elapsed_seconds INTEGER,
                elapsed_minutes REAL,
                personal_start_time TIMESTAMP,
                rubric_score INTEGER,
                time_rank INTEGER,
                time_rank_bonus INTEGER,
                total_score INTEGER,
                evaluation_status TEXT DEFAULT 'pending',
                feedback TEXT,
                breakdown TEXT,
                strengths TEXT,
                improvements TEXT,
                evaluated_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, week, submission_number),
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)

        # Indexes for fast queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_submissions_user_week
            ON submissions(user_id, week)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_submissions_elapsed
            ON submissions(week, elapsed_minutes)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_submissions_total_score
            ON submissions(week, total_score DESC)
        """)

        print(f"Database initialized at {DB_PATH}")


# ============== User CRUD ==============

def create_user(user_id: str, email: str, first_name: str = None, last_name: str = None) -> bool:
    """Create a new user."""
    with get_db() as conn:
        try:
            conn.execute(
                "INSERT INTO users (id, email, first_name, last_name) VALUES (?, ?, ?, ?)",
                (user_id, email, first_name, last_name)
            )
            return True
        except sqlite3.IntegrityError:
            return False


def get_user(user_id: str) -> Optional[Dict]:
    """Get user by ID."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        return dict(row) if row else None


def get_user_by_email(email: str) -> Optional[Dict]:
    """Get user by email."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE email = ?", (email,)
        ).fetchone()
        return dict(row) if row else None


# ============== Submission CRUD ==============

def create_submission(
    user_id: str,
    week: int,
    submission_number: int,
    github_url: str,
    submitted_at: str,
    elapsed_seconds: int,
    elapsed_minutes: float,
    personal_start_time: str = None
) -> int:
    """Create a new submission. Returns submission ID."""
    with get_db() as conn:
        cursor = conn.execute(
            """INSERT INTO submissions
               (user_id, week, submission_number, github_url, submitted_at,
                elapsed_seconds, elapsed_minutes, personal_start_time, evaluation_status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending')
               ON CONFLICT(user_id, week, submission_number) DO UPDATE SET
                 github_url = excluded.github_url,
                 submitted_at = excluded.submitted_at,
                 elapsed_seconds = excluded.elapsed_seconds,
                 elapsed_minutes = excluded.elapsed_minutes
            """,
            (user_id, week, submission_number, github_url, submitted_at,
             elapsed_seconds, elapsed_minutes, personal_start_time)
        )
        return cursor.lastrowid


def update_submission_evaluation(
    user_id: str,
    week: int,
    submission_number: int,
    rubric_score: int,
    time_rank: int,
    time_rank_bonus: int,
    total_score: int,
    feedback: str = None,
    breakdown: Dict = None,
    strengths: List[str] = None,
    improvements: List[str] = None,
    status: str = "completed"
) -> bool:
    """Update submission with evaluation results."""
    with get_db() as conn:
        cursor = conn.execute(
            """UPDATE submissions SET
                 rubric_score = ?,
                 time_rank = ?,
                 time_rank_bonus = ?,
                 total_score = ?,
                 feedback = ?,
                 breakdown = ?,
                 strengths = ?,
                 improvements = ?,
                 evaluation_status = ?,
                 evaluated_at = ?
               WHERE user_id = ? AND week = ? AND submission_number = ?
            """,
            (rubric_score, time_rank, time_rank_bonus, total_score,
             feedback,
             json.dumps(breakdown) if breakdown else None,
             json.dumps(strengths) if strengths else None,
             json.dumps(improvements) if improvements else None,
             status,
             datetime.now().isoformat(),
             user_id, week, submission_number)
        )
        return cursor.rowcount > 0


def get_submission(user_id: str, week: int, submission_number: int = None) -> Optional[Dict]:
    """Get a specific submission or the latest one for a user/week."""
    with get_db() as conn:
        if submission_number:
            row = conn.execute(
                """SELECT * FROM submissions
                   WHERE user_id = ? AND week = ? AND submission_number = ?""",
                (user_id, week, submission_number)
            ).fetchone()
        else:
            # Get latest submission
            row = conn.execute(
                """SELECT * FROM submissions
                   WHERE user_id = ? AND week = ?
                   ORDER BY submission_number DESC LIMIT 1""",
                (user_id, week)
            ).fetchone()

        if row:
            result = dict(row)
            # Parse JSON fields
            for field in ['breakdown', 'strengths', 'improvements']:
                if result.get(field):
                    try:
                        result[field] = json.loads(result[field])
                    except json.JSONDecodeError:
                        pass
            return result
        return None


def get_submission_history(user_id: str, week: int) -> List[Dict]:
    """Get all submissions for a user/week."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT * FROM submissions
               WHERE user_id = ? AND week = ?
               ORDER BY submission_number ASC""",
            (user_id, week)
        ).fetchall()

        result = []
        for row in rows:
            item = dict(row)
            for field in ['breakdown', 'strengths', 'improvements']:
                if item.get(field):
                    try:
                        item[field] = json.loads(item[field])
                    except json.JSONDecodeError:
                        pass
            result.append(item)
        return result


def get_submission_count(user_id: str, week: int) -> int:
    """Get the number of submissions for a user/week."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT COUNT(*) as count FROM submissions WHERE user_id = ? AND week = ?",
            (user_id, week)
        ).fetchone()
        return row['count'] if row else 0


# ============== Time Rank Queries ==============

def get_time_rank_by_elapsed(week: int, user_id: str) -> int:
    """
    Get time rank based on elapsed_minutes (shorter = better rank).
    Only considers the LATEST submission for each user.

    Returns rank (1-based, where 1 = shortest elapsed time)
    """
    with get_db() as conn:
        # Get all users' latest submissions with elapsed_minutes, ordered by elapsed time
        rows = conn.execute(
            """
            SELECT user_id, elapsed_minutes
            FROM submissions s1
            WHERE week = ?
              AND submission_number = (
                  SELECT MAX(submission_number)
                  FROM submissions s2
                  WHERE s2.user_id = s1.user_id AND s2.week = s1.week
              )
              AND elapsed_minutes IS NOT NULL
            ORDER BY elapsed_minutes ASC
            """,
            (week,)
        ).fetchall()

        # Find rank for the given user
        for rank, row in enumerate(rows, start=1):
            if row['user_id'] == user_id:
                return rank

        return 999  # Not found


def get_all_time_ranks(week: int) -> Dict[str, int]:
    """
    Get time ranks for all users in a week.
    Returns dict of {user_id: rank}
    """
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT user_id, elapsed_minutes
            FROM submissions s1
            WHERE week = ?
              AND submission_number = (
                  SELECT MAX(submission_number)
                  FROM submissions s2
                  WHERE s2.user_id = s1.user_id AND s2.week = s1.week
              )
              AND elapsed_minutes IS NOT NULL
            ORDER BY elapsed_minutes ASC
            """,
            (week,)
        ).fetchall()

        return {row['user_id']: rank for rank, row in enumerate(rows, start=1)}


# ============== Leaderboard Queries ==============

def get_week_leaderboard(week: int) -> List[Dict]:
    """
    Get leaderboard for a specific week.
    Uses latest submission's score for each user.
    """
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT
                s.user_id,
                s.total_score,
                s.rubric_score,
                s.time_rank,
                s.time_rank_bonus,
                s.elapsed_minutes,
                s.evaluation_status,
                u.first_name,
                u.last_name
            FROM submissions s
            JOIN users u ON s.user_id = u.id
            WHERE s.week = ?
              AND s.submission_number = (
                  SELECT MAX(submission_number)
                  FROM submissions s2
                  WHERE s2.user_id = s.user_id AND s2.week = s.week
              )
              AND s.evaluation_status = 'completed'
            ORDER BY s.total_score DESC
            """,
            (week,)
        ).fetchall()

        result = []
        for rank, row in enumerate(rows, start=1):
            item = dict(row)
            item['rank'] = rank
            item['participant'] = row['user_id']
            result.append(item)
        return result


def get_season_leaderboard() -> List[Dict]:
    """
    Get cumulative leaderboard across all weeks.
    Sums total_score from latest submission of each week.
    """
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT
                s.user_id,
                SUM(s.total_score) as total_score,
                COUNT(DISTINCT s.week) as weeks_completed,
                u.first_name,
                u.last_name
            FROM submissions s
            JOIN users u ON s.user_id = u.id
            WHERE s.submission_number = (
                SELECT MAX(submission_number)
                FROM submissions s2
                WHERE s2.user_id = s.user_id AND s2.week = s.week
            )
            AND s.evaluation_status = 'completed'
            GROUP BY s.user_id
            ORDER BY total_score DESC
            """
        ).fetchall()

        result = []
        for rank, row in enumerate(rows, start=1):
            item = dict(row)
            item['rank'] = rank
            item['participant'] = row['user_id']
            result.append(item)
        return result


# Initialize database on import
if __name__ == "__main__":
    init_db()
    print("Database initialized successfully!")
