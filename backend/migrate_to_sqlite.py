#!/usr/bin/env python3
"""
Migration Script: JSON files to SQLite

Migrates existing data from:
- data/users.json
- submissions/weekN/{participant_id}/metadata.json
- evaluations/weekN/{participant_id}.json

to SQLite database.
"""

import json
from pathlib import Path
from datetime import datetime

from database import (
    init_db, get_db, DB_PATH,
    create_user, get_user,
    create_submission, update_submission_evaluation
)

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
SUBMISSIONS_DIR = BASE_DIR / "submissions"
EVALUATIONS_DIR = BASE_DIR / "evaluations"


def migrate_users():
    """Migrate users from data/users.json"""
    users_file = DATA_DIR / "users.json"

    if not users_file.exists():
        print("No users.json found, skipping user migration")
        return 0

    with open(users_file) as f:
        data = json.load(f)

    # Handle both formats: {"users": [...]} or {user_id: {...}}
    if isinstance(data, dict) and "users" in data:
        users_list = data["users"]
    elif isinstance(data, list):
        users_list = data
    else:
        # Old format: {user_id: user_data}
        users_list = [{"user_id": uid, **udata} for uid, udata in data.items()]

    migrated = 0
    for user_data in users_list:
        user_id = user_data.get("user_id")
        if not user_id:
            continue

        if get_user(user_id):
            print(f"  User {user_id} already exists, skipping")
            continue

        # Generate email from user_id if not present
        email = user_data.get("email", f"{user_id}@claude-study.local")
        first_name = user_data.get("first_name")
        last_name = user_data.get("last_name")

        if create_user(user_id, email, first_name, last_name):
            print(f"  Migrated user: {user_id}")
            migrated += 1

    return migrated


def migrate_submissions_and_evaluations():
    """Migrate submissions and evaluations from JSON files"""
    migrated_submissions = 0
    migrated_evaluations = 0

    # Process each week
    for week in range(1, 6):
        submissions_dir = SUBMISSIONS_DIR / f"week{week}"
        evaluations_dir = EVALUATIONS_DIR / f"week{week}"

        if not submissions_dir.exists():
            continue

        print(f"\nProcessing Week {week}...")

        # Process each participant
        for participant_dir in submissions_dir.iterdir():
            if not participant_dir.is_dir():
                continue

            user_id = participant_dir.name
            metadata_file = participant_dir / "metadata.json"

            if not metadata_file.exists():
                continue

            with open(metadata_file) as f:
                metadata = json.load(f)

            # Get submission history
            history = metadata.get("submission_history", [])

            if not history:
                # Create single submission from metadata
                history = [{
                    "submission_number": 1,
                    "github_url": metadata.get("github_url"),
                    "submitted_at": metadata.get("submitted_at"),
                    "elapsed_seconds": metadata.get("elapsed_seconds"),
                    "elapsed_minutes": metadata.get("elapsed_minutes"),
                }]

            personal_start_time = metadata.get("personal_start_time")

            # Load evaluation if exists
            eval_data = None
            eval_file = evaluations_dir / f"{user_id}.json"
            if eval_file.exists():
                with open(eval_file) as f:
                    eval_data = json.load(f)

            # Migrate each submission
            for sub in history:
                submission_num = sub.get("submission_number", 1)

                # Create submission
                create_submission(
                    user_id=user_id,
                    week=week,
                    submission_number=submission_num,
                    github_url=sub.get("github_url"),
                    submitted_at=sub.get("submitted_at"),
                    elapsed_seconds=sub.get("elapsed_seconds"),
                    elapsed_minutes=sub.get("elapsed_minutes"),
                    personal_start_time=personal_start_time
                )
                migrated_submissions += 1
                print(f"    Submission: {user_id} week{week} try{submission_num}")

                # Check for inline evaluation in submission history
                sub_eval = sub.get("evaluation")

                # For the latest submission, use eval_data if no inline evaluation
                is_latest = submission_num == max(s.get("submission_number", 1) for s in history)

                if sub_eval and sub_eval.get("status") == "completed":
                    update_submission_evaluation(
                        user_id=user_id,
                        week=week,
                        submission_number=submission_num,
                        rubric_score=sub_eval.get("rubric", 0),
                        time_rank=sub_eval.get("time_rank", 999),
                        time_rank_bonus=sub_eval.get("time_rank_bonus", 0),
                        total_score=sub_eval.get("total", 0),
                        status="completed"
                    )
                    migrated_evaluations += 1
                    print(f"      Evaluation (inline): score={sub_eval.get('total', 0)}")

                elif is_latest and eval_data and eval_data.get("status") == "completed":
                    scores = eval_data.get("scores", {})
                    update_submission_evaluation(
                        user_id=user_id,
                        week=week,
                        submission_number=submission_num,
                        rubric_score=scores.get("rubric", 0),
                        time_rank=scores.get("time_rank", 999),
                        time_rank_bonus=scores.get("time_rank_bonus", 0),
                        total_score=scores.get("total", 0),
                        feedback=eval_data.get("feedback"),
                        breakdown=eval_data.get("breakdown"),
                        strengths=eval_data.get("strengths"),
                        improvements=eval_data.get("improvements"),
                        status="completed"
                    )
                    migrated_evaluations += 1
                    print(f"      Evaluation (file): score={scores.get('total', 0)}")

    return migrated_submissions, migrated_evaluations


def verify_migration():
    """Verify migration by printing stats"""
    with get_db() as conn:
        user_count = conn.execute("SELECT COUNT(*) as c FROM users").fetchone()['c']
        sub_count = conn.execute("SELECT COUNT(*) as c FROM submissions").fetchone()['c']
        eval_count = conn.execute(
            "SELECT COUNT(*) as c FROM submissions WHERE evaluation_status = 'completed'"
        ).fetchone()['c']

        print(f"\n{'='*50}")
        print("Migration Verification:")
        print(f"  Users: {user_count}")
        print(f"  Submissions: {sub_count}")
        print(f"  Evaluations: {eval_count}")
        print(f"{'='*50}")

        # Show sample data
        print("\nSample submissions:")
        rows = conn.execute("""
            SELECT user_id, week, submission_number, elapsed_minutes, total_score
            FROM submissions
            ORDER BY week, user_id, submission_number
            LIMIT 10
        """).fetchall()

        for row in rows:
            print(f"  {row['user_id']} week{row['week']} try{row['submission_number']}: "
                  f"elapsed={row['elapsed_minutes']}min, score={row['total_score']}")


def main():
    print("=" * 50)
    print("Claude Code Study - JSON to SQLite Migration")
    print("=" * 50)

    # Initialize database
    print("\n1. Initializing database...")
    init_db()

    # Migrate users
    print("\n2. Migrating users...")
    user_count = migrate_users()
    print(f"   Migrated {user_count} users")

    # Migrate submissions and evaluations
    print("\n3. Migrating submissions and evaluations...")
    sub_count, eval_count = migrate_submissions_and_evaluations()
    print(f"   Migrated {sub_count} submissions, {eval_count} evaluations")

    # Verify
    verify_migration()

    print(f"\nMigration complete! Database at: {DB_PATH}")


if __name__ == "__main__":
    main()
