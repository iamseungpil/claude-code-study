"""
Firestore Client for Claude Code Study
Handles all database operations with Google Cloud Firestore
"""

import os
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from google.cloud import firestore

# Initialize Firestore client
# In Cloud Run, authentication is automatic via service account
db = firestore.Client()

# Collection names
USERS_COLLECTION = "users"
PARTICIPANTS_COLLECTION = "participants"
CHALLENGES_COLLECTION = "challenges"
EVALUATIONS_COLLECTION = "evaluations"
SUBMISSIONS_COLLECTION = "submissions"


# ============== Users ==============

def get_user(user_id: str) -> Optional[dict]:
    """Get user by user_id."""
    doc = db.collection(USERS_COLLECTION).document(user_id.lower()).get()
    if doc.exists:
        return doc.to_dict()
    return None


def get_user_by_full_name(full_name: str) -> Optional[dict]:
    """Get user by full name."""
    docs = db.collection(USERS_COLLECTION).where("full_name_lower", "==", full_name.lower()).limit(1).stream()
    for doc in docs:
        return doc.to_dict()
    return None


def create_user(user_data: dict) -> dict:
    """Create a new user."""
    user_id = user_data["user_id"].lower()
    user_data["full_name_lower"] = user_data["full_name"].lower()
    user_data["created_at"] = datetime.now(timezone.utc).isoformat()

    db.collection(USERS_COLLECTION).document(user_id).set(user_data)
    return user_data


def delete_user(user_id: str) -> bool:
    """Delete a user."""
    doc_ref = db.collection(USERS_COLLECTION).document(user_id.lower())
    if doc_ref.get().exists:
        doc_ref.delete()
        return True
    return False


def list_users() -> List[dict]:
    """List all users."""
    docs = db.collection(USERS_COLLECTION).stream()
    return [doc.to_dict() for doc in docs]


# ============== Participants ==============

def get_participant(participant_id: str) -> Optional[dict]:
    """Get participant by ID."""
    doc = db.collection(PARTICIPANTS_COLLECTION).document(participant_id).get()
    if doc.exists:
        return doc.to_dict()
    return None


def create_participant(participant_data: dict) -> dict:
    """Create a new participant."""
    participant_id = participant_data["id"]
    participant_data["registered_at"] = datetime.now(timezone.utc).isoformat()

    db.collection(PARTICIPANTS_COLLECTION).document(participant_id).set(participant_data)
    return participant_data


def list_participants() -> List[dict]:
    """List all participants."""
    docs = db.collection(PARTICIPANTS_COLLECTION).stream()
    return [doc.to_dict() for doc in docs]


# ============== Challenges ==============

def get_challenge(week: int) -> dict:
    """Get challenge status for a specific week."""
    doc = db.collection(CHALLENGES_COLLECTION).document(f"week{week}").get()
    if doc.exists:
        return doc.to_dict()
    return {
        "status": "not_started",
        "start_time": None,
        "end_time": None,
        "personal_starts": {}
    }


def get_all_challenges() -> dict:
    """Get all challenges status."""
    challenges = {}
    for week in range(1, 6):
        challenges[f"week{week}"] = get_challenge(week)
    return challenges


def update_challenge(week: int, data: dict):
    """Update challenge data for a specific week."""
    db.collection(CHALLENGES_COLLECTION).document(f"week{week}").set(data, merge=True)


def start_challenge(week: int, admin_user_id: str) -> dict:
    """Start a challenge (admin only)."""
    start_time = datetime.now(timezone.utc).isoformat()
    data = {
        "status": "started",
        "start_time": start_time,
        "end_time": None,
        "started_by": admin_user_id,
        "personal_starts": {}
    }
    update_challenge(week, data)
    return data


def end_challenge(week: int) -> dict:
    """End a challenge (admin only)."""
    end_time = datetime.now(timezone.utc).isoformat()
    challenge = get_challenge(week)
    challenge["status"] = "ended"
    challenge["end_time"] = end_time
    update_challenge(week, challenge)
    return challenge


def start_personal_timer(week: int, user_id: str) -> dict:
    """Start personal timer for a user."""
    challenge = get_challenge(week)

    if "personal_starts" not in challenge:
        challenge["personal_starts"] = {}

    # Check if already started
    if user_id in challenge["personal_starts"]:
        return {
            "status": "already_started",
            "started_at": challenge["personal_starts"][user_id]["started_at"]
        }

    started_at = datetime.now(timezone.utc).isoformat()
    challenge["personal_starts"][user_id] = {
        "started_at": started_at,
        "status": "in_progress"
    }

    update_challenge(week, challenge)
    return {"status": "started", "started_at": started_at}


def update_personal_status(week: int, user_id: str, status: str):
    """Update personal status (e.g., to 'submitted')."""
    challenge = get_challenge(week)
    if user_id in challenge.get("personal_starts", {}):
        challenge["personal_starts"][user_id]["status"] = status
        update_challenge(week, challenge)


def restart_challenge(week: int) -> dict:
    """
    Restart a challenge by clearing all data and resetting to initial state.
    - Deletes all submissions for the week
    - Deletes all evaluations for the week
    - Resets challenge document to not_started state
    Returns counts of deleted documents.
    """
    deleted_submissions = 0
    deleted_evaluations = 0

    # Delete all submissions for this week
    submissions_query = db.collection(SUBMISSIONS_COLLECTION).where("week", "==", week)
    for doc in submissions_query.stream():
        doc.reference.delete()
        deleted_submissions += 1

    # Delete all evaluations for this week
    evaluations_query = db.collection(EVALUATIONS_COLLECTION).where("week", "==", week)
    for doc in evaluations_query.stream():
        doc.reference.delete()
        deleted_evaluations += 1

    # Reset challenge document to initial state
    initial_state = {
        "status": "not_started",
        "start_time": None,
        "end_time": None,
        "started_by": None,
        "personal_starts": {}
    }
    db.collection(CHALLENGES_COLLECTION).document(f"week{week}").set(initial_state)

    return {
        "week": week,
        "deleted_submissions": deleted_submissions,
        "deleted_evaluations": deleted_evaluations,
        "status": "restarted"
    }


# ============== Submissions ==============

def save_submission_metadata(week: int, participant_id: str, metadata: dict):
    """Save submission metadata."""
    doc_id = f"week{week}_{participant_id}"
    metadata["week"] = week
    metadata["participant_id"] = participant_id
    db.collection(SUBMISSIONS_COLLECTION).document(doc_id).set(metadata)


def get_submission_metadata(week: int, participant_id: str) -> Optional[dict]:
    """Get submission metadata."""
    doc_id = f"week{week}_{participant_id}"
    doc = db.collection(SUBMISSIONS_COLLECTION).document(doc_id).get()
    if doc.exists:
        return doc.to_dict()
    return None


def list_submissions(week: int) -> List[dict]:
    """List all submissions for a week."""
    docs = db.collection(SUBMISSIONS_COLLECTION).where("week", "==", week).stream()
    return [doc.to_dict() for doc in docs]


def get_submission_rank(week: int, participant_id: str) -> int:
    """Get submission rank based on submission time."""
    submissions = list_submissions(week)

    # Sort by submission time
    submissions.sort(key=lambda x: x.get("submitted_at", "9999"))

    for rank, sub in enumerate(submissions, start=1):
        if sub.get("participant_id") == participant_id:
            return rank

    return 999


# ============== Evaluations ==============

def save_evaluation(week: int, participant_id: str, result: dict):
    """Save evaluation result."""
    doc_id = f"week{week}_{participant_id}"
    result["week"] = week
    result["participant_id"] = participant_id
    db.collection(EVALUATIONS_COLLECTION).document(doc_id).set(result)


def get_evaluation(week: int, participant_id: str) -> Optional[dict]:
    """Get evaluation result."""
    doc_id = f"week{week}_{participant_id}"
    doc = db.collection(EVALUATIONS_COLLECTION).document(doc_id).get()
    if doc.exists:
        return doc.to_dict()
    return None


def list_evaluations(week: int) -> List[dict]:
    """List all evaluations for a week."""
    docs = db.collection(EVALUATIONS_COLLECTION).where("week", "==", week).stream()
    return [doc.to_dict() for doc in docs]


def get_week_leaderboard(week: int) -> List[dict]:
    """Get leaderboard for a specific week."""
    evaluations = list_evaluations(week)

    results = []
    for data in evaluations:
        if data.get("status") == "completed":
            scores = data.get("scores", {})
            results.append({
                "participant_id": data.get("participant_id") or data.get("participant"),
                "total": scores.get("total", 0),
                "rubric": scores.get("rubric", 0),
                "time_rank": scores.get("time_rank", 0),
                "time_rank_bonus": scores.get("time_rank_bonus", 0),
                "evaluated_at": data.get("evaluated_at")
            })

    # Sort by total score descending
    results.sort(key=lambda x: x["total"], reverse=True)

    # Add rank and medals
    for i, r in enumerate(results):
        r["rank"] = i + 1
        if i == 0:
            r["medal"] = "🥇"
        elif i == 1:
            r["medal"] = "🥈"
        elif i == 2:
            r["medal"] = "🥉"
        else:
            r["medal"] = ""

    return results


def get_season_leaderboard() -> List[dict]:
    """Get overall season leaderboard."""
    season_scores = {}

    for week in range(1, 6):
        week_leaderboard = get_week_leaderboard(week)

        for entry in week_leaderboard:
            pid = entry["participant_id"]
            if pid not in season_scores:
                season_scores[pid] = {
                    "participant_id": pid,
                    "total_points": 0,
                    "weeks_completed": 0,
                    "weekly_scores": {}
                }

            # Weekly ranking points
            rank = entry["rank"]
            if rank == 1:
                points = 10
            elif rank == 2:
                points = 7
            elif rank == 3:
                points = 5
            else:
                points = 3

            season_scores[pid]["total_points"] += points
            season_scores[pid]["weeks_completed"] += 1
            season_scores[pid]["weekly_scores"][f"week{week}"] = {
                "rank": rank,
                "points": points,
                "score": entry["total"]
            }

    # Sort by total points
    results = list(season_scores.values())
    results.sort(key=lambda x: x["total_points"], reverse=True)

    # Add season rank
    for i, r in enumerate(results):
        r["season_rank"] = i + 1
        if i == 0:
            r["title"] = "🎸 Master of Vibe Coding"
        elif i == 1:
            r["title"] = "🥈 Runner-up"
        elif i == 2:
            r["title"] = "🥉 3rd Place"
        else:
            r["title"] = ""

    return results
