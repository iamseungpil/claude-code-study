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
    """
    Save submission metadata with history tracking.
    Supports multiple submissions per user per week.
    """
    doc_id = f"week{week}_{participant_id}"
    doc_ref = db.collection(SUBMISSIONS_COLLECTION).document(doc_id)
    doc = doc_ref.get()

    metadata["week"] = week
    metadata["participant_id"] = participant_id

    if doc.exists:
        # Existing submission - append to history
        existing = doc.to_dict()
        submission_history = existing.get("submission_history", [])

        # Move current submission to history if exists
        if "current_submission" in existing:
            prev = existing["current_submission"].copy()
            prev["status"] = "superseded"
            submission_history.append(prev)

        submission_number = existing.get("total_submissions", len(submission_history)) + 1

        # Create new document structure
        new_data = {
            "week": week,
            "participant_id": participant_id,
            "total_submissions": submission_number,
            "first_submitted_at": existing.get("first_submitted_at") or existing.get("submitted_at"),
            "latest_submitted_at": metadata.get("submitted_at"),
            "current_submission": {
                "submission_number": submission_number,
                "github_url": metadata.get("github_url"),
                "submitted_at": metadata.get("submitted_at"),
                "personal_start_time": metadata.get("personal_start_time"),
                "elapsed_seconds": metadata.get("elapsed_seconds"),
                "elapsed_minutes": metadata.get("elapsed_minutes"),
                "status": "submitted"
            },
            "submission_history": submission_history
        }
        doc_ref.set(new_data)
        return submission_number
    else:
        # First submission
        new_data = {
            "week": week,
            "participant_id": participant_id,
            "total_submissions": 1,
            "first_submitted_at": metadata.get("submitted_at"),
            "latest_submitted_at": metadata.get("submitted_at"),
            "current_submission": {
                "submission_number": 1,
                "github_url": metadata.get("github_url"),
                "submitted_at": metadata.get("submitted_at"),
                "personal_start_time": metadata.get("personal_start_time"),
                "elapsed_seconds": metadata.get("elapsed_seconds"),
                "elapsed_minutes": metadata.get("elapsed_minutes"),
                "status": "submitted"
            },
            "submission_history": []
        }
        doc_ref.set(new_data)
        return 1


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
    """
    Get submission rank based on LATEST submission time.
    Recalculates rank based on each user's most recent submission.
    """
    submissions = list_submissions(week)

    # Sort by latest submission time
    submissions.sort(key=lambda x: x.get("latest_submitted_at") or x.get("submitted_at", "9999"))

    for rank, sub in enumerate(submissions, start=1):
        if sub.get("participant_id") == participant_id:
            return rank

    return 999


# ============== Evaluations ==============

def save_evaluation(week: int, participant_id: str, result: dict):
    """
    Save evaluation result with history tracking.
    Tracks best rubric score across all submissions.
    Time rank bonus is recalculated based on latest submission order.
    """
    doc_id = f"week{week}_{participant_id}"
    doc_ref = db.collection(EVALUATIONS_COLLECTION).document(doc_id)
    doc = doc_ref.get()

    result["week"] = week
    result["participant_id"] = participant_id

    # Get submission number from submission metadata
    submission_meta = get_submission_metadata(week, participant_id)
    submission_number = 1
    if submission_meta:
        current_sub = submission_meta.get("current_submission", {})
        submission_number = current_sub.get("submission_number", submission_meta.get("total_submissions", 1))

    result["submission_number"] = submission_number

    if doc.exists and result.get("status") == "completed":
        # Existing evaluation - track history and best score
        existing = doc.to_dict()
        evaluation_history = existing.get("evaluation_history", [])

        # Move current evaluation to history if exists and was completed
        if "current_evaluation" in existing:
            prev = existing["current_evaluation"].copy()
            evaluation_history.append(prev)
        elif existing.get("status") == "completed":
            # Legacy format - save to history
            evaluation_history.append({
                "submission_number": existing.get("submission_number", 1),
                "scores": existing.get("scores", {}),
                "evaluated_at": existing.get("evaluated_at"),
                "status": "completed"
            })

        # Determine best rubric score (excluding time bonus)
        current_rubric = result.get("scores", {}).get("rubric", 0)
        best_rubric = existing.get("best_rubric_score", 0)

        # Check history for best rubric score
        for hist in evaluation_history:
            hist_rubric = hist.get("scores", {}).get("rubric", 0)
            if hist_rubric > best_rubric:
                best_rubric = hist_rubric

        if current_rubric > best_rubric:
            best_rubric = current_rubric

        new_data = {
            "week": week,
            "participant_id": participant_id,
            "status": "completed",
            "total_evaluations": len(evaluation_history) + 1,
            "best_rubric_score": best_rubric,
            "current_evaluation": {
                "submission_number": submission_number,
                "scores": result.get("scores", {}),
                "feedback": result.get("feedback", {}),
                "build_status": result.get("build_status"),
                "evaluated_at": result.get("evaluated_at"),
                "status": "completed"
            },
            "evaluation_history": evaluation_history,
            # Keep top-level fields for backward compatibility with leaderboard
            "scores": result.get("scores", {}),
            "feedback": result.get("feedback", {}),
            "build_status": result.get("build_status"),
            "evaluated_at": result.get("evaluated_at"),
            "submission_number": submission_number
        }
        doc_ref.set(new_data)
    else:
        # First evaluation or error result
        if result.get("status") == "completed":
            result["best_rubric_score"] = result.get("scores", {}).get("rubric", 0)
            result["total_evaluations"] = 1
            result["current_evaluation"] = {
                "submission_number": submission_number,
                "scores": result.get("scores", {}),
                "feedback": result.get("feedback", {}),
                "build_status": result.get("build_status"),
                "evaluated_at": result.get("evaluated_at"),
                "status": "completed"
            }
            result["evaluation_history"] = []
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


def _calculate_time_rank_bonus(rank: int) -> int:
    """Calculate time rank bonus based on submission order."""
    if rank == 1:
        return 20
    elif rank == 2:
        return 17
    elif rank == 3:
        return 14
    elif rank == 4:
        return 11
    elif rank == 5:
        return 8
    else:
        return 5


def get_week_leaderboard(week: int) -> List[dict]:
    """
    Get leaderboard for a specific week.
    Includes both completed evaluations and pending submissions.
    Recalculates time rank bonus based on latest submission order.
    Uses best rubric score for each participant.
    """
    evaluations = list_evaluations(week)
    submissions = list_submissions(week)

    # Build submission order map based on LATEST submission time
    submission_order = []
    submission_map = {}  # participant_id -> submission data
    for sub in submissions:
        latest_time = sub.get("latest_submitted_at") or sub.get("submitted_at")
        participant_id = sub.get("participant_id")
        if latest_time and participant_id:
            submission_order.append({
                "participant_id": participant_id,
                "submitted_at": latest_time
            })
            submission_map[participant_id] = sub

    # Sort by submission time to get time rank
    submission_order.sort(key=lambda x: x.get("submitted_at", "9999"))

    # Create time rank map
    time_rank_map = {}
    for rank, sub in enumerate(submission_order, start=1):
        time_rank_map[sub["participant_id"]] = rank

    # Track which participants have completed evaluations
    evaluated_ids = set()
    results = []

    for data in evaluations:
        if data.get("status") == "completed":
            participant_id = data.get("participant_id") or data.get("participant")
            evaluated_ids.add(participant_id)

            # Use best rubric score if available
            rubric_score = data.get("best_rubric_score") or data.get("scores", {}).get("rubric", 0)

            # Recalculate time rank and bonus based on current submission order
            time_rank = time_rank_map.get(participant_id, 999)
            time_rank_bonus = _calculate_time_rank_bonus(time_rank)

            # Calculate new total
            total = rubric_score + time_rank_bonus

            results.append({
                "participant_id": participant_id,
                "status": "completed",
                "total": total,
                "rubric": rubric_score,
                "time_rank": time_rank,
                "time_rank_bonus": time_rank_bonus,
                "evaluated_at": data.get("evaluated_at"),
                "submission_count": data.get("total_evaluations", 1)
            })

    # Add pending submissions (submissions without completed evaluations)
    for participant_id, sub in submission_map.items():
        if participant_id not in evaluated_ids:
            current_sub = sub.get("current_submission", {})
            results.append({
                "participant_id": participant_id,
                "status": "pending",
                "submitted_at": sub.get("latest_submitted_at") or sub.get("submitted_at"),
                "github_url": current_sub.get("github_url") or sub.get("github_url"),
                "elapsed_minutes": current_sub.get("elapsed_minutes") or sub.get("elapsed_minutes"),
                "time_rank": time_rank_map.get(participant_id, 999)
            })

    # Sort: completed first (by total score descending), then pending (by submission time ascending)
    results.sort(key=lambda x: (
        0 if x.get("status") == "completed" else 1,  # completed first
        -x.get("total", 0),  # higher score first for completed
        x.get("submitted_at", "9999")  # earlier submission first for pending
    ))

    # Add rank and medals (only for completed entries)
    completed_rank = 0
    for r in results:
        if r.get("status") == "completed":
            completed_rank += 1
            r["rank"] = completed_rank
            if completed_rank == 1:
                r["medal"] = "🥇"
            elif completed_rank == 2:
                r["medal"] = "🥈"
            elif completed_rank == 3:
                r["medal"] = "🥉"
            else:
                r["medal"] = ""
        else:
            r["rank"] = "-"
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
                "score": entry.get("total", entry.get("scores", {}).get("total", 0))
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
