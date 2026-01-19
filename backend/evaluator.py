#!/usr/bin/env python3
"""
Claude Code Study Evaluation System
Evaluates submissions using time tracking + Claude Code AI evaluation

Scoring System:
- Rubric Score: 80 points max (evaluated by Claude)
- Time Rank Bonus: 20 points max (calculated based on submission order)
  - 1st: +20, 2nd: +17, 3rd: +14, 4th: +11, 5th: +8, 6th+: +5
"""

import json
import subprocess
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

# Configuration
BASE_DIR = Path(__file__).parent.parent
SUBMISSIONS_DIR = BASE_DIR / "submissions"
EVALUATIONS_DIR = BASE_DIR / "evaluations"
RUBRICS_DIR = BASE_DIR / "rubrics"
DATA_DIR = BASE_DIR / "data"

# Time limits per week (in minutes) - for reference, not for scoring
TIME_LIMITS = {
    1: 45,   # Week 1: Legacy Code Separation
    2: 50,   # Week 2: CLI tool
    3: 60,   # Week 3: Paper survey
    4: 60,   # Week 4: MVP
    5: 75,   # Week 5: Final project
}

# Time rank bonus points (based on submission order)
TIME_RANK_POINTS = {
    1: 20,   # 1st place
    2: 17,   # 2nd place
    3: 14,   # 3rd place
    4: 11,   # 4th place
    5: 8,    # 5th place
}
DEFAULT_TIME_RANK_POINTS = 5  # 6th place and beyond


def calculate_time_rank_bonus(rank: int) -> int:
    """Calculate time rank bonus based on submission order.

    Args:
        rank: The submission rank (1 = first to submit, 2 = second, etc.)

    Returns:
        Time rank bonus points (20, 17, 14, 11, 8, or 5)
    """
    return TIME_RANK_POINTS.get(rank, DEFAULT_TIME_RANK_POINTS)


def get_submission_rank(week: int, participant_id: str) -> int:
    """Get the submission rank for a participant based on submission time.

    Args:
        week: The week number
        participant_id: The participant's ID

    Returns:
        The rank (1-based, where 1 = first to submit)
    """
    submissions_dir = SUBMISSIONS_DIR / f"week{week}"

    if not submissions_dir.exists():
        return 999  # Default high rank if no submissions

    # Collect all submission times
    submissions = []
    for participant_dir in submissions_dir.iterdir():
        if participant_dir.is_dir():
            metadata_file = participant_dir / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file) as f:
                    meta = json.load(f)
                    if "submitted_at" in meta:
                        submissions.append({
                            "participant_id": participant_dir.name,
                            "submitted_at": meta["submitted_at"]
                        })

    # Sort by submission time (earliest first)
    submissions.sort(key=lambda x: x["submitted_at"])

    # Find the rank for the given participant
    for rank, sub in enumerate(submissions, start=1):
        if sub["participant_id"] == participant_id:
            return rank

    return 999  # Not found


def load_submission_metadata(week: int, participant_id: str) -> dict:
    """Load submission metadata (start_time, end_time)."""
    meta_path = SUBMISSIONS_DIR / f"week{week}" / participant_id / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return {}


def run_playwright_evaluation(week: int, participant_id: str) -> dict:
    """Run Playwright-based evaluation for supported weeks.

    Currently supports: Week 1 (UIGen Feature Sprint)

    Args:
        week: The week number
        participant_id: The participant's ID

    Returns:
        Dict with Playwright evaluation results or None if not supported
    """
    if week != 1:
        return None  # Playwright evaluation only for Week 1

    try:
        from playwright_evaluator import evaluate_submission_playwright

        result = evaluate_submission_playwright(week, participant_id)

        if result.get("status") == "completed":
            return {
                "playwright_scores": result.get("scores", {}).get("playwright", {}),
                "claude_md_score": result.get("scores", {}).get("claude_md", 0),
                "build_status": result.get("build_status"),
                "rubric_total": result.get("scores", {}).get("rubric_total", 0)
            }
        else:
            return {"error": result.get("error", "Playwright evaluation failed")}

    except ImportError:
        return {"error": "playwright_evaluator module not found"}
    except Exception as e:
        return {"error": f"Playwright evaluation error: {str(e)}"}


def run_claude_evaluation(week: int, participant_id: str) -> dict:
    """Run Claude Code to evaluate the submission."""
    submission_path = SUBMISSIONS_DIR / f"week{week}" / participant_id
    rubric_path = RUBRICS_DIR / f"week{week}_rubric.md"

    if not submission_path.exists():
        return {"error": f"Submission not found: {submission_path}"}

    if not rubric_path.exists():
        return {"error": f"Rubric not found: {rubric_path}"}

    # Claude Code command
    cmd = [
        "claude",
        "-p",
        f"evaluate-submission {week} {participant_id}"
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        # Parse Claude's JSON output
        output = result.stdout.strip()

        # Find JSON in output
        json_start = output.find('{')
        json_end = output.rfind('}') + 1

        if json_start >= 0 and json_end > json_start:
            return json.loads(output[json_start:json_end])
        else:
            return {"error": "No JSON found in Claude output", "raw": output}

    except subprocess.TimeoutExpired:
        return {"error": "Evaluation timed out"}
    except json.JSONDecodeError as e:
        return {"error": f"JSON parse error: {e}", "raw": output}
    except Exception as e:
        return {"error": str(e)}


def evaluate_submission(week: int, participant_id: str, use_playwright: bool = True) -> dict:
    """Complete evaluation: rubric score + time rank bonus.

    Scoring:
    - Rubric Score: Up to 80 points (evaluated by Claude or Playwright)
    - Time Rank Bonus: Up to 20 points (based on submission order)

    Args:
        week: The week number
        participant_id: The participant's ID
        use_playwright: Whether to use Playwright evaluation for Week 1 (default: True)
    """

    # Load metadata for time calculation
    metadata = load_submission_metadata(week, participant_id)

    # Calculate elapsed time (for display purposes)
    elapsed_minutes = metadata.get("elapsed_minutes")

    # Get submission rank and calculate time rank bonus
    time_rank = get_submission_rank(week, participant_id)
    time_rank_bonus = calculate_time_rank_bonus(time_rank)

    # Try Playwright evaluation for Week 1
    playwright_result = None
    if use_playwright and week == 1:
        print(f"Running Playwright evaluation for {participant_id}...")
        playwright_result = run_playwright_evaluation(week, participant_id)

        if playwright_result and "error" not in playwright_result:
            # Use Playwright results
            rubric_score = playwright_result.get("rubric_total", 0)
            total_score = rubric_score + time_rank_bonus

            result = {
                "participant": participant_id,
                "week": week,
                "status": "completed",
                "evaluation_method": "playwright",
                "scores": {
                    "rubric": rubric_score,
                    "time_rank": time_rank,
                    "time_rank_bonus": time_rank_bonus,
                    "total": total_score
                },
                "breakdown": {
                    "playwright": playwright_result.get("playwright_scores", {}),
                    "claude_md": playwright_result.get("claude_md_score", 0),
                    "build_status": playwright_result.get("build_status", "unknown")
                },
                "feedback": "Evaluated using Playwright E2E tests",
                "elapsed_minutes": elapsed_minutes,
                "evaluated_at": datetime.now().isoformat()
            }

            # Save result
            save_evaluation(week, participant_id, result)
            return result

    # Fallback to Claude evaluation
    print(f"Running Claude evaluation for {participant_id}...")
    claude_result = run_claude_evaluation(week, participant_id)

    if "error" in claude_result:
        return {
            "participant": participant_id,
            "week": week,
            "status": "error",
            "error": claude_result["error"],
            "evaluated_at": datetime.now().isoformat()
        }

    # Combine scores
    rubric_score = claude_result.get("rubric_score", 0)
    total_score = rubric_score + time_rank_bonus

    result = {
        "participant": participant_id,
        "week": week,
        "status": "completed",
        "scores": {
            "rubric": rubric_score,
            "time_rank": time_rank,
            "time_rank_bonus": time_rank_bonus,
            "total": total_score
        },
        "breakdown": claude_result.get("breakdown", {}),
        "feedback": claude_result.get("feedback", ""),
        "strengths": claude_result.get("strengths", []),
        "improvements": claude_result.get("improvements", []),
        "elapsed_minutes": elapsed_minutes,
        "evaluated_at": datetime.now().isoformat()
    }

    # Save result
    save_evaluation(week, participant_id, result)

    return result


def save_evaluation(week: int, participant_id: str, result: dict):
    """Save evaluation result to JSON file."""
    output_dir = EVALUATIONS_DIR / f"week{week}"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{participant_id}.json"
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Saved: {output_path}")


def get_pending_submissions(week: int) -> list:
    """Get list of submissions not yet evaluated."""
    submissions_dir = SUBMISSIONS_DIR / f"week{week}"
    evaluations_dir = EVALUATIONS_DIR / f"week{week}"

    if not submissions_dir.exists():
        return []

    pending = []
    for participant_dir in submissions_dir.iterdir():
        if participant_dir.is_dir():
            participant_id = participant_dir.name
            eval_path = evaluations_dir / f"{participant_id}.json"
            if not eval_path.exists():
                pending.append(participant_id)

    return pending


def evaluate_all(week: int):
    """Evaluate all pending submissions for a week."""
    pending = get_pending_submissions(week)

    print(f"Found {len(pending)} pending submissions for week {week}")

    for participant_id in pending:
        print(f"\nEvaluating: {participant_id}")
        result = evaluate_submission(week, participant_id)

        if result.get("status") == "completed":
            scores = result['scores']
            print(f"  Rubric: {scores['rubric']}, Time Rank: #{scores['time_rank']} (+{scores['time_rank_bonus']})")
            print(f"  Total: {scores['total']}")
        else:
            print(f"  Error: {result.get('error')}")


def recalculate_time_ranks(week: int):
    """Recalculate time rank bonuses for all evaluated submissions.

    This should be called after all submissions are evaluated to ensure
    time ranks are correctly calculated based on final submission order.
    """
    evaluations_dir = EVALUATIONS_DIR / f"week{week}"

    if not evaluations_dir.exists():
        print(f"No evaluations found for week {week}")
        return

    # Get all evaluations
    evaluations = []
    for eval_file in evaluations_dir.glob("*.json"):
        with open(eval_file) as f:
            data = json.load(f)
            if data.get("status") == "completed":
                evaluations.append({
                    "file": eval_file,
                    "data": data
                })

    print(f"Recalculating time ranks for {len(evaluations)} evaluations...")

    for eval_info in evaluations:
        data = eval_info["data"]
        participant_id = data["participant"]

        # Get new time rank
        time_rank = get_submission_rank(week, participant_id)
        time_rank_bonus = calculate_time_rank_bonus(time_rank)

        # Update scores
        rubric_score = data["scores"].get("rubric", 0)
        data["scores"]["time_rank"] = time_rank
        data["scores"]["time_rank_bonus"] = time_rank_bonus
        data["scores"]["total"] = rubric_score + time_rank_bonus

        # Remove old time_bonus if exists (backward compatibility)
        if "time_bonus" in data["scores"]:
            del data["scores"]["time_bonus"]

        # Save updated evaluation
        with open(eval_info["file"], 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"  {participant_id}: Rank #{time_rank} (+{time_rank_bonus}) = {data['scores']['total']}")


def generate_leaderboard(week: int) -> list:
    """Generate leaderboard for a specific week."""
    evaluations_dir = EVALUATIONS_DIR / f"week{week}"

    if not evaluations_dir.exists():
        return []

    results = []
    for eval_file in evaluations_dir.glob("*.json"):
        with open(eval_file) as f:
            data = json.load(f)
            if data.get("status") == "completed":
                scores = data["scores"]
                results.append({
                    "participant": data["participant"],
                    "total": scores["total"],
                    "rubric": scores["rubric"],
                    "time_rank": scores.get("time_rank", 0),
                    "time_rank_bonus": scores.get("time_rank_bonus", scores.get("time_bonus", 0))
                })

    # Sort by total score descending
    results.sort(key=lambda x: x["total"], reverse=True)

    # Add overall rank (by score, not submission time)
    for i, r in enumerate(results):
        r["rank"] = i + 1

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python evaluator.py evaluate <week> <participant_id> [--no-playwright]")
        print("  python evaluator.py evaluate-playwright <week> <participant_id>")
        print("  python evaluator.py evaluate-all <week>")
        print("  python evaluator.py recalculate-ranks <week>")
        print("  python evaluator.py leaderboard <week>")
        sys.exit(1)

    command = sys.argv[1]

    if command == "evaluate" and len(sys.argv) >= 4:
        week = int(sys.argv[2])
        participant_id = sys.argv[3]
        use_playwright = "--no-playwright" not in sys.argv
        result = evaluate_submission(week, participant_id, use_playwright=use_playwright)
        print(json.dumps(result, indent=2))

    elif command == "evaluate-playwright" and len(sys.argv) >= 4:
        week = int(sys.argv[2])
        participant_id = sys.argv[3]
        result = run_playwright_evaluation(week, participant_id)
        print(json.dumps(result, indent=2))

    elif command == "evaluate-all" and len(sys.argv) >= 3:
        week = int(sys.argv[2])
        evaluate_all(week)

    elif command == "recalculate-ranks" and len(sys.argv) >= 3:
        week = int(sys.argv[2])
        recalculate_time_ranks(week)

    elif command == "leaderboard" and len(sys.argv) >= 3:
        week = int(sys.argv[2])
        leaderboard = generate_leaderboard(week)
        print(json.dumps(leaderboard, indent=2))

    else:
        print("Invalid command")
        sys.exit(1)
