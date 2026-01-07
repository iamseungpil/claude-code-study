#!/usr/bin/env python3
"""
Claude Code Study Evaluation System
Evaluates submissions using time tracking + Claude Code AI evaluation
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

# Time limits per week (in minutes)
TIME_LIMITS = {
    1: 45,   # Week 1: Project cleanup
    2: 50,   # Week 2: CLI tool
    3: 60,   # Week 3: Paper survey
    4: 60,   # Week 4: MVP
    5: 75,   # Week 5: Final project
}


def calculate_time_bonus(week: int, elapsed_minutes: float) -> int:
    """Calculate time bonus based on completion speed."""
    limit = TIME_LIMITS.get(week, 60)
    ratio = elapsed_minutes / limit
    
    if ratio <= 0.70:
        return 10
    elif ratio <= 0.85:
        return 5
    elif ratio <= 1.0:
        return 0
    else:
        # -5 points per 5 minutes over
        overtime = elapsed_minutes - limit
        penalty = int(overtime / 5) * 5
        return -penalty


def load_submission_metadata(week: int, participant_id: str) -> dict:
    """Load submission metadata (start_time, end_time)."""
    meta_path = SUBMISSIONS_DIR / f"week{week}" / participant_id / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return {}


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


def evaluate_submission(week: int, participant_id: str) -> dict:
    """Complete evaluation: time bonus + Claude evaluation."""
    
    # Load metadata for time calculation
    metadata = load_submission_metadata(week, participant_id)
    
    # Calculate elapsed time
    elapsed_minutes = None
    time_bonus = 0
    
    if "start_time" in metadata and "end_time" in metadata:
        start = datetime.fromisoformat(metadata["start_time"])
        end = datetime.fromisoformat(metadata["end_time"])
        elapsed = (end - start).total_seconds() / 60
        elapsed_minutes = round(elapsed, 1)
        time_bonus = calculate_time_bonus(week, elapsed)
    
    # Run Claude evaluation
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
    total_score = rubric_score + time_bonus
    
    result = {
        "participant": participant_id,
        "week": week,
        "status": "completed",
        "scores": {
            "rubric": rubric_score,
            "time_bonus": time_bonus,
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
            print(f"  Score: {result['scores']['total']}")
        else:
            print(f"  Error: {result.get('error')}")


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
                results.append({
                    "participant": data["participant"],
                    "total": data["scores"]["total"],
                    "rubric": data["scores"]["rubric"],
                    "time_bonus": data["scores"]["time_bonus"]
                })
    
    # Sort by total score descending
    results.sort(key=lambda x: x["total"], reverse=True)
    
    # Add rank
    for i, r in enumerate(results):
        r["rank"] = i + 1
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python evaluator.py evaluate <week> <participant_id>")
        print("  python evaluator.py evaluate-all <week>")
        print("  python evaluator.py leaderboard <week>")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "evaluate" and len(sys.argv) >= 4:
        week = int(sys.argv[2])
        participant_id = sys.argv[3]
        result = evaluate_submission(week, participant_id)
        print(json.dumps(result, indent=2))
        
    elif command == "evaluate-all" and len(sys.argv) >= 3:
        week = int(sys.argv[2])
        evaluate_all(week)
        
    elif command == "leaderboard" and len(sys.argv) >= 3:
        week = int(sys.argv[2])
        leaderboard = generate_leaderboard(week)
        print(json.dumps(leaderboard, indent=2))
        
    else:
        print("Invalid command")
        sys.exit(1)
