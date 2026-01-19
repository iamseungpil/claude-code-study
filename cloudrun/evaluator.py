#!/usr/bin/env python3
"""
Claude Code Study Evaluation System (Cloud Run Version)
Evaluates submissions using Claude Code CLI with ephemeral /tmp storage

Scoring System:
- Rubric Score: 80 points max (evaluated by Claude)
- Time Rank Bonus: 20 points max (calculated based on submission order)
  - 1st: +20, 2nd: +17, 3rd: +14, 4th: +11, 5th: +8, 6th+: +5
"""

import json
import subprocess
import shutil
import uuid
import os
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import firestore_client as db

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration - use /tmp for ephemeral storage in Cloud Run
TEMP_DIR = Path("/tmp")
RUBRICS_DIR = Path(__file__).parent / "rubrics"

# Time rank bonus points
TIME_RANK_POINTS = {
    1: 20,   # 1st place
    2: 17,   # 2nd place
    3: 14,   # 3rd place
    4: 11,   # 4th place
    5: 8,    # 5th place
}
DEFAULT_TIME_RANK_POINTS = 5


def calculate_time_rank_bonus(rank: int) -> int:
    """Calculate time rank bonus based on submission order."""
    return TIME_RANK_POINTS.get(rank, DEFAULT_TIME_RANK_POINTS)


def clone_github_repo(github_url: str, target_dir: Path) -> bool:
    """Clone a GitHub repository to target directory."""
    try:
        if target_dir.exists():
            shutil.rmtree(target_dir)

        target_dir.mkdir(parents=True, exist_ok=True)

        result = subprocess.run(
            ["git", "clone", "--depth", "1", github_url, str(target_dir)],
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode != 0:
            logger.error(f"Git clone failed: {result.stderr}")
            return False

        return True
    except Exception as e:
        logger.error(f"Clone error: {e}")
        return False


def run_build_verification(code_dir: Path) -> dict:
    """Run npm install and build to verify the submission."""
    result = {
        "npm_install": None,
        "npm_build": None,
        "success": False
    }

    try:
        # npm install
        install_result = subprocess.run(
            ["npm", "install", "--cache", "/tmp/npm-cache"],
            cwd=str(code_dir),
            capture_output=True,
            text=True,
            timeout=180
        )
        result["npm_install"] = install_result.returncode == 0

        if not result["npm_install"]:
            logger.warning(f"npm install failed: {install_result.stderr[:500]}")
            return result

        # npm run build
        build_result = subprocess.run(
            ["npm", "run", "build"],
            cwd=str(code_dir),
            capture_output=True,
            text=True,
            timeout=180
        )
        result["npm_build"] = build_result.returncode == 0

        if not result["npm_build"]:
            logger.warning(f"npm build failed: {build_result.stderr[:500]}")

        result["success"] = result["npm_install"] and result["npm_build"]

    except subprocess.TimeoutExpired:
        logger.error("Build verification timed out")
    except Exception as e:
        logger.error(f"Build verification error: {e}")

    return result


def run_claude_evaluation(code_dir: Path, week: int, participant_id: str) -> dict:
    """Run Claude Code CLI to evaluate the submission."""
    rubric_path = RUBRICS_DIR / f"week{week}_rubric.md"

    if not rubric_path.exists():
        return {"error": f"Rubric not found: {rubric_path}"}

    # Read rubric content
    with open(rubric_path) as f:
        rubric_content = f.read()

    # Create evaluation prompt
    prompt = f"""You are evaluating a Week {week} submission for the Claude Code Study.

## Rubric
{rubric_content}

## Instructions
1. Review the code in the current directory
2. Score according to the rubric (max 80 points)
3. Provide constructive feedback

## Output Format
Return ONLY a JSON object with this structure:
{{
    "rubric_score": <number 0-80>,
    "breakdown": {{
        "stage1": <number>,
        "stage2": <number>,
        "stage3": <number>,
        "stage4": <number>
    }},
    "feedback": "<overall feedback>",
    "strengths": ["<strength1>", "<strength2>"],
    "improvements": ["<improvement1>", "<improvement2>"]
}}
"""

    try:
        # Run Claude Code CLI
        result = subprocess.run(
            ["claude", "-p", prompt, "--output-format", "json"],
            cwd=str(code_dir),
            capture_output=True,
            text=True,
            timeout=300,
            env={
                **os.environ,
                "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY", "")
            }
        )

        output = result.stdout.strip()

        # Find JSON in output
        json_start = output.find('{')
        json_end = output.rfind('}') + 1

        if json_start >= 0 and json_end > json_start:
            return json.loads(output[json_start:json_end])
        else:
            logger.error(f"No JSON in Claude output: {output[:500]}")
            return {"error": "No JSON found in Claude output", "raw": output[:500]}

    except subprocess.TimeoutExpired:
        return {"error": "Evaluation timed out"}
    except json.JSONDecodeError as e:
        return {"error": f"JSON parse error: {e}"}
    except Exception as e:
        return {"error": str(e)}


def evaluate_submission(week: int, participant_id: str, github_url: str) -> dict:
    """Complete evaluation: clone, build verify, Claude eval, time rank bonus."""

    # Create unique temp directory
    session_id = str(uuid.uuid4())[:8]
    work_dir = TEMP_DIR / "submissions" / session_id
    code_dir = work_dir / "code"

    try:
        # 1. Clone repository
        logger.info(f"Cloning {github_url} to {code_dir}")
        if not clone_github_repo(github_url, code_dir):
            return {
                "participant": participant_id,
                "week": week,
                "status": "error",
                "error": "Failed to clone repository",
                "evaluated_at": datetime.now(timezone.utc).isoformat()
            }

        # 2. Run build verification
        logger.info("Running build verification")
        build_result = run_build_verification(code_dir)

        # 3. Get submission metadata and calculate time rank
        metadata = db.get_submission_metadata(week, participant_id)
        elapsed_minutes = metadata.get("elapsed_minutes") if metadata else None

        time_rank = db.get_submission_rank(week, participant_id)
        time_rank_bonus = calculate_time_rank_bonus(time_rank)

        # 4. Run Claude evaluation
        logger.info("Running Claude evaluation")
        claude_result = run_claude_evaluation(code_dir, week, participant_id)

        if "error" in claude_result:
            return {
                "participant": participant_id,
                "week": week,
                "status": "error",
                "error": claude_result["error"],
                "build_status": build_result,
                "evaluated_at": datetime.now(timezone.utc).isoformat()
            }

        # 5. Calculate final score
        rubric_score = claude_result.get("rubric_score", 0)

        # Apply build failure penalty (50% of rubric score)
        if not build_result["success"]:
            rubric_score = int(rubric_score * 0.5)
            logger.info(f"Build failed, applying 50% penalty: {rubric_score}")

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
            "build_status": build_result,
            "breakdown": claude_result.get("breakdown", {}),
            "feedback": claude_result.get("feedback", ""),
            "strengths": claude_result.get("strengths", []),
            "improvements": claude_result.get("improvements", []),
            "elapsed_minutes": elapsed_minutes,
            "evaluated_at": datetime.now(timezone.utc).isoformat()
        }

        # 6. Save to Firestore
        db.save_evaluation(week, participant_id, result)

        logger.info(f"Evaluation complete: {participant_id} - Total: {total_score}")
        return result

    finally:
        # 7. Cleanup temp directory
        if work_dir.exists():
            try:
                shutil.rmtree(work_dir)
                logger.info(f"Cleaned up temp directory: {work_dir}")
            except Exception as e:
                logger.warning(f"Failed to cleanup temp directory: {e}")


def trigger_evaluation(week: int, participant_id: str, github_url: str):
    """Entry point for background task evaluation."""
    try:
        logger.info(f"Starting evaluation: week={week}, participant={participant_id}")
        result = evaluate_submission(week, participant_id, github_url)

        if result.get("status") == "completed":
            logger.info(f"Evaluation successful: {result['scores']}")
        else:
            logger.error(f"Evaluation failed: {result.get('error')}")

    except Exception as e:
        logger.error(f"Evaluation exception: {e}")
        # Save error result
        db.save_evaluation(week, participant_id, {
            "participant": participant_id,
            "week": week,
            "status": "error",
            "error": str(e),
            "evaluated_at": datetime.now(timezone.utc).isoformat()
        })


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print("Usage: python evaluator.py <week> <participant_id> <github_url>")
        sys.exit(1)

    week = int(sys.argv[1])
    participant_id = sys.argv[2]
    github_url = sys.argv[3]

    result = evaluate_submission(week, participant_id, github_url)
    print(json.dumps(result, indent=2))
