#!/usr/bin/env python3
"""
Claude Code Study Evaluation System
Evaluates submissions using time tracking + Claude Code AI evaluation

Scoring System:
- Rubric Score: 80-90 points max (evaluated by Claude, varies by week)
- Time Rank Bonus: 20 points max (calculated based on elapsed time)
  - 1st: +20, 2nd: +17, 3rd: +14, 4th: +11, 5th: +8, 6th+: +5
"""

import json
import subprocess
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

# Import SQLite database module
from database import (
    init_db,
    get_time_rank_by_elapsed,
    create_submission,
    update_submission_evaluation,
    get_submission,
    get_submission_count
)

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

# Time rank bonus points (based on elapsed time - shorter = better)
TIME_RANK_POINTS = {
    1: 20,   # 1st place (shortest elapsed time)
    2: 17,   # 2nd place
    3: 14,   # 3rd place
    4: 11,   # 4th place
    5: 8,    # 5th place
}
DEFAULT_TIME_RANK_POINTS = 5  # 6th place and beyond

# Maximum rubric scores per week (rubric_score + time_bonus = 100)
MAX_RUBRIC_SCORES = {
    1: 80,   # Week 1: 80 + 20 time bonus = 100
    2: 80,   # Week 2: 80 + 20 time bonus = 100
    3: 90,   # Week 3: 90 + 10 time bonus = 100
    4: 90,   # Week 4: 90 + 10 time bonus = 100
    5: 90,   # Week 5: 90 + 10 time bonus = 100
}


def calculate_time_rank_bonus(rank: int) -> int:
    """Calculate time rank bonus based on submission order.

    Args:
        rank: The submission rank (1 = first to submit, 2 = second, etc.)

    Returns:
        Time rank bonus points (20, 17, 14, 11, 8, or 5)
    """
    return TIME_RANK_POINTS.get(rank, DEFAULT_TIME_RANK_POINTS)


def get_submission_rank(week: int, participant_id: str) -> int:
    """Get the submission rank based on elapsed time (shorter = better).

    Uses SQLite database to calculate rank based on elapsed_minutes.
    Only considers the latest submission for each participant.

    Args:
        week: The week number
        participant_id: The participant's ID

    Returns:
        The rank (1-based, where 1 = shortest elapsed time)
    """
    # Try SQLite first
    try:
        rank = get_time_rank_by_elapsed(week, participant_id)
        if rank != 999:
            return rank
    except Exception as e:
        print(f"SQLite rank query failed: {e}, falling back to JSON")

    # Fallback to JSON files if SQLite fails
    submissions_dir = SUBMISSIONS_DIR / f"week{week}"

    if not submissions_dir.exists():
        return 999

    # Collect all submissions with elapsed_minutes
    submissions = []
    for participant_dir in submissions_dir.iterdir():
        if participant_dir.is_dir():
            metadata_file = participant_dir / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, encoding='utf-8') as f:
                    meta = json.load(f)
                    # Use elapsed_minutes for ranking (shorter = better)
                    if "elapsed_minutes" in meta:
                        submissions.append({
                            "participant_id": participant_dir.name,
                            "elapsed_minutes": meta["elapsed_minutes"]
                        })

    # Sort by elapsed time (shortest first = rank 1)
    submissions.sort(key=lambda x: x["elapsed_minutes"])

    # Find rank for the given participant
    for rank, sub in enumerate(submissions, start=1):
        if sub["participant_id"] == participant_id:
            return rank

    return 999  # Not found


def load_submission_metadata(week: int, participant_id: str) -> dict:
    """Load submission metadata (start_time, end_time)."""
    meta_path = SUBMISSIONS_DIR / f"week{week}" / participant_id / "metadata.json"
    if meta_path.exists():
        with open(meta_path, encoding='utf-8') as f:
            return json.load(f)
    return {}




def read_submission_code(submission_path: Path) -> dict:
    """Read key code files from a submission for code-only evaluation.

    Args:
        submission_path: Path to the submission directory

    Returns:
        Dict with code files, CLAUDE.md content, package.json, and structure
    """
    result = {
        "files": {},
        "claude_md": None,
        "package_json": None,
        "structure": []
    }

    # Check if code is in 'code' subdirectory
    code_dir = submission_path
    if (submission_path / "code" / "package.json").exists():
        code_dir = submission_path / "code"

    # Read CLAUDE.md
    for claude_path in [code_dir / "CLAUDE.md", submission_path / "CLAUDE.md"]:
        if claude_path.exists():
            content = claude_path.read_text(encoding='utf-8')
            result["claude_md"] = content[:8000]  # Limit size
            break

    # Read package.json
    pkg_path = code_dir / "package.json"
    if pkg_path.exists():
        try:
            result["package_json"] = json.loads(pkg_path.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            pass

    # Read key source files
    patterns = [
        # Week 1: React/Next.js components
        "src/components/**/*.tsx",
        "src/components/**/*.ts",
        "src/app/**/*.tsx",
        "src/app/**/*.ts",
        "src/lib/**/*.tsx",
        "src/lib/**/*.ts",
        "app/**/*.tsx",
        "components/**/*.tsx",
        # Week 2: Hooks files
        "hooks/**/*.js",
        "hooks/**/*.ts",
        ".claude/settings.json",
        ".claude/settings.example.json",
        # Week 3+: Skills and MCP
        "scripts/**/*.js",
        "scripts/**/*.ts",
    ]

    for pattern in patterns:
        for f in code_dir.glob(pattern):
            if "node_modules" in str(f):
                continue
            rel_path = str(f.relative_to(code_dir))
            try:
                content = f.read_text(encoding='utf-8')
                if len(content) < 15000:  # Skip huge files
                    result["files"][rel_path] = content
            except Exception:
                pass

    # Get directory structure
    try:
        result["structure"] = [
            str(p.relative_to(code_dir))
            for p in code_dir.rglob("*")
            if p.is_file() and "node_modules" not in str(p) and ".git" not in str(p)
        ][:100]
    except Exception:
        pass

    return result


def run_code_only_evaluation(week: int, participant_id: str) -> dict:
    """Evaluate submission by reading code only (no npm/build/E2E).

    This method reads the source code files and sends them to Claude
    for evaluation based on the rubric criteria.

    Args:
        week: The week number
        participant_id: The participant's ID

    Returns:
        Dict with evaluation results
    """
    submission_path = SUBMISSIONS_DIR / f"week{week}" / participant_id
    rubric_path = RUBRICS_DIR / f"week{week}_rubric.md"

    if not submission_path.exists():
        return {"error": f"Submission not found: {submission_path}"}

    if not rubric_path.exists():
        return {"error": f"Rubric not found: {rubric_path}"}

    # Read code files
    print(f"Reading code files from {submission_path}...")
    code_data = read_submission_code(submission_path)

    if not code_data["files"]:
        return {"error": "No source files found in submission"}

    # Read rubric
    rubric_content = rubric_path.read_text(encoding='utf-8')

    # Format code files for Claude
    code_summary = []
    for path, content in sorted(code_data["files"].items()):
        # Prioritize key files
        if any(key in path.lower() for key in ["headeractions", "commandpalette", "header"]):
            code_summary.insert(0, f"### {path}\n```tsx\n{content}\n```")
        else:
            code_summary.append(f"### {path}\n```tsx\n{content}\n```")

    # Limit total code size
    total_code = "\n\n".join(code_summary[:30])
    if len(total_code) > 50000:
        total_code = total_code[:50000] + "\n\n... (truncated)"

    # Get max rubric score for this week
    max_rubric_score = MAX_RUBRIC_SCORES.get(week, 80)

    # Build evaluation prompt
    prompt = f"""You are evaluating a Week {week} submission for Claude Code Study.

## Evaluation Mode: Code Analysis Only
Evaluate based on code review. DO NOT run any commands.

## CLAUDE.md Content
{code_data.get("claude_md") or "No CLAUDE.md found"}

## Project Structure (partial)
{chr(10).join(code_data["structure"][:40])}

## Source Code Files
{total_code}

## Rubric (MUST follow this scoring criteria strictly)
{rubric_content}

## CRITICAL INSTRUCTIONS
1. Read the rubric carefully - it contains ALL scoring criteria
2. Score EACH category according to the rubric breakdown
3. Maximum rubric score is {max_rubric_score} points for Week {week}
4. The rubric file contains JSON output examples - follow that format exactly
5. feedback should be 2-3 sentences in Korean

## OUTPUT FORMAT REQUIREMENT
YOU MUST OUTPUT ONLY A VALID JSON OBJECT. NO EXPLANATIONS, NO MARKDOWN, NO TEXT BEFORE OR AFTER THE JSON.

Start your response with {{ and end with }}. Do not include any other text.

Required JSON structure:
{{
    "rubric_score": <number 0-{max_rubric_score}>,
    "breakdown": {{
        "stage_1": <number>,
        "stage_2": <number>,
        "stage_3": <number>,
        "documentation": <number>
    }},
    "feedback": "<2-3 sentence summary in Korean>",
    "strengths": ["<strength 1>", "<strength 2>", "<strength 3>"],
    "improvements": ["<improvement 1>", "<improvement 2>"]
}}

OUTPUT ONLY THE JSON OBJECT ABOVE. NO OTHER TEXT."""

    # Run Claude CLI
    try:
        print("Running Claude code analysis...")

        # Pass prompt via stdin to avoid command line length limits
        result = subprocess.run(
            "claude --output-format json -p -",
            input=prompt,
            capture_output=True,
            text=True,
            timeout=180,  # 3 minute timeout
            env={**os.environ, "CI": "true"},
            shell=True,  # Required on Windows to find npm-installed commands
            encoding='utf-8',
            errors='replace'  # Handle encoding errors gracefully
        )

        output = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()

        # Debug logging
        print(f"Claude CLI exit code: {result.returncode}")
        if stderr:
            print(f"Claude CLI stderr: {stderr[:500]}")
        print(f"Claude CLI output length: {len(output)}")
        print(f"Claude CLI output preview: {output[:300]}...")

        # Try to parse JSON from output
        # Claude with --output-format json returns JSON directly
        try:
            parsed = json.loads(output)
            # Handle nested result structure from Claude CLI
            if "result" in parsed:
                inner_result = parsed["result"]
                if isinstance(inner_result, str):
                    # The result might be a JSON string or plain text
                    inner_result = inner_result.strip()
                    # Find JSON object in the result string
                    json_start = inner_result.find('{')
                    json_end = inner_result.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        try:
                            return json.loads(inner_result[json_start:json_end])
                        except json.JSONDecodeError as e:
                            print(f"Failed to parse inner JSON: {e}")
                            print(f"Inner result: {inner_result[:500]}")
                            return {"error": f"Inner JSON parse error: {e}", "raw": inner_result[:500]}
                    else:
                        # Plain text response, not JSON
                        return {"error": "Claude returned text instead of JSON", "raw": inner_result[:500]}
                elif isinstance(inner_result, dict):
                    return inner_result
            elif "rubric_score" in parsed:
                return parsed
            else:
                return {"error": "Unexpected JSON structure", "raw": str(parsed)[:500]}
        except json.JSONDecodeError as e:
            print(f"Failed to parse outer JSON: {e}")
            print(f"Raw output: {output[:500]}")

        # Fallback: find JSON in output
        json_start = output.find('{')
        json_end = output.rfind('}') + 1

        if json_start >= 0 and json_end > json_start:
            try:
                return json.loads(output[json_start:json_end])
            except json.JSONDecodeError as e:
                return {"error": f"Fallback JSON parse error: {e}", "raw": output[json_start:json_start+500]}
        else:
            return {"error": "No valid JSON in Claude output", "raw": output[:500]}

    except subprocess.TimeoutExpired:
        return {"error": "Code evaluation timed out"}
    except json.JSONDecodeError as e:
        return {"error": f"JSON parse error: {e}"}
    except Exception as e:
        return {"error": f"Code evaluation error: {str(e)}"}


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
    """Complete evaluation: rubric score + time rank bonus.

    Scoring:
    - Rubric Score: Up to 80 points (evaluated by Claude code analysis)
    - Time Rank Bonus: Up to 20 points (based on submission order)

    Args:
        week: The week number
        participant_id: The participant's ID
    """

    # Load metadata for time calculation
    metadata = load_submission_metadata(week, participant_id)

    # Calculate elapsed time (for display purposes)
    elapsed_minutes = metadata.get("elapsed_minutes")

    # Get submission rank and calculate time rank bonus
    time_rank = get_submission_rank(week, participant_id)
    time_rank_bonus = calculate_time_rank_bonus(time_rank)

    # Code-only evaluation (Claude reads code and evaluates)
    print(f"Running code-only evaluation for {participant_id}...")
    claude_result = run_code_only_evaluation(week, participant_id)

    if "error" in claude_result:
        error_result = {
            "participant": participant_id,
            "week": week,
            "status": "error",
            "error": claude_result["error"],
            "evaluated_at": datetime.now().isoformat()
        }
        # Save error result so it's visible in the frontend
        save_evaluation(week, participant_id, error_result)
        return error_result

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
    """Save evaluation result to JSON file, SQLite, and update submission history."""
    # Save to evaluation JSON file (for quick lookup - backward compatibility)
    output_dir = EVALUATIONS_DIR / f"week{week}"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{participant_id}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Saved JSON: {output_path}")

    # First, update metadata.json to get the correct submission_number
    metadata_path = SUBMISSIONS_DIR / f"week{week}" / participant_id / "metadata.json"
    submission_num = 1  # Default

    if metadata_path.exists():
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            # Get submission_number from metadata.json (source of truth)
            history = metadata.get("submission_history", [])
            if history:
                submission_num = len(history)  # Latest submission number
                # Add evaluation to the latest submission entry
                latest_submission = history[-1]
                latest_submission["evaluation"] = {
                    "rubric": result.get("scores", {}).get("rubric"),
                    "time_rank": result.get("scores", {}).get("time_rank"),
                    "time_rank_bonus": result.get("scores", {}).get("time_rank_bonus"),
                    "total": result.get("scores", {}).get("total"),
                    "status": result.get("status"),
                    "evaluated_at": result.get("evaluated_at")
                }
                metadata["submission_history"] = history

                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)

                print(f"Updated submission history: {metadata_path}")
        except Exception as e:
            print(f"Warning: Failed to update metadata.json: {e}")

    # Save to SQLite database with correct submission_number
    try:
        scores = result.get("scores", {})
        update_submission_evaluation(
            user_id=participant_id,
            week=week,
            submission_number=submission_num,
            rubric_score=scores.get("rubric", 0),
            time_rank=scores.get("time_rank", 999),
            time_rank_bonus=scores.get("time_rank_bonus", 0),
            total_score=scores.get("total", 0),
            feedback=result.get("feedback"),
            breakdown=result.get("breakdown"),
            strengths=result.get("strengths"),
            improvements=result.get("improvements"),
            status=result.get("status", "completed")
        )
        print(f"Saved to SQLite: {participant_id} week{week} try{submission_num}")
    except Exception as e:
        print(f"Warning: SQLite save failed: {e}")


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
        with open(eval_file, encoding='utf-8') as f:
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
        with open(eval_info["file"], 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"  {participant_id}: Rank #{time_rank} (+{time_rank_bonus}) = {data['scores']['total']}")


def generate_leaderboard(week: int) -> list:
    """Generate leaderboard for a specific week."""
    evaluations_dir = EVALUATIONS_DIR / f"week{week}"

    if not evaluations_dir.exists():
        return []

    results = []
    for eval_file in evaluations_dir.glob("*.json"):
        with open(eval_file, encoding='utf-8') as f:
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
        print("  python evaluator.py evaluate <week> <participant_id>")
        print("  python evaluator.py evaluate-all <week>")
        print("  python evaluator.py recalculate-ranks <week>")
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
