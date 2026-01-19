#!/usr/bin/env python3
"""
Claude Code Study Evaluation System (Cloud Run Version)
Evaluates submissions using:
1. Build verification (npm install, npm run build)
2. E2E testing with Playwright (actual functionality testing)
3. Claude Code CLI evaluation (code review + test results)

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
import signal
import time
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, List

import firestore_client as db

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration - use /tmp for ephemeral storage in Cloud Run
TEMP_DIR = Path("/tmp")
RUBRICS_DIR = Path(__file__).parent / "rubrics"
TESTS_DIR = Path(__file__).parent / "e2e_tests"

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
        "success": False,
        "errors": []
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
            result["errors"].append(f"npm install failed: {install_result.stderr[:500]}")
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
            result["errors"].append(f"npm build failed: {build_result.stderr[:500]}")
            logger.warning(f"npm build failed: {build_result.stderr[:500]}")

        result["success"] = result["npm_install"] and result["npm_build"]

    except subprocess.TimeoutExpired:
        result["errors"].append("Build verification timed out")
        logger.error("Build verification timed out")
    except Exception as e:
        result["errors"].append(str(e))
        logger.error(f"Build verification error: {e}")

    return result


def start_dev_server(code_dir: Path, port: int = 3000) -> Optional[subprocess.Popen]:
    """Start the development server in background."""
    try:
        # Start dev server
        process = subprocess.Popen(
            ["npm", "run", "dev", "--", "--port", str(port)],
            cwd=str(code_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid  # Create new process group for cleanup
        )

        # Wait for server to be ready (max 30 seconds)
        for _ in range(30):
            time.sleep(1)
            try:
                import urllib.request
                urllib.request.urlopen(f"http://localhost:{port}", timeout=2)
                logger.info(f"Dev server started on port {port}")
                return process
            except:
                continue

        logger.warning("Dev server did not respond in time")
        return process  # Return anyway, might still work

    except Exception as e:
        logger.error(f"Failed to start dev server: {e}")
        return None


def stop_dev_server(process: subprocess.Popen):
    """Stop the development server."""
    if process:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait(timeout=5)
        except:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except:
                pass


def run_e2e_tests(code_dir: Path, week: int, port: int = 3000) -> dict:
    """Run Playwright E2E tests for the submission."""
    result = {
        "ran": False,
        "passed": 0,
        "failed": 0,
        "total": 0,
        "test_results": [],
        "errors": []
    }

    # Check if test file exists for this week
    test_file = TESTS_DIR / f"week{week}.spec.js"
    if not test_file.exists():
        logger.warning(f"No E2E tests found for week {week}")
        result["errors"].append(f"No E2E tests configured for week {week}")
        return result

    try:
        # Copy test file to project
        test_dest = code_dir / "e2e" / f"week{week}.spec.js"
        test_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(test_file, test_dest)

        # Install @playwright/test in the submission directory
        # This is needed because npx playwright test requires @playwright/test package
        logger.info("Installing @playwright/test in submission directory")
        install_pw = subprocess.run(
            ["npm", "install", "--save-dev", "@playwright/test", "--cache", "/tmp/npm-cache"],
            cwd=str(code_dir),
            capture_output=True,
            text=True,
            timeout=60
        )
        if install_pw.returncode != 0:
            logger.warning(f"Failed to install @playwright/test: {install_pw.stderr[:200]}")
            result["errors"].append("Failed to install Playwright test runner")
            return result

        # Create playwright config if not exists
        playwright_config = code_dir / "playwright.config.js"
        if not playwright_config.exists():
            playwright_config.write_text(f"""
module.exports = {{
  testDir: './e2e',
  timeout: 30000,
  use: {{
    baseURL: 'http://localhost:{port}',
    headless: true,
  }},
  reporter: [['json', {{ outputFile: 'test-results.json' }}]],
}};
""")

        # Run Playwright tests
        test_result = subprocess.run(
            ["npx", "playwright", "test", "--reporter=json"],
            cwd=str(code_dir),
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ, "CI": "true"}
        )

        result["ran"] = True

        # Parse test results
        results_file = code_dir / "test-results.json"
        if results_file.exists():
            with open(results_file) as f:
                test_data = json.load(f)

            for suite in test_data.get("suites", []):
                for spec in suite.get("specs", []):
                    test_name = spec.get("title", "Unknown")
                    test_passed = spec.get("ok", False)

                    result["total"] += 1
                    if test_passed:
                        result["passed"] += 1
                    else:
                        result["failed"] += 1

                    result["test_results"].append({
                        "name": test_name,
                        "passed": test_passed
                    })
        else:
            # Fallback: parse from stdout
            if "passed" in test_result.stdout.lower():
                result["passed"] = 1
                result["total"] = 1

    except subprocess.TimeoutExpired:
        result["errors"].append("E2E tests timed out")
        logger.error("E2E tests timed out")
    except Exception as e:
        result["errors"].append(str(e))
        logger.error(f"E2E test error: {e}")

    return result


def run_claude_evaluation(code_dir: Path, week: int, participant_id: str,
                          build_result: dict, e2e_result: dict) -> dict:
    """Run Claude Code CLI to evaluate the submission."""
    rubric_path = RUBRICS_DIR / f"week{week}_rubric.md"

    if not rubric_path.exists():
        return {"error": f"Rubric not found: {rubric_path}"}

    # Read rubric content
    with open(rubric_path) as f:
        rubric_content = f.read()

    # Format E2E test results for Claude
    e2e_summary = "E2E 테스트 미실행"
    if e2e_result.get("ran"):
        passed = e2e_result.get("passed", 0)
        total = e2e_result.get("total", 0)
        e2e_summary = f"E2E 테스트: {passed}/{total} 통과"

        test_details = []
        for t in e2e_result.get("test_results", []):
            status = "✅" if t["passed"] else "❌"
            test_details.append(f"  {status} {t['name']}")
        if test_details:
            e2e_summary += "\n" + "\n".join(test_details)

    # Create evaluation prompt
    prompt = f"""You are evaluating a Week {week} submission for the Claude Code Study.

## Build Status
- npm install: {"Pass" if build_result.get("npm_install") else "Fail"}
- npm build: {"Pass" if build_result.get("npm_build") else "Fail"}

## E2E Test Results
{e2e_summary}

## Rubric
{rubric_content}

## Instructions
1. Review the code in the current directory
2. Consider both code quality AND E2E test results
3. Score according to the rubric (max 80 points)
4. If E2E tests failed, deduct points for non-working features
5. Provide constructive feedback

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
    "feedback": "<overall feedback including E2E test observations>",
    "strengths": ["<strength1>", "<strength2>"],
    "improvements": ["<improvement1>", "<improvement2>"]
}}
"""

    try:
        # Run Claude Code CLI in non-interactive mode
        # --print (-p): Print mode, non-interactive
        # --output-format json: Return JSON output
        # ANTHROPIC_API_KEY: Required for API authentication
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            logger.error("ANTHROPIC_API_KEY not set")
            return {"error": "ANTHROPIC_API_KEY environment variable not set"}

        logger.info("Running Claude Code CLI evaluation")
        result = subprocess.run(
            ["claude", "-p", prompt, "--output-format", "json"],
            cwd=str(code_dir),
            capture_output=True,
            text=True,
            timeout=300,
            env={
                **os.environ,
                "ANTHROPIC_API_KEY": api_key,
                "CI": "true",  # Disable interactive prompts
                "TERM": "dumb"  # Prevent terminal escape codes
            }
        )

        output = result.stdout.strip()
        stderr = result.stderr.strip()

        # Log Claude CLI execution result
        if result.returncode != 0:
            logger.error(f"Claude CLI failed with code {result.returncode}")
            logger.error(f"stderr: {stderr[:500]}")
            return {"error": f"Claude CLI failed: {stderr[:300]}"}

        # Find JSON in output
        json_start = output.find('{')
        json_end = output.rfind('}') + 1

        if json_start >= 0 and json_end > json_start:
            return json.loads(output[json_start:json_end])
        else:
            logger.error(f"No JSON in Claude output: {output[:500]}")
            logger.error(f"stderr: {stderr[:500]}")
            return {"error": "No JSON found in Claude output", "raw": output[:500]}

    except subprocess.TimeoutExpired:
        return {"error": "Evaluation timed out"}
    except json.JSONDecodeError as e:
        return {"error": f"JSON parse error: {e}"}
    except Exception as e:
        return {"error": str(e)}


def evaluate_submission(week: int, participant_id: str, github_url: str) -> dict:
    """Complete evaluation: clone, build verify, E2E test, Claude eval, time rank bonus."""

    # Create unique temp directory
    session_id = str(uuid.uuid4())[:8]
    work_dir = TEMP_DIR / "submissions" / session_id
    code_dir = work_dir / "code"
    dev_server = None

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

        # 3. Run E2E tests (only if build succeeded)
        e2e_result = {"ran": False, "passed": 0, "failed": 0, "total": 0, "test_results": []}

        if build_result["success"]:
            logger.info("Starting dev server for E2E tests")
            dev_server = start_dev_server(code_dir, port=3000)

            if dev_server:
                logger.info("Running E2E tests")
                e2e_result = run_e2e_tests(code_dir, week, port=3000)
            else:
                e2e_result["errors"] = ["Failed to start dev server"]

        # 4. Get submission metadata and calculate time rank
        metadata = db.get_submission_metadata(week, participant_id)
        elapsed_minutes = metadata.get("elapsed_minutes") if metadata else None

        time_rank = db.get_submission_rank(week, participant_id)
        time_rank_bonus = calculate_time_rank_bonus(time_rank)

        # 5. Run Claude evaluation
        logger.info("Running Claude evaluation")
        claude_result = run_claude_evaluation(code_dir, week, participant_id, build_result, e2e_result)

        if "error" in claude_result:
            return {
                "participant": participant_id,
                "week": week,
                "status": "error",
                "error": claude_result["error"],
                "build_status": build_result,
                "e2e_status": e2e_result,
                "evaluated_at": datetime.now(timezone.utc).isoformat()
            }

        # 6. Calculate final score
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
            "e2e_status": {
                "ran": e2e_result.get("ran", False),
                "passed": e2e_result.get("passed", 0),
                "failed": e2e_result.get("failed", 0),
                "total": e2e_result.get("total", 0),
                "test_results": e2e_result.get("test_results", [])
            },
            "breakdown": claude_result.get("breakdown", {}),
            "feedback": claude_result.get("feedback", ""),
            "strengths": claude_result.get("strengths", []),
            "improvements": claude_result.get("improvements", []),
            "elapsed_minutes": elapsed_minutes,
            "evaluated_at": datetime.now(timezone.utc).isoformat()
        }

        # 7. Save to Firestore
        db.save_evaluation(week, participant_id, result)

        logger.info(f"Evaluation complete: {participant_id} - Total: {total_score}")
        return result

    finally:
        # 8. Stop dev server
        if dev_server:
            stop_dev_server(dev_server)

        # 9. Cleanup temp directory (DELETE REPO)
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
