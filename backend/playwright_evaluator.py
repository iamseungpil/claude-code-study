#!/usr/bin/env python3
"""
Playwright-based Evaluation System for Week 1 UIGen Challenge

This script:
1. Clones/prepares the submitted project
2. Installs dependencies and runs npm run dev
3. Runs Playwright E2E tests
4. Collects results and calculates scores

Usage:
    python playwright_evaluator.py <week> <participant_id>
    python playwright_evaluator.py <week> <participant_id> --github-url <url>
"""

import json
import subprocess
import os
import sys
import time
import signal
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import shutil

# Configuration
BASE_DIR = Path(__file__).parent.parent
SUBMISSIONS_DIR = BASE_DIR / "submissions"
EVALUATIONS_DIR = BASE_DIR / "evaluations"
TESTS_DIR = BASE_DIR / "tests"

# Port for dev server
DEV_SERVER_PORT = 3000
DEV_SERVER_TIMEOUT = 60  # seconds to wait for dev server to start


def clone_or_update_submission(
    week: int, participant_id: str, github_url: Optional[str] = None
) -> Dict[str, Any]:
    """Clone or update the submission from GitHub.

    Args:
        week: Week number
        participant_id: Participant ID
        github_url: Optional GitHub URL (if not provided, expects project already exists)

    Returns:
        Dict with status and path information
    """
    submission_dir = SUBMISSIONS_DIR / f"week{week}" / participant_id

    if github_url:
        # Clone if URL provided
        if submission_dir.exists():
            # Remove existing directory
            shutil.rmtree(submission_dir)

        submission_dir.parent.mkdir(parents=True, exist_ok=True)

        try:
            result = subprocess.run(
                ["git", "clone", "--depth", "1", github_url, str(submission_dir)],
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode != 0:
                return {
                    "success": False,
                    "error": f"Git clone failed: {result.stderr}"
                }

        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Git clone timed out"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    if not submission_dir.exists():
        return {
            "success": False,
            "error": f"Submission directory not found: {submission_dir}"
        }

    # Check if project is in 'code' subdirectory (server.py clones to code/)
    code_subdir = submission_dir / "code"
    if code_subdir.exists() and (code_subdir / "package.json").exists():
        return {
            "success": True,
            "path": str(code_subdir)
        }

    # Otherwise, project is at root (direct clone or github_url provided)
    return {
        "success": True,
        "path": str(submission_dir)
    }


def install_dependencies(project_path: str) -> Dict[str, Any]:
    """Install npm dependencies.

    Args:
        project_path: Path to the project directory

    Returns:
        Dict with status information
    """
    try:
        project_dir = Path(project_path)

        # Copy pre-installed uigen node_modules if available (speeds up npm install significantly)
        # Check both cloudrun path (Docker) and local path
        uigen_base_paths = [
            Path("/app/uigen-base/node_modules"),  # Docker/Cloud Run
            BASE_DIR / "cloudrun" / "uigen-base" / "node_modules",  # Local development
        ]

        target_modules = project_dir / "node_modules"
        if not target_modules.exists():
            for uigen_base in uigen_base_paths:
                if uigen_base.exists():
                    print("Copying pre-installed uigen node_modules...")
                    shutil.copytree(uigen_base, target_modules, symlinks=True)
                    print("Pre-installed node_modules copied successfully")
                    break

        # npm install (will be much faster with pre-copied node_modules)
        result = subprocess.run(
            ["npm", "install", "--cache", "/tmp/npm-cache"],
            cwd=project_path,
            capture_output=True,
            text=True,
            timeout=120  # Reduced from 300s since node_modules is pre-copied
        )

        if result.returncode != 0:
            return {
                "success": False,
                "error": f"npm install failed: {result.stderr}"
            }

        return {"success": True}

    except subprocess.TimeoutExpired:
        return {"success": False, "error": "npm install timed out"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def run_build(project_path: str) -> Dict[str, Any]:
    """Run npm build to verify the project builds.

    Args:
        project_path: Path to the project directory

    Returns:
        Dict with status information
    """
    try:
        # Run prisma generate and migrate if prisma folder exists
        prisma_dir = Path(project_path) / "prisma"
        if prisma_dir.exists():
            print("Running prisma generate...")
            prisma_result = subprocess.run(
                ["npx", "prisma", "generate"],
                cwd=project_path,
                capture_output=True,
                text=True,
                timeout=60
            )
            if prisma_result.returncode != 0:
                print(f"Prisma generate warning: {prisma_result.stderr}")

            # Run prisma migrate to create database tables (needed for auth)
            print("Running prisma migrate...")
            migrate_result = subprocess.run(
                ["npx", "prisma", "migrate", "dev", "--skip-generate"],
                cwd=project_path,
                capture_output=True,
                text=True,
                timeout=120,
                env={**os.environ, "SKIP_ENV_VALIDATION": "1"}
            )
            if migrate_result.returncode != 0:
                print(f"Prisma migrate warning: {migrate_result.stderr}")

        result = subprocess.run(
            ["npm", "run", "build"],
            cwd=project_path,
            capture_output=True,
            text=True,
            timeout=300
        )

        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr
        }

    except subprocess.TimeoutExpired:
        return {"success": False, "error": "npm build timed out"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def find_available_port(start_port: int = DEV_SERVER_PORT) -> int:
    """Find an available port starting from start_port.

    Args:
        start_port: Port number to start searching from

    Returns:
        An available port number
    """
    import socket
    for port in range(start_port, start_port + 100):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            continue
    return start_port  # Fallback


def start_dev_server(project_path: str, port: int = DEV_SERVER_PORT) -> tuple[Optional[subprocess.Popen], int]:
    """Start the development server.

    Args:
        project_path: Path to the project directory
        port: Port number for the dev server

    Returns:
        Tuple of (Popen object if successful, actual port used)
    """
    try:
        # Clean up stale .next build artifacts to avoid turbopack errors
        next_dir = Path(project_path) / ".next"
        if next_dir.exists():
            print("Cleaning up .next directory...")
            shutil.rmtree(next_dir)

        # Find an available port
        actual_port = find_available_port(port)
        print(f"Using port {actual_port}")

        # Start dev server in background
        env = os.environ.copy()
        env["PORT"] = str(actual_port)

        process = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=project_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            preexec_fn=os.setsid  # Create new process group for cleanup
        )

        # Wait for server to be ready
        start_time = time.time()
        while time.time() - start_time < DEV_SERVER_TIMEOUT:
            # Check if process is still running
            if process.poll() is not None:
                # Process died
                stdout, stderr = process.communicate()
                print(f"Dev server exited early. stderr: {stderr.decode()[:500]}")
                return None, actual_port

            try:
                response = urllib.request.urlopen(f"http://localhost:{actual_port}", timeout=2)
                if response.status == 200:
                    return process, actual_port
            except urllib.error.HTTPError as e:
                # Server is running but returned an error (OK for now)
                if e.code in [500, 404]:
                    # Server is up, just has an error page
                    return process, actual_port
            except:
                time.sleep(1)

        # Timeout - try to kill the process
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except:
            pass
        return None, actual_port

    except Exception as e:
        print(f"Error starting dev server: {e}")
        return None, port


def stop_dev_server(process: subprocess.Popen):
    """Stop the development server.

    Args:
        process: The Popen object of the dev server
    """
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        process.wait(timeout=5)
    except:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except:
            pass


def run_playwright_tests(week: int, port: int = DEV_SERVER_PORT) -> Dict[str, Any]:
    """Run Playwright E2E tests.

    Args:
        week: Week number
        port: Port where the dev server is running

    Returns:
        Dict with test results and scores
    """
    tests_dir = TESTS_DIR / f"week{week}"

    if not tests_dir.exists():
        return {"success": False, "error": f"Tests directory not found: {tests_dir}"}

    try:
        # Set environment variable for test URL
        env = os.environ.copy()
        env["UIGEN_URL"] = f"http://localhost:{port}"

        # Run Playwright tests
        result = subprocess.run(
            [
                "npx", "playwright", "test",
                "--project=uigen-evaluation",
                "--reporter=json"
            ],
            cwd=str(tests_dir),
            capture_output=True,
            text=True,
            timeout=300,
            env=env
        )

        # Try to parse JSON results
        try:
            # Find JSON in output
            json_output = result.stdout
            if "{" in json_output:
                json_start = json_output.find("{")
                json_end = json_output.rfind("}") + 1
                test_results = json.loads(json_output[json_start:json_end])
            else:
                test_results = {}
        except json.JSONDecodeError:
            test_results = {}

        # Read score report if generated
        score_report_path = tests_dir / "test-results" / "score-report.json"
        if score_report_path.exists():
            with open(score_report_path) as f:
                score_report = json.load(f)
        else:
            score_report = {}

        return {
            "success": result.returncode == 0,
            "test_results": test_results,
            "score_report": score_report,
            "stdout": result.stdout,
            "stderr": result.stderr
        }

    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Playwright tests timed out"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def analyze_claude_md(project_path: str) -> Dict[str, Any]:
    """Analyze CLAUDE.md file for quality.

    Args:
        project_path: Path to the project directory

    Returns:
        Dict with analysis results
    """
    claude_md_path = Path(project_path) / "CLAUDE.md"

    if not claude_md_path.exists():
        return {
            "exists": False,
            "score": 0,
            "feedback": "CLAUDE.md file not found"
        }

    with open(claude_md_path) as f:
        content = f.read()

    # Basic quality checks
    length = len(content)
    has_headings = "##" in content or "#" in content
    has_patterns = any(word in content.lower() for word in
                      ["pattern", "learning", "context", "error", "fix", "solution"])
    has_code_blocks = "```" in content

    # Calculate score (max 10 points)
    score = 0
    feedback = []

    if length > 100:
        score += 2
        feedback.append("Has substantial content")
    if length > 500:
        score += 2
        feedback.append("Has detailed documentation")
    if has_headings:
        score += 2
        feedback.append("Uses proper headings")
    if has_patterns:
        score += 2
        feedback.append("Documents patterns/learnings")
    if has_code_blocks:
        score += 2
        feedback.append("Includes code examples")

    return {
        "exists": True,
        "length": length,
        "score": min(score, 10),
        "feedback": ", ".join(feedback) if feedback else "Basic documentation"
    }


def calculate_playwright_scores(test_results: Dict[str, Any]) -> Dict[str, int]:
    """Calculate scores from Playwright test results.

    Args:
        test_results: Results from run_playwright_tests

    Returns:
        Dict with score breakdown
    """
    scores = {
        "stage_1_clear_all": 0,  # max 20 (button 10 + dialog 10)
        "stage_2_download_zip": 0,  # max 25 (button 10 + zip 15)
        "stage_3_keyboard": 0,  # max 25 (shortcut 10 + palette 10 + esc 5)
        "playwright_total": 0  # max 70
    }

    # Extract from score report if available
    score_report = test_results.get("score_report", {})
    if score_report:
        report_scores = score_report.get("scores", {})

        stage1 = report_scores.get("stage1", {})
        scores["stage_1_clear_all"] = stage1.get("total", 0)

        stage2 = report_scores.get("stage2", {})
        scores["stage_2_download_zip"] = stage2.get("total", 0)

        stage3 = report_scores.get("stage3", {})
        scores["stage_3_keyboard"] = stage3.get("total", 0)

        scores["playwright_total"] = report_scores.get("playwrightTotal", 0)

    return scores


def evaluate_submission_playwright(
    week: int,
    participant_id: str,
    github_url: Optional[str] = None
) -> Dict[str, Any]:
    """Complete Playwright-based evaluation.

    Args:
        week: Week number
        participant_id: Participant ID
        github_url: Optional GitHub URL

    Returns:
        Dict with complete evaluation results
    """
    result = {
        "participant": participant_id,
        "week": week,
        "evaluation_type": "playwright",
        "evaluated_at": datetime.now().isoformat(),
        "steps": [],
        "scores": {},
        "status": "pending"
    }

    # Step 1: Clone/prepare submission
    print(f"Step 1: Preparing submission...")
    clone_result = clone_or_update_submission(week, participant_id, github_url)
    result["steps"].append({"step": "clone", "result": clone_result})

    if not clone_result.get("success"):
        result["status"] = "error"
        result["error"] = clone_result.get("error")
        return result

    project_path = clone_result["path"]

    # Step 2: Install dependencies
    print(f"Step 2: Installing dependencies...")
    install_result = install_dependencies(project_path)
    result["steps"].append({"step": "install", "result": install_result})

    if not install_result.get("success"):
        result["status"] = "error"
        result["error"] = install_result.get("error")
        return result

    # Step 3: Build verification
    print(f"Step 3: Running build...")
    build_result = run_build(project_path)
    result["steps"].append({"step": "build", "result": {
        "success": build_result.get("success"),
        "error": build_result.get("error")
    }})
    result["build_status"] = "passed" if build_result.get("success") else "failed"

    # Step 4: Start dev server
    print(f"Step 4: Starting dev server...")
    dev_server, actual_port = start_dev_server(project_path)

    if not dev_server:
        result["status"] = "error"
        result["error"] = "Failed to start dev server"
        result["steps"].append({"step": "dev_server", "result": {"success": False}})
        return result

    result["steps"].append({"step": "dev_server", "result": {"success": True, "port": actual_port}})

    try:
        # Step 5: Run Playwright tests
        print(f"Step 5: Running Playwright tests...")
        test_results = run_playwright_tests(week, actual_port)
        result["steps"].append({"step": "playwright_tests", "result": {
            "success": test_results.get("success"),
            "error": test_results.get("error")
        }})

        # Step 6: Calculate scores
        playwright_scores = calculate_playwright_scores(test_results)
        result["scores"]["playwright"] = playwright_scores

        # Step 7: Analyze CLAUDE.md
        print(f"Step 6: Analyzing CLAUDE.md...")
        claude_md_result = analyze_claude_md(project_path)
        result["steps"].append({"step": "claude_md", "result": claude_md_result})
        result["scores"]["claude_md"] = claude_md_result.get("score", 0)

        # Calculate total
        result["scores"]["rubric_subtotal"] = (
            playwright_scores["playwright_total"] +
            claude_md_result.get("score", 0)
        )

        # Apply build penalty if needed
        if result["build_status"] == "failed":
            result["scores"]["build_penalty"] = -int(result["scores"]["rubric_subtotal"] * 0.5)
            result["scores"]["rubric_total"] = (
                result["scores"]["rubric_subtotal"] +
                result["scores"]["build_penalty"]
            )
        else:
            result["scores"]["build_penalty"] = 0
            result["scores"]["rubric_total"] = result["scores"]["rubric_subtotal"]

        result["status"] = "completed"

    finally:
        # Always stop the dev server
        print(f"Cleaning up: Stopping dev server...")
        stop_dev_server(dev_server)

    return result


def save_evaluation(week: int, participant_id: str, result: Dict[str, Any]):
    """Save evaluation result to JSON file.

    Args:
        week: Week number
        participant_id: Participant ID
        result: Evaluation result dict
    """
    output_dir = EVALUATIONS_DIR / f"week{week}"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{participant_id}_playwright.json"
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Saved: {output_path}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Playwright-based evaluation for UIGen projects"
    )
    parser.add_argument("week", type=int, help="Week number")
    parser.add_argument("participant_id", help="Participant ID")
    parser.add_argument("--github-url", help="GitHub URL to clone from")
    parser.add_argument("--output", "-o", help="Output file path")

    args = parser.parse_args()

    print(f"\n=== Playwright Evaluation ===")
    print(f"Week: {args.week}")
    print(f"Participant: {args.participant_id}")
    if args.github_url:
        print(f"GitHub URL: {args.github_url}")
    print()

    # Run evaluation
    result = evaluate_submission_playwright(
        args.week,
        args.participant_id,
        args.github_url
    )

    # Save result
    save_evaluation(args.week, args.participant_id, result)

    # Print summary
    print(f"\n=== Evaluation Summary ===")
    print(f"Status: {result['status']}")

    if result['status'] == 'completed':
        scores = result['scores']
        pw = scores.get('playwright', {})

        print(f"\nScores:")
        print(f"  Stage 1 (Clear All): {pw.get('stage_1_clear_all', 0)}/20")
        print(f"  Stage 2 (Download ZIP): {pw.get('stage_2_download_zip', 0)}/25")
        print(f"  Stage 3 (Keyboard): {pw.get('stage_3_keyboard', 0)}/25")
        print(f"  Playwright Total: {pw.get('playwright_total', 0)}/70")
        print(f"  CLAUDE.md: {scores.get('claude_md', 0)}/10")

        if scores.get('build_penalty', 0) < 0:
            print(f"  Build Penalty: {scores['build_penalty']}")

        print(f"\n  Rubric Total: {scores.get('rubric_total', 0)}/80")
        print(f"  (Time Rank Bonus calculated separately)")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")

    # Output JSON if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nFull results written to: {args.output}")

    return 0 if result['status'] == 'completed' else 1


if __name__ == "__main__":
    sys.exit(main())
