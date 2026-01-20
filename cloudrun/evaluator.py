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
import re
import urllib.error
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


def read_submission_code(code_dir: Path) -> dict:
    """Read key code files for evaluation (code-only mode, no npm/E2E).

    This function reads source code files directly for Claude evaluation
    without running npm install or E2E tests - saves memory and time.

    Args:
        code_dir: Path to the cloned submission directory

    Returns:
        dict with files, claude_md, package_json, and structure
    """
    result = {
        "files": {},
        "claude_md": None,
        "package_json": None,
        "structure": []
    }

    try:
        # Read CLAUDE.md (memory-driven development check)
        claude_md = code_dir / "CLAUDE.md"
        if claude_md.exists():
            result["claude_md"] = claude_md.read_text()[:5000]

        # Read package.json
        pkg = code_dir / "package.json"
        if pkg.exists():
            result["package_json"] = json.loads(pkg.read_text())

        # Read key source files (app/, src/, components/, pages/)
        patterns = [
            "app/**/*.tsx", "src/**/*.tsx", "components/**/*.tsx",
            "pages/**/*.tsx", "app/**/*.ts", "src/**/*.ts",
            "app/**/*.jsx", "src/**/*.jsx", "components/**/*.jsx",
            "pages/**/*.jsx", "app/**/*.js", "src/**/*.js"
        ]
        for pattern in patterns:
            for f in code_dir.glob(pattern):
                # Skip node_modules
                if "node_modules" in str(f):
                    continue
                rel_path = str(f.relative_to(code_dir))
                try:
                    content = f.read_text()
                    if len(content) < 10000:  # Skip huge files
                        result["files"][rel_path] = content
                except Exception as e:
                    logger.warning(f"Failed to read {rel_path}: {e}")

        # Get directory structure (exclude node_modules, .git, etc.)
        exclude_dirs = {".git", "node_modules", ".next", "dist", "build", "__pycache__"}
        structure = []
        for p in code_dir.rglob("*"):
            if p.is_file():
                # Skip excluded directories
                if any(exc in str(p) for exc in exclude_dirs):
                    continue
                structure.append(str(p.relative_to(code_dir)))
        result["structure"] = structure[:100]  # Limit to 100 files

    except Exception as e:
        logger.error(f"Error reading submission code: {e}")

    return result


def validate_github_url(url: str) -> bool:
    """
    Validate GitHub URL format to prevent command injection.

    Only allows standard GitHub HTTPS URLs:
    - https://github.com/username/repo
    - https://github.com/username/repo.git
    """
    if not url:
        return False
    # GitHub HTTPS URL pattern - strict validation
    pattern = r'^https://github\.com/[\w\-\.]+/[\w\-\.]+(?:\.git)?$'
    return bool(re.match(pattern, url))


def clone_github_repo(github_url: str, target_dir: Path) -> bool:
    """Clone a GitHub repository to target directory."""
    # SECURITY: Validate URL format before execution to prevent command injection
    if not validate_github_url(github_url):
        logger.error(f"Invalid GitHub URL format: {github_url}")
        return False

    try:
        if target_dir.exists():
            shutil.rmtree(target_dir)

        target_dir.mkdir(parents=True, exist_ok=True)

        result = subprocess.run(
            ["git", "clone", "--depth", "1", github_url, str(target_dir)],
            capture_output=True,
            text=True,
            timeout=60  # Reduced from 120
        )

        if result.returncode != 0:
            logger.error(f"Git clone failed: {result.stderr}")
            return False

        return True
    except subprocess.TimeoutExpired:
        logger.error("Git clone timed out")
        return False
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
        # Don't copy pre-installed node_modules - npm binaries have hardcoded paths
        # that break when copied to a different location. Just run npm install.
        # The npm cache will still help with performance.

        # npm install
        install_result = subprocess.run(
            ["npm", "install", "--cache", "/tmp/npm-cache"],
            cwd=str(code_dir),
            capture_output=True,
            text=True,
            timeout=120  # Reduced from 180s since node_modules is pre-copied
        )
        result["npm_install"] = install_result.returncode == 0

        if not result["npm_install"]:
            result["errors"].append(f"npm install failed: {install_result.stderr[:500]}")
            logger.warning(f"npm install failed: {install_result.stderr[:500]}")
            return result

        # Run prisma generate if prisma schema exists
        prisma_schema = code_dir / "prisma" / "schema.prisma"
        if prisma_schema.exists():
            logger.info("Running prisma generate")
            try:
                prisma_result = subprocess.run(
                    ["npx", "prisma", "generate"],
                    cwd=str(code_dir),
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                if prisma_result.returncode != 0:
                    logger.warning(f"prisma generate warning: {prisma_result.stderr[:200]}")
            except Exception as e:
                logger.warning(f"prisma generate error: {e}")

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
        import urllib.request
        for _ in range(30):
            time.sleep(1)
            try:
                urllib.request.urlopen(f"http://localhost:{port}", timeout=2)
                logger.info(f"Dev server started on port {port}")
                return process
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                logger.debug(f"Server not ready yet: {e}")
                continue

        logger.warning("Dev server did not respond in time")
        return process  # Return anyway, might still work

    except Exception as e:
        logger.error(f"Failed to start dev server: {e}")
        return None


def stop_dev_server(process: subprocess.Popen):
    """Safely stop the development server with proper cleanup."""
    if process is None:
        return

    # Check if process is still running
    if process.poll() is not None:
        logger.debug("Dev server already terminated")
        return

    try:
        pgid = os.getpgid(process.pid)
        os.killpg(pgid, signal.SIGTERM)

        # Wait for graceful shutdown
        try:
            process.wait(timeout=5)
            logger.debug("Dev server terminated gracefully")
        except subprocess.TimeoutExpired:
            logger.warning("Dev server did not terminate, sending SIGKILL")
            os.killpg(pgid, signal.SIGKILL)
            process.wait(timeout=2)

    except ProcessLookupError:
        logger.debug("Process group already gone")
    except OSError as e:
        logger.warning(f"Error stopping dev server: {e}")


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


def run_e2e_tests_with_mcp(code_dir: Path, week: int, port: int = 3000) -> dict:
    """
    Run E2E tests using Claude Code CLI with Playwright MCP tools.

    This function uses Claude's MCP integration to perform browser-based testing
    through natural language instructions, providing more flexible and intelligent
    test execution compared to static Playwright test scripts.

    Args:
        code_dir: Path to the submission code directory
        week: Week number for the challenge
        port: Port where the dev server is running

    Returns:
        dict with test results including:
        - ran: bool indicating if tests executed
        - passed: number of passed test criteria
        - failed: number of failed test criteria
        - total: total number of test criteria
        - test_results: list of individual test outcomes
        - errors: list of any errors encountered
        - method: 'mcp' to indicate this method was used
    """
    result = {
        "ran": False,
        "passed": 0,
        "failed": 0,
        "total": 0,
        "test_results": [],
        "errors": [],
        "method": "mcp"
    }

    # Load test criteria from rubric or predefined test cases
    test_criteria = _get_mcp_test_criteria(week, port)
    if not test_criteria:
        result["errors"].append(f"No MCP test criteria defined for week {week}")
        return result

    # Build the MCP test prompt
    prompt = f"""You have access to Playwright MCP tools for browser automation.

Test the application running at http://localhost:{port}

## Test Criteria
{test_criteria}

## Instructions
1. Use the playwright tools to navigate to the page
2. Execute each test criterion using appropriate MCP tools
3. For each criterion, determine if it PASSED or FAILED
4. Return your findings as a JSON object

## Required Output Format
Return ONLY a valid JSON object with this exact structure:
{{
    "test_results": [
        {{"name": "test criterion name", "passed": true/false, "details": "what happened"}}
    ],
    "summary": "brief overall summary"
}}
"""

    try:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            result["errors"].append("ANTHROPIC_API_KEY not set")
            return result

        logger.info("Running E2E tests with Playwright MCP")

        # Run Claude Code CLI with MCP tools enabled
        # --allowedTools permits all playwright MCP tools
        cli_result = subprocess.run(
            [
                "claude", "-p", prompt,
                "--output-format", "json",
                "--allowedTools", "mcp__playwright__*"
            ],
            cwd=str(code_dir),
            capture_output=True,
            text=True,
            timeout=180,  # 3 minutes for MCP tests
            env={
                **os.environ,
                "ANTHROPIC_API_KEY": api_key,
                "CI": "true",
                "TERM": "dumb"
            }
        )

        if cli_result.returncode != 0:
            logger.warning(f"MCP test CLI returned non-zero: {cli_result.returncode}")
            logger.warning(f"stderr: {cli_result.stderr[:500]}")
            result["errors"].append(f"Claude CLI error: {cli_result.stderr[:300]}")
            return result

        # Parse the response
        output = cli_result.stdout.strip()
        json_start = output.find('{')
        json_end = output.rfind('}') + 1

        if json_start >= 0 and json_end > json_start:
            test_output = json.loads(output[json_start:json_end])
            result["ran"] = True

            # Process test results
            for test in test_output.get("test_results", []):
                test_name = test.get("name", "Unknown")
                test_passed = test.get("passed", False)

                result["total"] += 1
                if test_passed:
                    result["passed"] += 1
                else:
                    result["failed"] += 1

                result["test_results"].append({
                    "name": test_name,
                    "passed": test_passed,
                    "details": test.get("details", "")
                })

            logger.info(f"MCP tests completed: {result['passed']}/{result['total']} passed")
        else:
            result["errors"].append("No valid JSON in MCP test output")
            logger.warning(f"No JSON found in output: {output[:500]}")

    except subprocess.TimeoutExpired:
        result["errors"].append("MCP E2E tests timed out (180s)")
        logger.error("MCP E2E tests timed out")
    except json.JSONDecodeError as e:
        result["errors"].append(f"Failed to parse MCP test results: {e}")
        logger.error(f"JSON parse error in MCP results: {e}")
    except Exception as e:
        result["errors"].append(f"MCP test error: {str(e)}")
        logger.error(f"MCP test exception: {e}")

    return result


def _get_mcp_test_criteria(week: int, port: int) -> Optional[str]:
    """
    Get MCP test criteria for a specific week.

    This function returns natural language test criteria that Claude
    can execute using Playwright MCP tools.

    Args:
        week: Week number
        port: Port where the app is running

    Returns:
        String with test criteria or None if not defined
    """
    # Week-specific test criteria for MCP-based testing
    # These are written as natural language instructions for Claude
    criteria_map = {
        1: """
1. Page Load Test
   - Navigate to http://localhost:{port}
   - Verify the page loads successfully
   - Check for a main heading or title element

2. Todo Input Test
   - Look for an input field for adding todos
   - Type "Test todo item" into the input
   - Submit the todo (press Enter or click Add button)
   - Verify the todo appears in the list

3. Todo Completion Test
   - Find a todo item in the list
   - Click to mark it as complete
   - Verify visual indication of completion (strikethrough, checkbox, etc.)

4. Todo Delete Test
   - Find a todo item with a delete button
   - Click the delete button
   - Verify the item is removed from the list
""",
        2: """
1. Page Load Test
   - Navigate to http://localhost:{port}
   - Verify the page loads successfully
   - Check for main application elements

2. Clear All Button Test
   - Look for a "Clear All" or similar button
   - Click the button
   - Verify a confirmation dialog or action occurs

3. Filter Functionality Test
   - Look for filter options (All, Active, Completed)
   - Click on different filter options
   - Verify the list updates accordingly

4. Persistence Test
   - Add a todo item
   - Refresh the page
   - Verify the todo item persists
""",
        3: """
1. Page Load Test
   - Navigate to http://localhost:{port}
   - Verify the page loads successfully

2. Data Display Test
   - Look for data visualization or list components
   - Verify data is rendered on the page

3. Interactive Element Test
   - Find interactive elements (buttons, links)
   - Click on them and verify response

4. Error Handling Test
   - Try invalid inputs if input fields exist
   - Verify appropriate error handling
""",
    }

    criteria = criteria_map.get(week)
    if criteria:
        return criteria.format(port=port)

    # Default criteria for undefined weeks
    return f"""
1. Page Load Test
   - Navigate to http://localhost:{port}
   - Verify the page loads successfully
   - Check for main content elements

2. Basic Interaction Test
   - Find interactive elements on the page
   - Test basic interactions
   - Verify responses
""".format(port=port)


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


def run_claude_evaluation_code_only(code_dir: Path, week: int,
                                     participant_id: str, code_data: dict) -> dict:
    """Evaluate submission by reading code only (no npm/E2E).

    This function evaluates submissions based on code analysis alone,
    without running npm install, npm build, or E2E tests. This saves
    memory and works within Cloud Run free tier constraints.

    Uses Claude CLI for evaluation (requires @anthropic-ai/claude-code installed).

    Args:
        code_dir: Path to the cloned submission directory
        week: Week number
        participant_id: Participant ID
        code_data: Dict from read_submission_code() with files, claude_md, etc.

    Returns:
        dict with rubric_score, breakdown, feedback, strengths, improvements
    """
    rubric_path = RUBRICS_DIR / f"week{week}_rubric.md"

    if not rubric_path.exists():
        return {"error": f"Rubric not found: {rubric_path}"}

    # Read rubric content
    with open(rubric_path) as f:
        rubric_content = f.read()

    # Format code files for Claude
    code_summary = []
    for path, content in list(code_data.get("files", {}).items())[:20]:  # Limit to 20 files
        # Truncate very long files
        truncated = content[:5000] + "...(truncated)" if len(content) > 5000 else content
        code_summary.append(f"### {path}\n```tsx\n{truncated}\n```")

    # Format project structure
    structure_str = "\n".join(code_data.get("structure", [])[:50])

    # Format CLAUDE.md content
    claude_md_content = code_data.get("claude_md", "No CLAUDE.md found")

    # Create evaluation prompt
    prompt = f"""You are evaluating a Week {week} submission for Claude Code Study.

## Evaluation Mode: Code Analysis Only
(No build/E2E tests due to resource constraints - evaluate based on code review)

## CLAUDE.md (Memory-Driven Development)
{claude_md_content}

## Project Structure
{structure_str}

## Source Code Files
{chr(10).join(code_summary)}

## Rubric
{rubric_content}

## Instructions
1. Analyze the code structure and implementation
2. Check if CLAUDE.md documents the project properly
3. Evaluate feature completeness based on code patterns
4. Score according to rubric (max 80 points)
5. Deduct points if code has obvious bugs or missing features
6. Be fair - without E2E tests, focus on code quality and completeness

## Output Format
Return ONLY valid JSON:
{{
    "rubric_score": <0-80>,
    "breakdown": {{"stage1": <n>, "stage2": <n>, "stage3": <n>, "stage4": <n>}},
    "feedback": "<feedback>",
    "strengths": ["<s1>", "<s2>"],
    "improvements": ["<i1>", "<i2>"]
}}"""

    try:
        logger.info("Running Claude CLI evaluation (code-only mode)")

        # Check if credentials file exists (OAuth login)
        credentials_file = Path("/root/.claude/.credentials.json")
        if credentials_file.exists():
            logger.info("Using OAuth credentials from /root/.claude/.credentials.json")
        else:
            # Fallback to API key if no credentials file
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            if not api_key:
                logger.error("No credentials file and ANTHROPIC_API_KEY not set")
                return {"error": "No authentication available (no credentials file or API key)"}

        # Use Claude CLI for evaluation
        # --print (-p): Print mode, non-interactive
        # --output-format json: Request JSON output
        # CI=true environment variable handles non-interactive mode
        # HOME=/root ensures Claude CLI finds ~/.claude/.credentials.json
        env = {
            **os.environ,
            "CI": "true",
            "TERM": "dumb",
            "HOME": "/root"  # Ensure Claude CLI finds credentials at /root/.claude/
        }

        result = subprocess.run(
            [
                "claude", "-p", prompt,
                "--output-format", "json"
            ],
            cwd=str(code_dir),
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            env=env
        )

        output = result.stdout.strip()
        stderr = result.stderr.strip()

        logger.info(f"Claude CLI exit code: {result.returncode}")
        if stderr:
            logger.info(f"Claude CLI stderr (first 500 chars): {stderr[:500]}")

        if result.returncode != 0:
            logger.error(f"Claude CLI failed with code {result.returncode}")
            logger.error(f"stdout: {output[:500]}")
            logger.error(f"stderr: {stderr[:500]}")
            return {"error": f"Claude CLI failed (exit {result.returncode}): {stderr[:300] or output[:300]}"}

        # Find JSON in output
        json_start = output.find('{')
        json_end = output.rfind('}') + 1

        if json_start >= 0 and json_end > json_start:
            parsed = json.loads(output[json_start:json_end])
            logger.info(f"Claude CLI evaluation successful, rubric_score: {parsed.get('rubric_score')}")
            return parsed
        else:
            logger.error(f"No JSON in Claude output: {output[:500]}")
            return {"error": "No JSON found in Claude output", "raw": output[:500]}

    except subprocess.TimeoutExpired:
        logger.error("Claude CLI evaluation timed out (300s)")
        return {"error": "Evaluation timed out (300s)"}
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}")
        return {"error": f"JSON parse error: {e}"}
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        return {"error": str(e)}


def evaluate_submission(week: int, participant_id: str, github_url: str) -> dict:
    """Complete evaluation: clone, read code, Claude eval, time rank bonus.

    CODE-ONLY MODE: Skips npm install, npm build, and E2E tests to save memory.
    This allows evaluation to run within Cloud Run free tier (512Mi-2Gi).
    """

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

        # 2. SKIP build verification - code-only mode
        build_result = {
            "npm_install": None,
            "npm_build": None,
            "success": None,
            "skipped": True,
            "reason": "Code-only evaluation mode (no npm/E2E)"
        }
        logger.info("Skipping build verification (code-only mode)")

        # 3. SKIP E2E tests - code-only mode
        e2e_result = {
            "ran": False,
            "skipped": True,
            "reason": "Code-only evaluation mode (no npm/E2E)",
            "method": "code_only"
        }
        logger.info("Skipping E2E tests (code-only mode)")

        # 4. Read code files directly
        logger.info("Reading submission code files")
        code_data = read_submission_code(code_dir)
        logger.info(f"Read {len(code_data.get('files', {}))} source files")

        # 5. Get submission metadata and calculate time rank
        metadata = db.get_submission_metadata(week, participant_id)
        elapsed_minutes = metadata.get("elapsed_minutes") if metadata else None

        time_rank = db.get_submission_rank(week, participant_id)
        time_rank_bonus = calculate_time_rank_bonus(time_rank)

        # 6. Run Claude evaluation (code-only mode)
        logger.info("Running Claude evaluation (code-only mode)")
        claude_result = run_claude_evaluation_code_only(
            code_dir, week, participant_id, code_data
        )

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

        # 7. Calculate final score (no build penalty since we didn't build)
        rubric_score = claude_result.get("rubric_score", 0)
        total_score = rubric_score + time_rank_bonus

        result = {
            "participant": participant_id,
            "week": week,
            "status": "completed",
            "evaluation_mode": "code_only",
            "scores": {
                "rubric": rubric_score,
                "time_rank": time_rank,
                "time_rank_bonus": time_rank_bonus,
                "total": total_score
            },
            "build_status": build_result,
            "e2e_status": e2e_result,
            "code_analysis": {
                "files_read": len(code_data.get("files", {})),
                "has_claude_md": code_data.get("claude_md") is not None,
                "structure_count": len(code_data.get("structure", []))
            },
            "breakdown": claude_result.get("breakdown", {}),
            "feedback": claude_result.get("feedback", ""),
            "strengths": claude_result.get("strengths", []),
            "improvements": claude_result.get("improvements", []),
            "elapsed_minutes": elapsed_minutes,
            "evaluated_at": datetime.now(timezone.utc).isoformat()
        }

        # 8. Save to Firestore
        db.save_evaluation(week, participant_id, result)

        logger.info(f"Evaluation complete: {participant_id} - Total: {total_score}")
        return result

    finally:
        # Cleanup temp directory
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
            # BUG FIX: Save error results to Firestore
            # Previously error results were not being saved, only logged
            db.save_evaluation(week, participant_id, result)

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
