# Evaluate Submission

Evaluate a participant's submission for the Claude Code study.

## Usage
```
/project:evaluate-submission <week> <participant_id>
```

## Process

### Step 1: Load Rubric
Read `rubrics/week{N}_rubric.md` for scoring criteria.

### Step 2: Build Verification (REQUIRED)
**You MUST run these commands to verify the code builds:**

```bash
cd submissions/week{N}/{participant_id}
npm install --cache /tmp/npm-cache
npm run build
```

**Build Results:**
- If `npm install` fails: Stop evaluation, report error
- If `npm run build` fails: Apply 50% penalty to Stage scores as per rubric
- Record build_status in output JSON

### Step 3: Code Review
After build verification, review the code according to rubric criteria:
- Check for required features (buttons, dialogs, handlers)
- Verify correct imports and implementations
- Review CLAUDE.md quality

### Step 4: Score Each Category
Apply rubric criteria systematically. Consider:
- Build status (pass/fail)
- Stage 1: Clear All Files implementation
- Stage 2: Download ZIP implementation
- Stage 3: Keyboard Shortcuts (bonus)
- CLAUDE.md quality

### Step 5: Output JSON
Print evaluation result in exact format below.

## Output Format (MUST be valid JSON)

```json
{
  "participant": "<participant_id>",
  "week": <week_number>,
  "status": "completed",
  "scores": {
    "rubric": <number 0-80>,
    "time_rank": <number>,
    "time_rank_bonus": <number>,
    "total": <number>
  },
  "build_status": {
    "npm_install": "pass" | "fail",
    "npm_build": "pass" | "fail",
    "build_error": "<error message if failed>"
  },
  "breakdown": {
    "stage_1_clear_all": <0-20>,
    "stage_2_download_zip": <0-25>,
    "stage_3_keyboard": <0-20>,
    "claude_md_quality": <0-15>
  },
  "feedback": "<2-3 sentence overall feedback>",
  "strengths": ["<strength1>", "<strength2>"],
  "improvements": ["<improvement1>", "<improvement2>"],
  "evaluated_at": "<ISO timestamp>"
}
```

## Important Rules

- **ALWAYS run npm install and npm run build first**
- Be fair and consistent
- Follow rubric exactly
- Maximum rubric score is 80 (time bonus added separately, up to 20)
- Provide actionable feedback
- Output ONLY the JSON, no other text
- If build fails, apply 50% penalty to stage scores
