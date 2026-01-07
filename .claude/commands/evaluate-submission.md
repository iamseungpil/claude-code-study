# Evaluate Submission

Evaluate a participant's submission for the Claude Code study.

## Usage
```
/project:evaluate-submission <week> <participant_id>
```

## Process

1. **Load Rubric**: Read `rubrics/week{N}_rubric.md` for scoring criteria

2. **Analyze Submission**: Review files in `submissions/week{N}/{participant_id}/`

3. **Score Each Category**: Apply rubric criteria systematically

4. **Generate Feedback**: Provide constructive feedback

5. **Output JSON**: Print evaluation result in exact format below

## Output Format (MUST be valid JSON)

```json
{
  "rubric_score": <number 0-90>,
  "breakdown": {
    "<category1>": <score>,
    "<category2>": <score>,
    "<category3>": <score>
  },
  "feedback": "<2-3 sentence overall feedback>",
  "strengths": ["<strength1>", "<strength2>"],
  "improvements": ["<improvement1>", "<improvement2>"]
}
```

## Important Rules

- Be fair and consistent
- Follow rubric exactly
- Maximum rubric_score is 90 (time bonus added separately)
- Provide actionable feedback
- Output ONLY the JSON, no other text
