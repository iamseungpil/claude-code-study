# Week 1 Rubric: Project Folder Cleanup

## Challenge Summary
Given a messy ML project folder, analyze its purpose, identify important files, remove clutter, and write clear documentation.

## Time Limit
45 minutes

## Required Outputs
1. `analysis.md` - Project analysis report
2. `README.md` - Updated project documentation

---

## Scoring Criteria (90 points + 10 time bonus)

### 1. Project Analysis (40 points)

| Points | Criteria |
|--------|----------|
| 10 | **Purpose Identification**: Correctly identifies what the project does |
| 10 | **Core Files**: Identifies the main entry point and essential files |
| 10 | **Dependency Analysis**: Maps file relationships correctly |
| 10 | **Cleanup Recommendations**: Lists files safe to delete with reasons |

**Evaluation Questions:**
- Did they understand the project's purpose?
- Did they identify which `train*.py` is the real one?
- Did they trace imports and dependencies?
- Are cleanup suggestions safe and well-reasoned?

### 2. Cleanup Quality (30 points)

| Points | Criteria |
|--------|----------|
| 10 | **Accuracy**: No important files marked for deletion |
| 10 | **Completeness**: Found most unnecessary files |
| 10 | **Reasoning**: Each deletion has clear justification |

**Red Flags (deduct points):**
- Suggesting deletion of actually-used files (-5 each)
- Missing obvious duplicates (-3 each)
- No justification for deletions (-5)

### 3. README Quality (20 points)

| Points | Criteria |
|--------|----------|
| 5 | **Project Description**: Clear 1-2 sentence summary |
| 5 | **Usage Instructions**: How to run the project |
| 5 | **File Structure**: Documents what each file does |
| 5 | **Requirements**: Lists dependencies |

**Good README includes:**
- What the project does
- How to install and run
- What each important file is for

---

## Output JSON Format

```json
{
  "rubric_score": 75,
  "breakdown": {
    "analysis": 35,
    "cleanup": 25,
    "readme": 15
  },
  "feedback": "Strong analysis of project structure. Correctly identified train_final_real.py as the main script. README could include more usage examples.",
  "strengths": [
    "Accurate identification of core files",
    "Good dependency tracing"
  ],
  "improvements": [
    "Add installation instructions to README",
    "Include example commands"
  ]
}
```

---

## Evaluation Checklist

### Analysis (check each)
- [ ] Project purpose clearly stated
- [ ] Main entry point identified
- [ ] File dependencies mapped
- [ ] Duplicates identified
- [ ] Temp/cache files identified

### Cleanup (check each)
- [ ] No false positives (important files marked for deletion)
- [ ] Duplicates flagged
- [ ] Old versions flagged
- [ ] Cache/temp flagged
- [ ] Each deletion has reason

### README (check each)
- [ ] Project description exists
- [ ] Usage/run instructions exist
- [ ] File structure documented
- [ ] Dependencies listed
