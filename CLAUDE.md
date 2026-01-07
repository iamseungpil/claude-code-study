# Claude Code Study Evaluation System

## Purpose
Automated evaluation system for 5-week Claude Code study group.

## Key Directories
- `submissions/weekN/` - Participant submissions (git clone)
- `evaluations/weekN/` - Evaluation results (JSON)
- `rubrics/` - Week-specific scoring criteria
- `challenges/` - Challenge materials for participants

## Evaluation Flow
1. Participant submits GitHub URL
2. System clones to `submissions/weekN/participant_id/`
3. Run: `claude -p "evaluate-submission weekN participant_id"`
4. Result saved to `evaluations/weekN/participant_id.json`

## Scoring Formula
```
total = rubric_score + time_bonus
time_bonus = +10 (≤70%) | +5 (≤85%) | 0 (on time) | -5/5min (late)
```

## Commands
- `/project:evaluate-submission <week> <participant>` - Evaluate one
- `/project:evaluate-all <week>` - Evaluate all pending

## Important
- Read `rubrics/weekN_rubric.md` before evaluating
- Output JSON to `evaluations/weekN/participant_id.json`
- Be fair and consistent across all participants

## Known Issues & Fixes (2026-01-07)

### Issue 1: Port Conflict with VS Code Helper
- **Cause**: VS Code Helper (PID 821) occupies `localhost:8001` and `localhost:8002` for internal communication (vcom-tunnel, teradataordbms)
- **Solution**: Use port **8003** (or higher) which VS Code doesn't use
- **How to check free ports**:
  ```bash
  for port in 8003 8080 9000; do
    lsof -i :$port > /dev/null 2>&1 || echo "Port $port is FREE"
  done
  ```
- **Files Modified**:
  - `backend/server.py` (port=8003, CORS updated)
  - `frontend/config.js` (port detection logic updated)
- **Important**: Always use port 8003 for local development

### Issue 2: httpx Exception Handling
- **Cause**: `get_profile_image_url()` had try-except outside AsyncClient initialization
- **Solution**: Wrapped entire function in try-except for defensive programming
- **File Modified**: `backend/server.py` (lines 227-253)

### Issue 3: Leaderboard API Route Order
- **Cause**: FastAPI route order - `/api/leaderboard/{week}` (dynamic) was defined before `/api/leaderboard/season` (static), causing "season" to be parsed as integer
- **Solution**: Moved `/api/leaderboard/season` route BEFORE `/{week}` route, created helper function `_get_week_leaderboard_data()`
- **File Modified**: `backend/server.py` (lines 717-822)

### Issue 4: Profile Image Not Setting for sundong.kim Members
- **Cause**: Users registered with old code version (before `last_name` parameter was added)
- **Solution**: `get_profile_image_url()` now uses `firstname+lastname` pattern (e.g., "seungpillee.png")
- **Note**: Existing users need to re-register to get profile image
