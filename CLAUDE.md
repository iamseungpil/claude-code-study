# Claude Code Study Evaluation System

## Purpose
Automated evaluation system for 5-week Claude Code study group.

## Architecture

### Deployment Modes
```
┌─────────────────────────────────────────────────────────────────┐
│                    Local Development                            │
│  ┌──────────────┐     ┌──────────────┐                         │
│  │   Browser    │────▶│   FastAPI    │  localhost:8003         │
│  │              │◀────│   Backend    │  (serves frontend too)  │
│  └──────────────┘     └──────────────┘                         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│              Production (Cloudflare + Local Backend)            │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐    │
│  │   Browser    │────▶│  Cloudflare  │────▶│   FastAPI    │    │
│  │              │◀────│    Pages     │◀────│   Backend    │    │
│  └──────────────┘     └──────────────┘     └──────────────┘    │
│                        (frontend)      Cloudflare (local:8003) │
│                                         Tunnel                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components
- **Frontend**: Static HTML/JS served from `frontend/` directory
- **Backend**: FastAPI server (`backend/server.py`) on port 8003
- **Config**: `frontend/config.js` - API base URL detection
- **Data**: `data/users.json`, `data/challenges.json` (gitignored)

### Starting Local Server
```bash
# Start backend (serves both API and frontend)
python backend/server.py

# Or using uvicorn directly
uvicorn backend.server:app --host 0.0.0.0 --port 8003

# Access at http://localhost:8003
```

### Cloudflare Tunnel (for remote access)
```bash
# Expose local backend via Cloudflare Tunnel
cloudflared tunnel --url http://localhost:8003

# Copy generated URL (e.g., https://xxx.trycloudflare.com)
# Update frontend/config.js: CONFIGURED_API_BASE = 'https://xxx.trycloudflare.com'
```

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

## Known Issues & Fixes (2026-01-19)

### Issue 5: /index.html 404 Error
- **Cause**: FastAPI had route for `/` but NOT for `/index.html`
- **Symptom**: Clicking "login" link from week1.html caused connection error
- **Solution**: Added explicit `/index.html` route in `backend/server.py`
- **File Modified**: `backend/server.py` (line ~1077)

### Issue 6: config.js 404 Error (Login Failure)
- **Cause**: No route for static JS files in FastAPI
- **Symptom**: `API_BASE` undefined, all API calls failed
- **Solution**: Added routes for `config.js` and all `week*-learn.html` files
- **File Modified**: `backend/server.py` (lines ~1117-1141)
- **Lesson**: When using FileResponse for static files, ALL files need explicit routes

### Issue 7: Challenge Sections Hidden Before Timer Start
- **Cause**: `stagesSection`, `claudemdSection`, `scoringSection` had `class="hidden"`
- **Symptom**: Users couldn't see what to implement before starting timer
- **Solution**: Removed `hidden` class from all three sections
- **Files Modified**: `frontend/week1.html`, `frontend/week2.html`

### Issue 8: npm Cache Permission Error
- **Cause**: npm cache folder had root-owned files
- **Symptom**: `EACCES: permission denied` during `npm install`
- **Solution**: Use `--cache /tmp/npm-cache` flag or fix permissions with `sudo chown -R $(whoami) ~/.npm`

## Evaluation System Notes

### Build Verification (Required)
The evaluation command (`/project:evaluate-submission`) MUST run:
```bash
cd submissions/weekN/{participant_id}
npm install --cache /tmp/npm-cache
npm run build
```
- Build failure = 50% penalty on stage scores
- Build status recorded in `build_status` field of evaluation JSON

### Evaluation Feedback Display
- After submission, frontend polls `/api/evaluations/{week}/{participant_id}`
- Displays score breakdown, feedback, strengths, improvements
- Polling interval: 10 seconds, max 30 attempts (5 minutes)
