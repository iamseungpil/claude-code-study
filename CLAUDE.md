# Claude Code Study Platform

## Purpose
Study group platform for a 5-week Claude Code curriculum (submissions + leaderboard + admin review).

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

## Review Flow
1. Participant submits a GitHub URL
2. System clones to `submissions/weekN/participant_id/`
3. System creates/updates a pending review record in `evaluations/weekN/participant_id.json`
4. Admin reviews the submission and saves the score/feedback (manual)

## Scoring Formula
```
total = rubric_score + time_bonus
time_bonus = +10 (≤70%) | +5 (≤85%) | 0 (on time) | -5/5min (late)
```

## Admin Review
- Manual review is submitted via the admin panel (UI) and stored as JSON in `evaluations/weekN/`.

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

## Review System Notes

### Review Method
- Review is manual: admins read the submission and enter rubric scores + feedback.

### Review Display
- After submission, the UI shows "Pending review" until an admin saves a completed review.

## Current Deployment Setup (2026-01-20)

### Architecture
```
Frontend (Cloudflare Pages) → Cloudflare Tunnel → Local Backend (localhost:8003)
```

### Why Local Backend?
- Local backend stores submissions and supports admin review workflows
- Cloudflare Tunnel provides secure remote access
- No cold start delays
- SQLite database for persistent storage

### Running the System
```bash
# 1. Start local backend
cd /Users/iamseungpil/LSP/claude-code-study
python backend/server.py

# 2. Start Cloudflare Tunnel (in another terminal)
cloudflared tunnel --url http://localhost:8003

# 3. Update frontend/config.js with the tunnel URL
```

### Data Storage
- **Local**: `data/`, `submissions/`, `evaluations/` directories
- **Cloudflare Pages**: Serves static frontend files only
- Submissions are cloned to `submissions/weekN/participant_id/code/`

### Troubleshooting
- Check backend logs: `tail -f /tmp/backend.log`
- Restart backend: `pkill -f "python backend/server.py" && python backend/server.py`

### Challenge Page Access Rules (IMPORTANT)
- **Non-started challenges (week 2-5)**: Redirecting to index.html is CORRECT behavior
- Users should NOT be able to access challenge pages before admin starts them
- Only fix if a STARTED challenge is not accessible

### JWT_SECRET Configuration (IMPORTANT)
- The backend uses JWT for authentication
- **Without a persistent JWT_SECRET**, tokens become invalid after server restart
- **Solution**: Create `.env` file with `JWT_SECRET=<random-hex-string>`
- Generate: `openssl rand -hex 32`
- The backend loads `.env` automatically using python-dotenv
- `.env` is gitignored - never commit secrets!

### Issue 9: Cloudflare Pages Redirect Loop (2026-01-20)
- **Cause**: Two conflicting `_redirects` files in the repository
  - Root `/_redirects`: `/index.html` → `/frontend/index.html` (301)
  - Frontend `/frontend/_redirects`: `/frontend/*` → `/index.html` (301)
- **Symptom**: "ERR_TOO_MANY_REDIRECTS" when accessing claude-code-study.pages.dev
- **Solution**: Delete the root `_redirects` file, keep only `frontend/_redirects`
- **IMPORTANT**: Cloudflare Pages build output is set to `frontend/` directory
  - Files are served at root: `/index.html`, `/week1.html`, etc.
  - The `frontend/_redirects` becomes `/_redirects` after deployment
  - NEVER create a root `_redirects` file - it will conflict!

### Issue 10: Cloudflare Pages Pretty URLs Conflict (2026-01-20)
- **Cause**: Cloudflare Pages "Pretty URLs" feature auto-redirects `.html` requests to clean URLs
  - `_redirects` rules with `200` status caused redirect loop when combined with Pretty URLs
  - Example: `/leaderboard → /leaderboard.html 200` + Pretty URLs = loop
- **Symptom**: HTML pages return 308 redirects to themselves or 404 errors
- **Solution**:
  1. Use forced rewrites (`200!`) instead of normal rewrites (`200`) in `_redirects`
  2. Update all internal links to use clean URLs (e.g., `href="leaderboard"` not `href="leaderboard.html"`)
- **Files Modified**:
  - `frontend/_redirects` - Changed all `200` to `200!`
  - All HTML files - Updated internal links to clean URLs

### Issue 11: JavaScript Redirect Using index.html (2026-01-20)
- **Cause**: Challenge pages had `window.location.href = 'index.html'` for login redirects
  - Cloudflare Pages returns 308 redirect for `index.html` → `/`
  - This causes unnecessary redirect chain
- **Symptom**: Non-logged-in users redirected to `/index` URL instead of `/`
- **Solution**: Changed all JavaScript redirects from `'index.html'` to `'/'`
- **Files Modified**: All `week*.html` files

### Issue 12: Windows Encoding Error in Evaluation (2026-01-29)
- Legacy note: this issue occurred when evaluation output contained non-ASCII characters.
- Current system still enforces `encoding='utf-8'` for server-side file I/O.

### Issue 13: Windows Encoding Error in Server API (2026-01-29)
- **Cause**: `server.py` had 27 `open()` calls without `encoding='utf-8'` parameter
  - Leaderboard API failed when reading evaluation files saved with Korean text
- **Symptom**:
  - `UnicodeDecodeError: 'cp949' codec can't decode byte 0xec` (read without encoding)
  - `UnicodeDecodeError: 'utf-8' codec can't decode byte 0xb8` (file saved as cp949)
- **Solution**:
  1. Added `encoding='utf-8'` to all 27 `open()` calls in server.py
  2. Converted existing cp949-encoded JSON files to UTF-8
- **File Modified**: `backend/server.py`

### Issue 14: Claude CLI Returning Text Instead of JSON (2026-01-29)
- Legacy note: this issue only applied to the removed automated Claude-based evaluation pipeline.

### Cloudflare Pages Configuration (IMPORTANT)
- **Build output directory**: `frontend`
- **Files served at**: Root paths (`/index.html`, `/week1.html`, etc.)
- **Only ONE `_redirects` file**: `frontend/_redirects`
- **DO NOT** create `_redirects` at repository root
- **Use forced rewrites** (`200!`) to avoid Pretty URLs conflict
- **Use clean URLs** in all internal links (no `.html` extension)
- **Use `/` for JavaScript redirects** to home page (not `index.html`)

## Timer & Submission System (2026-01-20)

### Timer Behavior
- Timer starts when user clicks "Start Timer" and records `personal_start_time`
- Timer uses start/end time calculation (not counting seconds)
- **Timer continues running after submit** - does NOT stop on submission
- Elapsed time = `current_time - personal_start_time`

### Submission History
- Backend tracks all submissions in `metadata.json` under `submission_history` array
- Each submission records: `submission_number`, `github_url`, `submitted_at`, `elapsed_seconds`, `elapsed_minutes`
- API endpoint: `GET /api/submissions/{week}/{participant_id}/history`
- Frontend displays submission history with "1st Try", "2nd Try", etc.
- Latest submission highlighted with evaluation scores when available

### Resubmission Flow
1. User can resubmit anytime (button changes to "🔄 Resubmit")
2. New submission is appended to `submission_history`
3. The submission becomes "Pending review" until an admin completes the review
4. Time rank bonus is calculated based on the latest submission time

## E2E Testing with Playwright (2026-01-20)

### Test Setup
- **Location**: `tests/week1/`
- **Framework**: Playwright (TypeScript)
- **Config**: `playwright.config.ts`

### Running Tests
```bash
cd tests/week1
npm install
npx playwright install chromium
npx playwright test registration.spec.ts --project=site-tests
```

### Test Files
- `site.spec.ts` - General site functionality tests
- `registration.spec.ts` - User registration and challenge participation tests
- `uigen.spec.ts` - UIGen project evaluation tests

### E2E Test Results (2026-01-20)
- **Registration Test**: PASSED (user `iamseungpil` already exists)
- **Login Test**: PASSED
- **Challenge Page Navigation**: PASSED
- **Challenge Submission**: PASSED

### Current Tunnel URL (2026-01-29)
- **URL**: `https://stage-present-dosage-unlike.trycloudflare.com`
- **Protocol**: HTTP/2 (more stable than QUIC on Windows)
- **Note**: Tunnel URL changes on each restart. Update `frontend/config.js` AND GitHub webhook!

### Webhook Update Hook (IMPORTANT)
- **Location**: `.claude/hooks/update-webhook.sh`
- **Purpose**: Reminds you to update GitHub webhook when tunnel URL changes
- **Usage**: `./.claude/hooks/update-webhook.sh`
- **What it does**:
  1. Reads current Cloudflare Tunnel URL from `cloudflared.log`
  2. Compares with URL in `frontend/config.js`
  3. Displays step-by-step instructions for updating GitHub webhook
- **When to run**:
  - After restarting cloudflared tunnel
  - When seeing GitHub webhook connection errors
  - Before pushing critical updates

### GitHub Webhook Configuration
- **Endpoint**: `https://[tunnel-url]/webhook/github`
- **Current**: `https://stage-present-dosage-unlike.trycloudflare.com/webhook/github`
- **Secret**: Stored in `backend/.env.webhook`
- **Events**: Push events only
- **Update manually at**: https://github.com/iamseungpil/claude-code-study/settings/hooks
- **CRITICAL**: Must update webhook URL every time tunnel restarts (Quick Tunnels use random URLs)

### Cloudflare Tunnel Tips (Windows)
- Use `--protocol http2` flag for more stable connections
- QUIC protocol may timeout on some networks
- Example: `cloudflared tunnel --url http://localhost:8003 --protocol http2`
- **Production Tip**: Use Named Tunnels for stable URLs (no manual webhook updates needed)

### Auto-Deploy Test: 2026-01-20 22:18:11

### Webhook Test: 2026-01-20 22:21:51

### Auto-Deploy Verified: 2026-01-20 22:25:22

## Frontend Modularity Guidelines (2026-01-31)

### Shared Code Architecture
```
frontend/
├── config.js              # API base URL detection (localhost vs Cloudflare)
├── week-common.js         # Week challenge pages: auth, timer, submission, history
├── week-common.css        # Week challenge pages: nav-blur, card, buttons, timer
├── learn-common.js        # Learn pages: quiz engine, lesson toggle, progress, state
├── learn-common.css       # Learn pages: lesson cards, code blocks, quiz options
├── week1.html … week5.html        # Challenge pages (WEEK_CONFIG + week-common.js)
└── week1-learn.html … week5-learn.html  # Learn pages (LEARN_CONFIG + learn-common.js)
```

### Script Load Order (Critical)
```
config.js → <inline> CONFIG object + data </inline> → common.js
```
- `config.js` sets `window.API_BASE`
- Inline script declares `WEEK_CONFIG` or `LEARN_CONFIG` + page data (e.g., `quizQuestions`)
- `week-common.js` / `learn-common.js` reads the config and initializes behavior

### Adding a New Week Page
1. Copy an existing `weekN.html` and update `WEEK_CONFIG` values (week number, colors, feature flags)
2. Copy an existing `weekN-learn.html` and update:
   - `LEARN_CONFIG` (week, totalLessons, lessonStart, storageKey, challengeUrl, quizLabels, resultEmoji)
   - CSS custom properties (`--accent-rgb`, `--accent-from`, `--accent-to`)
   - `quizQuestions` array
3. No changes needed to common JS/CSS files unless adding new shared behavior

### Week-Specific Overrides
- If a page needs custom behavior, load `week-common.js` first, then override functions in a subsequent inline `<script>` block
- Example: `week1.html` overrides `displaySubmissionHistory()` and adds `checkExistingEvaluation()`

### WEEK_CONFIG Properties
| Property | Type | Description |
|----------|------|-------------|
| `week` | number | Week number (1-5) |
| `allowResubmit` | boolean | Enable resubmit button (week1-2: true) |
| `hasChallengeStatusCheck` | boolean | Check if challenge started (week3-5: true) |
| `hasEvaluation` | boolean | Show evaluation feedback (week1 only) |
| `revealSectionIds` | string[] | Section IDs to show when timer starts |
| `downloadBtn` | object/null | `{ id, url }` for download button, or null |
| `leaderboardBox` | object | Tailwind color classes for leaderboard card |

### LEARN_CONFIG Properties
| Property | Type | Description |
|----------|------|-------------|
| `week` | number | Week number (1-5) |
| `totalLessons` | number | Number of lessons (5 or 6) |
| `lessonStart` | number | First lesson index (0 for week1, 1 for week2-5) |
| `storageKey` | string | sessionStorage key for state persistence |
| `challengeUrl` | string | URL path for challenge button |
| `quizLabels` | object | `{ correct, incorrect, correctAnswer }` display text |
| `resultEmoji` | function | `(score) => string` for quiz result display |

### Backend Helper Functions
- `validate_week(week)`: Validates week parameter is 1-5, raises HTTPException if not
- `get_challenge(week)`: Loads challenges data and returns `(challenges, week_data, week_key)` tuple
- `ensure_submission_history(metadata)`: Ensures backward-compatible `submission_history` array exists
- `ALLOWED_STATIC_ASSETS`: Dict mapping JS/CSS filenames to media types for static serving

## Auto-Deploy via GitHub Actions (2026-02-01)

### How It Works
- **`git push origin main`** triggers GitHub Actions automatically — no manual `wrangler deploy` needed
- Two separate workflows handle Workers and Frontend:

| Path Change | Workflow | Action |
|-------------|----------|--------|
| `workers/**` | `.github/workflows/deploy-worker.yml` | `wrangler deploy` (Workers) |
| `frontend/**` | `.github/workflows/deploy-frontend.yml` | `wrangler pages deploy frontend` (Pages) |

### Deployment Rule
- **ALWAYS commit and push to deploy** — never run `wrangler deploy` manually
- Manual deploy can cause confusion (local deploy succeeds but CI may fail due to uncommitted files)
- Push ensures CI/CD validates the exact same code that gets deployed

### Verifying Deployment
```bash
# Check GitHub Actions status after push
gh run list --limit 3

# View specific run details
gh run view <run-id>
```

### Submission Collection (Central Repo)
- Repo: `iamseungpil/claude-code-study-submissions`
- When a student submits, Workers triggers `workflow_dispatch` → GitHub Actions clones student repo → commits to `weekN/user_id/`
- Requires `GITHUB_PAT` wrangler secret (for triggering dispatch) and `SUBMISSIONS_PAT` repo secret (for cloning private repos)
- Backfill existing submissions: `python backfill_submissions.py --week 1`

## CLI Evaluation Workflow (2026-02-01)

### Architecture
```
Browser → Cloudflare Workers → D1 (SQL)
                ▲
                │ evaluate.py calls Workers API
                │
         Local (Mac)
         1. git clone
         2. claude -p (headless eval)
         3. POST result to Workers
```

- All production data lives in Workers D1 (users, submissions, evaluations)
- Workers cannot run git clone → evaluation runs locally, posts results via API
- Admin API authenticated via `X-Admin-Key` header (wrangler secret)

### Usage
```bash
# Single evaluation
python evaluate.py 1 iamseungpil

# Dry run (preview, no save)
python evaluate.py 1 iamseungpil --dry-run

# All pending for a week
python evaluate.py 1 --all

# Keep cloned repo for debugging
python evaluate.py 1 iamseungpil --keep
```

### Environment Variables
```bash
# In .env (project root)
WORKERS_API_BASE=https://claude-code-study-api.iamseungpil.workers.dev
ADMIN_API_KEY=<same value as wrangler secret>
```

### Evaluation Flow
1. Fetch submission info from Workers API (`GET /api/submissions/{week}`)
2. `git clone --depth 1` to temp directory
3. Read rubric from `rubrics/weekN_rubric.md`
4. Run `claude -p` with rubric + code path → JSON result
5. Clean up temp directory (always, via `try/finally`)
6. POST result to Workers Admin API (`POST /api/admin/evaluations/{week}/{pid}`)
7. Server calculates time_rank_bonus and total_score

### Admin API Key Setup
```bash
# Generate key
openssl rand -hex 32

# Register as wrangler secret (not in wrangler.toml)
cd workers && npx wrangler secret put ADMIN_API_KEY

# Add same key to .env for evaluate.py
```

## Test Accounts

| Account | ID | Password | Purpose |
|---------|-----|----------|---------|
| Test Evaluator | `testeval3` | `testeval3` | E2E testing for evaluation pipeline (week 3) |
