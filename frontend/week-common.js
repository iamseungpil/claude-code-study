/**
 * week-common.js - Shared JavaScript for week challenge pages (week1-5.html)
 *
 * Usage: Each weekN.html must define WEEK_CONFIG before loading this script.
 *
 * Required WEEK_CONFIG properties:
 *   week: number (1-5)
 *   allowResubmit: boolean
 *   hasChallengeStatusCheck: boolean
 *   hasEvaluation: boolean
 *   revealSectionIds: string[]
 *   downloadBtn: { id: string, url: string } | null
 *   leaderboardBox: { bgClass, borderClass, textClass, btnBgClass, btnHoverClass, btnTextClass }
 *
 * Load order: config.js → <script>WEEK_CONFIG = {...}</script> → week-common.js
 */

const API_BASE = window.API_BASE || '';
const WEEK = WEEK_CONFIG.week;

let timerInterval = null;
let currentUser = null;
let personalStartTime = null;

// === Authentication ===

async function checkLoginStatus() {
    const token = localStorage.getItem('auth_token');

    if (!token) {
        alert('Please login to access the challenge page.');
        window.location.href = '/';
        return false;
    }

    try {
        const response = await fetch(`${API_BASE}/api/auth/me`, {
            headers: { 'Authorization': `Bearer ${token}` }
        });

        if (response.ok) {
            currentUser = await response.json();
            const loginWarning = document.getElementById('login-warning');
            const startBtn = document.getElementById('startBtn');
            const submitBtn = document.getElementById('submitBtn');
            if (loginWarning) loginWarning.classList.add('hidden');
            if (startBtn) {
                startBtn.disabled = false;
                startBtn.classList.remove('opacity-50', 'cursor-not-allowed');
            }
            if (submitBtn) {
                submitBtn.disabled = false;
                submitBtn.classList.remove('opacity-50', 'cursor-not-allowed');
            }
            return true;
        } else {
            localStorage.removeItem('auth_token');
            alert('Your session has expired. Please login again.');
            window.location.href = '/';
            return false;
        }
    } catch (e) {
        console.error('Auth check failed:', e);
        alert('Authentication failed. Please login again.');
        window.location.href = '/';
        return false;
    }
}

// === Challenge Status (week3-5 pattern) ===

async function loadChallengeStatus() {
    try {
        const response = await fetch(`${API_BASE}/api/challenge/${WEEK}/status`);
        if (response.ok) {
            const data = await response.json();

            if (data.status !== 'started') {
                alert(`Week ${WEEK} Challenge has not started yet. Please wait for the admin to start the challenge.`);
                window.location.href = '/';
                return;
            }
        }
    } catch (e) {
        console.error('Failed to check challenge status:', e);
    }
}

// === Personal Status ===

async function loadPersonalStatus() {
    const token = localStorage.getItem('auth_token');
    if (!token || !currentUser) return;

    try {
        const response = await fetch(`${API_BASE}/api/challenge/${WEEK}/my-status`, {
            headers: { 'Authorization': `Bearer ${token}` }
        });

        if (response.ok) {
            const data = await response.json();

            if (data.challenge_status !== 'started') {
                if (!WEEK_CONFIG.hasChallengeStatusCheck) {
                    // week1/2: handle challenge not started here
                    alert(`Week ${WEEK} Challenge has not started yet. Please wait for the admin to start the challenge.`);
                    window.location.href = '/';
                }
                return;
            }

            if (data.personal_status === 'in_progress' || data.personal_status === 'submitted') {
                personalStartTime = data.personal_start_time;

                document.getElementById('timerContainer').classList.remove('hidden');
                document.getElementById('startBtn').disabled = true;
                document.getElementById('startBtn').classList.add('opacity-50', 'cursor-not-allowed');
                document.getElementById('startBtn').innerHTML = 'Timer Started';

                timerInterval = setInterval(updateTimer, 1000);
                updateTimer();

                showChallengeForm();

                if (data.personal_status === 'submitted') {
                    if (WEEK_CONFIG.allowResubmit) {
                        document.getElementById('submitBtn').innerHTML = '🔄 Resubmit';
                        showStatus('You have already submitted. You can resubmit to improve your score (time rank will be recalculated).', 'text-blue-400');
                        // week1: check existing evaluation
                        if (WEEK_CONFIG.hasEvaluation && typeof checkExistingEvaluation === 'function') {
                            await checkExistingEvaluation();
                        }
                    } else {
                        document.getElementById('submitBtn').disabled = true;
                        document.getElementById('submitBtn').classList.add('opacity-50', 'cursor-not-allowed');
                        document.getElementById('submitBtn').innerHTML = 'Submitted';
                        showStatus('Already submitted for this week', 'text-green-400');
                    }
                }
            }
        }
    } catch (e) {
        console.error('Failed to load personal status:', e);
    }
}

// === DOMContentLoaded ===

document.addEventListener('DOMContentLoaded', async () => {
    // Set download button URL if configured
    if (WEEK_CONFIG.downloadBtn) {
        const downloadEl = document.getElementById(WEEK_CONFIG.downloadBtn.id);
        if (downloadEl && window.API_BASE) {
            downloadEl.href = `${window.API_BASE}${WEEK_CONFIG.downloadBtn.url}`;
        } else if (downloadEl) {
            downloadEl.href = WEEK_CONFIG.downloadBtn.url;
        }
    }

    // Check login first - will redirect if not logged in
    const isLoggedIn = await checkLoginStatus();
    if (!isLoggedIn) return;

    // Check challenge status (week3-5 pattern)
    if (WEEK_CONFIG.hasChallengeStatusCheck) {
        await loadChallengeStatus();
    }

    await loadPersonalStatus();

    // week1: check evaluation before history
    if (WEEK_CONFIG.hasEvaluation && currentUser && typeof checkExistingEvaluation === 'function') {
        await checkExistingEvaluation();
    }

    // Load submission history
    await loadSubmissionHistory();
});

// === Timer ===

function updateTimer() {
    if (!personalStartTime) return;
    const startDate = new Date(personalStartTime);
    const elapsed = Math.floor((Date.now() - startDate.getTime()) / 1000);
    const minutes = Math.floor(elapsed / 60).toString().padStart(2, '0');
    const seconds = (elapsed % 60).toString().padStart(2, '0');
    document.getElementById('timerDisplay').textContent = `${minutes}:${seconds}`;
}

// === Show Challenge Form ===

function showChallengeForm() {
    // Hide the "Challenge Details Locked" message
    const beforeTimerMsg = document.getElementById('beforeTimerMessage');
    if (beforeTimerMsg) beforeTimerMsg.classList.add('hidden');

    // Show hidden sections
    WEEK_CONFIG.revealSectionIds.forEach(id => {
        const el = document.getElementById(id);
        if (el) el.classList.remove('hidden');
    });

    // Show submission form elements
    const instructions = document.getElementById('instructionsContainer');
    const githubUrl = document.getElementById('githubUrlContainer');
    const submitBtn = document.getElementById('submitBtn');
    if (instructions) instructions.classList.remove('hidden');
    if (githubUrl) githubUrl.classList.remove('hidden');
    if (submitBtn) submitBtn.classList.remove('hidden');

    const hint = document.getElementById('timerNotStartedHint');
    if (hint) hint.classList.add('hidden');
}

// === Start Challenge ===

async function startChallenge() {
    if (!currentUser) {
        showStatus('Please login first', 'text-amber-400');
        return;
    }

    const token = localStorage.getItem('auth_token');

    try {
        const response = await fetch(`${API_BASE}/api/challenge/${WEEK}/start-personal`, {
            method: 'POST',
            headers: { 'Authorization': `Bearer ${token}` }
        });

        if (response.ok) {
            const data = await response.json();
            personalStartTime = data.started_at;

            document.getElementById('timerContainer').classList.remove('hidden');
            timerInterval = setInterval(updateTimer, 1000);
            updateTimer();

            document.getElementById('startBtn').disabled = true;
            document.getElementById('startBtn').classList.add('opacity-50', 'cursor-not-allowed');
            document.getElementById('startBtn').innerHTML = 'Timer Started';

            showChallengeForm();

            showStatus('Timer started! Good luck!', 'text-green-400');
        } else {
            const error = await response.json();
            showStatus(error.detail || 'Failed to start timer', 'text-red-400');
        }
    } catch (e) {
        showStatus('Connection error', 'text-red-400');
    }
}

// === Submit Handler ===

document.getElementById('submitForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    if (!currentUser) {
        showStatus('Please login first', 'text-amber-400');
        return;
    }

    const githubUrl = document.getElementById('githubUrl').value;

    if (!githubUrl) {
        showStatus('Please enter GitHub URL', 'text-amber-400');
        return;
    }

    try {
        const token = localStorage.getItem('auth_token');
        const response = await fetch(`${API_BASE}/api/submissions/submit`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
            },
            body: JSON.stringify({
                week: WEEK,
                github_url: githubUrl
            })
        });

        // Timer continues running after submit (don't stop it)

        if (response.ok) {
            const data = await response.json();
            const submissionNum = data.submission_number || 1;
            const isResubmission = data.is_resubmission || submissionNum > 1;

            // Show success message with leaderboard link (colors from config)
            const lb = WEEK_CONFIG.leaderboardBox;
            const statusEl = document.getElementById('statusMessage');
            statusEl.innerHTML = `
                <div class="text-green-400 font-medium mb-2">
                    ${isResubmission ? `Resubmission #${submissionNum} complete!` : 'Submission complete!'}
                </div>
                <div class="text-gray-300 mb-3">
                    Elapsed time: <span class="font-bold">${data.elapsed_minutes?.toFixed(1) || '-'}</span> minutes
                </div>
                <div class="${lb.bgClass} border ${lb.borderClass} rounded-xl p-4 inline-block">
                    <div class="${lb.textClass} mb-2">Your entry is now on the leaderboard!</div>
                    <a href="leaderboard" class="inline-flex items-center gap-2 ${lb.btnBgClass} hover:${lb.btnHoverClass} ${lb.btnTextClass} font-medium px-4 py-2 rounded-lg transition-all">
                        <span>🏆</span> View Leaderboard
                    </a>
                </div>
                <div class="text-gray-500 text-sm mt-3">
                    Evaluation in progress... Results will appear shortly.
                </div>
            `;
            statusEl.className = 'mt-6 text-center';
            statusEl.classList.remove('hidden');

            if (WEEK_CONFIG.allowResubmit) {
                document.getElementById('submitBtn').disabled = false;
                document.getElementById('submitBtn').classList.remove('opacity-50', 'cursor-not-allowed');
            }
            document.getElementById('submitBtn').innerHTML = '🔄 Resubmit';

            // Refresh submission history
            await loadSubmissionHistory();
        } else {
            const error = await response.json();
            if (response.status === 401) {
                localStorage.removeItem('auth_token');
                alert('Your session has expired. Please login again.');
                window.location.href = '/';
                return;
            }
            showStatus('Submission failed: ' + (error.detail || 'Unknown error'), 'text-red-400');
        }

    } catch (error) {
        showStatus('Submission failed: ' + error.message, 'text-red-400');
    }
});

// === Show Status ===

function showStatus(message, colorClass) {
    const el = document.getElementById('statusMessage');
    el.textContent = message;
    el.className = `mt-6 text-center text-lg font-medium ${colorClass}`;
    el.classList.remove('hidden');
}

// === Submission History ===

async function loadSubmissionHistory() {
    if (!currentUser) return;

    try {
        const response = await fetch(`${API_BASE}/api/submissions/${WEEK}/${currentUser.user_id}/history`);
        if (response.ok) {
            const data = await response.json();
            displaySubmissionHistory(data.submission_history || []);
        }
    } catch (error) {
        console.error('Failed to load submission history:', error);
    }
}

/**
 * Default displaySubmissionHistory (used by week2-5).
 * week1 overrides this function in its inline script.
 */
function displaySubmissionHistory(history) {
    const container = document.getElementById('submissionHistory');
    const list = document.getElementById('submissionHistoryList');

    if (!history || history.length === 0) {
        container.classList.add('hidden');
        return;
    }

    container.classList.remove('hidden');
    list.innerHTML = '';

    const ordinals = ['1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th', '9th', '10th'];

    history.forEach((submission, index) => {
        const isLatest = index === history.length - 1;
        const ordinal = ordinals[index] || `${index + 1}th`;
        const submittedAt = new Date(submission.submitted_at);
        const timeStr = submittedAt.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
        const dateStr = submittedAt.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });

        const card = document.createElement('div');
        card.className = `p-4 rounded-xl ${isLatest ? 'bg-accent/10 border border-accent/30' : 'bg-dark-800/50 border border-white/5'}`;

        let scoreHtml = '';
        const evalData = submission.evaluation;
        if (evalData && evalData.total !== undefined && evalData.total !== null) {
            scoreHtml = `
                <div class="mt-3 pt-3 border-t border-white/10">
                    <div class="flex justify-between items-center">
                        <span class="text-gray-400 text-sm">Score</span>
                        <span class="font-bold text-lg ${evalData.total >= 80 ? 'text-green-400' : evalData.total >= 60 ? 'text-accent' : 'text-amber-400'}">
                            ${evalData.total} pts
                        </span>
                    </div>
                    <div class="text-xs text-gray-500 mt-1">
                        Rubric: ${evalData.rubric || 0} + Time Bonus: ${evalData.time_rank_bonus || 0}
                    </div>
                </div>
            `;
        } else if (isLatest) {
            scoreHtml = '<div class="mt-3 pt-3 border-t border-white/10 text-sm text-amber-400">⏳ Pending review</div>';
        }

        card.innerHTML = `
            <div class="flex justify-between items-start">
                <div>
                    <div class="flex items-center gap-2">
                        <span class="font-semibold ${isLatest ? 'text-accent' : 'text-gray-300'}">${ordinal} Try</span>
                        ${isLatest ? '<span class="text-xs bg-accent/20 text-accent px-2 py-0.5 rounded-full">Latest</span>' : ''}
                    </div>
                    <div class="text-sm text-gray-500 mt-1">${dateStr} at ${timeStr}</div>
                </div>
                <div class="text-right">
                    <div class="font-mono text-lg ${isLatest ? 'text-white' : 'text-gray-400'}">${submission.elapsed_minutes?.toFixed(1) || '-'} min</div>
                </div>
            </div>
            ${scoreHtml}
        `;

        list.appendChild(card);
    });
}
