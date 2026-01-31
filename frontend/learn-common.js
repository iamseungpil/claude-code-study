/**
 * learn-common.js - Shared JavaScript for week learn pages (week1-5-learn.html)
 *
 * Usage: Each weekN-learn.html must define LEARN_CONFIG and quizQuestions before loading this script.
 *
 * Required LEARN_CONFIG properties:
 *   week: number (1-5)
 *   totalLessons: number (5 or 6)
 *   lessonStart: number (0 for week1, 1 for week2-5)
 *   storageKey: string (e.g., 'week3_learn_state')
 *   challengeUrl: string (e.g., 'week3')
 *   quizLabels: { correct: string, incorrect: string, correctAnswer: string }
 *   resultEmoji: function(score) -> string
 *
 * Required: quizQuestions array defined before this script loads.
 *
 * Load order: config.js → <script>LEARN_CONFIG = {...}; quizQuestions = [...];</script> → learn-common.js
 */

const API_BASE = window.API_BASE || '';
const WEEK = LEARN_CONFIG.week;

// === Challenge Status Check ===

async function checkChallengeStatus() {
    try {
        const response = await fetch(`${API_BASE}/api/challenge/${WEEK}/status`);
        if (response.ok) {
            const data = await response.json();

            if (data.status !== 'started') {
                // Disable all challenge buttons
                const challengeButtons = [
                    document.getElementById('challengeNavBtn'),
                    document.getElementById('challengeQuizBtn'),
                    document.getElementById('challengeBtn')
                ];

                challengeButtons.forEach(btn => {
                    if (btn) {
                        btn.classList.add('opacity-50', 'cursor-not-allowed', 'pointer-events-none');
                    }
                });

                // Add warning message after main CTA button
                const mainBtn = document.getElementById('challengeBtn');
                if (mainBtn && mainBtn.parentElement) {
                    const existingMsg = mainBtn.parentElement.querySelector('.challenge-warning');
                    if (!existingMsg) {
                        const msg = document.createElement('p');
                        msg.className = 'challenge-warning text-sm text-amber-400 mt-2 text-center w-full';
                        msg.textContent = 'Challenge not started by admin yet';
                        mainBtn.parentElement.appendChild(msg);
                    }
                }
            }
        }
    } catch (e) {
        console.error('Failed to check challenge status:', e);
    }
}

// === State Management ===

let currentQuestion = 0;
let score = 0;
let answeredQuestions = new Array(quizQuestions.length).fill(null);
let completedLessons = new Set();

function loadState() {
    const saved = sessionStorage.getItem(LEARN_CONFIG.storageKey);
    if (saved) {
        const state = JSON.parse(saved);
        completedLessons = new Set(state.completedLessons || []);
        score = state.quizScore || 0;
        answeredQuestions = state.answeredQuestions || new Array(quizQuestions.length).fill(null);
        currentQuestion = state.currentQuestion || 0;
    }
    updateProgress();
    updateLessonStatuses();
    renderQuiz();
}

function saveState() {
    const state = {
        completedLessons: Array.from(completedLessons),
        quizScore: score,
        answeredQuestions: answeredQuestions,
        currentQuestion: currentQuestion
    };
    sessionStorage.setItem(LEARN_CONFIG.storageKey, JSON.stringify(state));
}

// === Lesson Functions ===

function toggleLesson(num) {
    const lesson = document.getElementById(`lesson${num}`);
    const content = lesson.querySelector('.lesson-content');
    const chevron = lesson.querySelector('.chevron');

    document.querySelectorAll('.lesson-card').forEach(card => {
        if (card.id !== `lesson${num}`) {
            card.classList.remove('open');
            card.querySelector('.lesson-content').classList.remove('open');
            card.querySelector('.chevron').classList.remove('open');
        }
    });

    lesson.classList.toggle('open');
    content.classList.toggle('open');
    chevron.classList.toggle('open');
}

function completeLesson(num) {
    completedLessons.add(num);
    saveState();
    updateLessonStatuses();
    updateProgress();

    const lastLesson = LEARN_CONFIG.lessonStart + LEARN_CONFIG.totalLessons - 1;
    if (num < lastLesson) {
        setTimeout(() => toggleLesson(num + 1), 300);
    }
}

function updateLessonStatuses() {
    const start = LEARN_CONFIG.lessonStart;
    const end = start + LEARN_CONFIG.totalLessons;
    for (let i = start; i < end; i++) {
        const status = document.getElementById(`lesson${i}-status`);
        if (status && completedLessons.has(i)) {
            status.textContent = 'Completed';
            status.className = 'text-xs px-3 py-1 rounded-full bg-green-500/20 text-green-400';
        }
    }
}

function updateProgress() {
    const total = LEARN_CONFIG.totalLessons;
    const lessonProgress = completedLessons.size;
    const progressPercent = (lessonProgress / total) * 100;

    document.getElementById('progressText').textContent = `${lessonProgress}/${total} Lessons`;
    document.getElementById('progressFill').style.width = `${progressPercent}%`;
    document.getElementById('quizScoreDisplay').textContent = score;

    if (lessonProgress === total && score >= 7) {
        document.getElementById('completionStatus').classList.remove('hidden');
    }
}

// === Quiz Functions ===

function renderQuiz() {
    const container = document.getElementById('quizContainer');
    const results = document.getElementById('quizResults');

    if (answeredQuestions.every(a => a !== null)) {
        container.classList.add('hidden');
        results.classList.remove('hidden');
        document.getElementById('finalScore').textContent = score;
        document.getElementById('resultEmoji').textContent = LEARN_CONFIG.resultEmoji(score);
        return;
    }

    container.classList.remove('hidden');
    results.classList.add('hidden');

    const q = quizQuestions[currentQuestion];
    const labels = LEARN_CONFIG.quizLabels;
    document.getElementById('quizProgress').textContent = `Question ${currentQuestion + 1}/${quizQuestions.length}`;

    container.innerHTML = `
        <div class="mb-6">
            <p class="text-lg font-medium mb-6">${q.question}</p>
            <div class="space-y-3">
                ${q.options.map((opt, i) => `
                    <div class="quiz-option card rounded-xl p-4 ${answeredQuestions[currentQuestion] === i ? (i === q.correct ? 'correct' : 'incorrect') : ''}"
                         onclick="selectAnswer(${i})" ${answeredQuestions[currentQuestion] !== null ? 'style="pointer-events: none;"' : ''}>
                        <div class="flex items-center gap-3">
                            <div class="w-8 h-8 rounded-lg bg-dark-600 flex items-center justify-center text-sm font-bold">
                                ${String.fromCharCode(65 + i)}
                            </div>
                            <span>${opt}</span>
                            ${answeredQuestions[currentQuestion] === i ? (i === q.correct ? `<span class="ml-auto text-green-400">${labels.correct}</span>` : `<span class="ml-auto text-red-400">${labels.incorrect}</span>`) : ''}
                            ${answeredQuestions[currentQuestion] !== null && i === q.correct && answeredQuestions[currentQuestion] !== i ? `<span class="ml-auto text-green-400">${labels.correctAnswer}</span>` : ''}
                        </div>
                    </div>
                `).join('')}
            </div>
        </div>
        <div class="flex justify-between items-center">
            <button onclick="prevQuestion()" class="btn-secondary px-4 py-2 rounded-xl text-sm" ${currentQuestion === 0 ? 'disabled style="opacity: 0.5; cursor: not-allowed;"' : ''}>
                Previous
            </button>
            <div class="flex gap-1">
                ${quizQuestions.map((_, i) => `
                    <div class="w-2 h-2 rounded-full ${i === currentQuestion ? 'bg-accent' : answeredQuestions[i] !== null ? (answeredQuestions[i] === quizQuestions[i].correct ? 'bg-green-500' : 'bg-red-500') : 'bg-gray-600'}"></div>
                `).join('')}
            </div>
            <button onclick="nextQuestion()" class="btn-primary px-4 py-2 rounded-xl text-sm" ${answeredQuestions[currentQuestion] === null ? 'disabled style="opacity: 0.5; cursor: not-allowed;"' : ''}>
                ${currentQuestion === quizQuestions.length - 1 ? 'Finish' : 'Next'}
            </button>
        </div>
    `;
}

function selectAnswer(index) {
    if (answeredQuestions[currentQuestion] !== null) return;

    answeredQuestions[currentQuestion] = index;
    if (index === quizQuestions[currentQuestion].correct) {
        score++;
    }
    saveState();
    updateProgress();
    renderQuiz();
}

function nextQuestion() {
    if (answeredQuestions[currentQuestion] === null) return;
    if (currentQuestion < quizQuestions.length - 1) {
        currentQuestion++;
        saveState();
        renderQuiz();
    } else {
        renderQuiz();
    }
}

function prevQuestion() {
    if (currentQuestion > 0) {
        currentQuestion--;
        saveState();
        renderQuiz();
    }
}

function retakeQuiz() {
    currentQuestion = 0;
    score = 0;
    answeredQuestions = new Array(quizQuestions.length).fill(null);
    saveState();
    updateProgress();
    renderQuiz();
}

// === Initialize on page load ===

document.addEventListener('DOMContentLoaded', () => {
    loadState();

    // Open the first incomplete lesson
    const start = LEARN_CONFIG.lessonStart;
    const end = start + LEARN_CONFIG.totalLessons;
    for (let i = start; i < end; i++) {
        if (!completedLessons.has(i)) {
            toggleLesson(i);
            break;
        }
    }

    // Check challenge status
    checkChallengeStatus();
});
