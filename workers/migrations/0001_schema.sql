-- Claude Code Study - D1 Schema

CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    full_name TEXT NOT NULL,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    password_hash TEXT,
    profile_image TEXT,
    role TEXT NOT NULL DEFAULT 'participant',
    registered_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS challenges (
    week INTEGER PRIMARY KEY CHECK (week BETWEEN 1 AND 5),
    status TEXT NOT NULL DEFAULT 'not_started',
    start_time TEXT,
    end_time TEXT,
    started_by TEXT
);

CREATE TABLE IF NOT EXISTS personal_timers (
    user_id TEXT NOT NULL,
    week INTEGER NOT NULL,
    started_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'in_progress',
    PRIMARY KEY (user_id, week)
);

CREATE TABLE IF NOT EXISTS submissions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    week INTEGER NOT NULL,
    submission_number INTEGER NOT NULL,
    github_url TEXT NOT NULL,
    submitted_at TEXT NOT NULL,
    elapsed_seconds REAL NOT NULL,
    elapsed_minutes REAL NOT NULL,
    personal_start_time TEXT,
    global_start_time TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (user_id, week, submission_number)
);

CREATE TABLE IF NOT EXISTS evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    week INTEGER NOT NULL,
    submission_number INTEGER NOT NULL,
    rubric_score INTEGER NOT NULL,
    time_rank INTEGER NOT NULL,
    time_rank_bonus INTEGER NOT NULL,
    total_score INTEGER NOT NULL,
    feedback TEXT DEFAULT '',
    strengths TEXT,
    improvements TEXT,
    status TEXT NOT NULL DEFAULT 'pending_review',
    evaluated_at TEXT,
    evaluated_by TEXT,
    UNIQUE (user_id, week, submission_number)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_submissions_user_week ON submissions(user_id, week);
CREATE INDEX IF NOT EXISTS idx_submissions_elapsed ON submissions(week, elapsed_minutes);
CREATE INDEX IF NOT EXISTS idx_evaluations_week ON evaluations(week, total_score DESC);

-- Seed challenge rows
INSERT OR IGNORE INTO challenges (week) VALUES (1),(2),(3),(4),(5);
