-- Users table
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT UNIQUE NOT NULL,
    full_name TEXT NOT NULL,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    profile_image TEXT,
    role TEXT DEFAULT 'participant',
    registered_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_users_user_id ON users(user_id);

-- Challenges table
CREATE TABLE IF NOT EXISTS challenges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    week INTEGER UNIQUE NOT NULL,
    status TEXT DEFAULT 'not_started',
    start_time TEXT,
    end_time TEXT,
    started_by TEXT
);

-- Personal challenges
CREATE TABLE IF NOT EXISTS personal_challenges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    week INTEGER NOT NULL,
    started_at TEXT NOT NULL,
    status TEXT DEFAULT 'in_progress',
    UNIQUE(user_id, week)
);

CREATE INDEX IF NOT EXISTS idx_personal_challenges_user_week ON personal_challenges(user_id, week);

-- Submissions
CREATE TABLE IF NOT EXISTS submissions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    week INTEGER NOT NULL,
    github_url TEXT NOT NULL,
    submitted_at TEXT NOT NULL,
    elapsed_seconds REAL,
    status TEXT DEFAULT 'submitted',
    UNIQUE(user_id, week)
);

CREATE INDEX IF NOT EXISTS idx_submissions_week ON submissions(week);
CREATE INDEX IF NOT EXISTS idx_submissions_user ON submissions(user_id);

-- Evaluations
CREATE TABLE IF NOT EXISTS evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    week INTEGER NOT NULL,
    status TEXT DEFAULT 'pending',
    rubric_score INTEGER,
    time_rank INTEGER,
    time_rank_bonus INTEGER,
    total_score INTEGER,
    build_npm_install TEXT,
    build_npm_build TEXT,
    breakdown_json TEXT,
    feedback TEXT,
    strengths_json TEXT,
    improvements_json TEXT,
    evaluated_at TEXT,
    UNIQUE(user_id, week)
);

CREATE INDEX IF NOT EXISTS idx_evaluations_week ON evaluations(week);

-- Initial challenge data
INSERT OR IGNORE INTO challenges (week, status) VALUES (1, 'not_started');
INSERT OR IGNORE INTO challenges (week, status) VALUES (2, 'not_started');
INSERT OR IGNORE INTO challenges (week, status) VALUES (3, 'not_started');
INSERT OR IGNORE INTO challenges (week, status) VALUES (4, 'not_started');
INSERT OR IGNORE INTO challenges (week, status) VALUES (5, 'not_started');
