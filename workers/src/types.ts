// Cloudflare Workers Environment Bindings
export interface Env {
  DB: D1Database;
  SESSIONS: KVNamespace;
  JWT_SECRET: string;
  ANTHROPIC_API_KEY: string;
  ENVIRONMENT: string;
}

// User types
export interface User {
  id: number;
  user_id: string;
  full_name: string;
  first_name: string;
  last_name: string;
  password_hash: string;
  profile_image: string | null;
  role: 'admin' | 'participant';
  registered_at: string;
}

export interface UserPayload {
  user_id: string;
  full_name: string;
  first_name: string;
  last_name: string;
  role: 'admin' | 'participant';
  exp: number;
}

// Challenge types
export interface Challenge {
  id: number;
  week: number;
  status: 'not_started' | 'started' | 'ended';
  start_time: string | null;
  end_time: string | null;
  started_by: string | null;
}

export interface PersonalChallenge {
  id: number;
  user_id: string;
  week: number;
  started_at: string;
  status: 'in_progress' | 'submitted';
}

// Submission types
export interface Submission {
  id: number;
  user_id: string;
  week: number;
  github_url: string;
  submitted_at: string;
  elapsed_seconds: number | null;
  status: 'submitted' | 'evaluating' | 'evaluated' | 'error';
}

// Evaluation types
export interface Evaluation {
  id: number;
  user_id: string;
  week: number;
  status: 'pending' | 'completed' | 'error';
  rubric_score: number | null;
  time_rank: number | null;
  time_rank_bonus: number | null;
  total_score: number | null;
  build_npm_install: string | null;
  build_npm_build: string | null;
  breakdown_json: string | null;
  feedback: string | null;
  strengths_json: string | null;
  improvements_json: string | null;
  evaluated_at: string | null;
}

// API Request/Response types
export interface RegisterRequest {
  user_id: string;
  password: string;
  first_name: string;
  last_name: string;
}

export interface LoginRequest {
  user_id: string;
  password: string;
}

export interface SubmitRequest {
  week: number;
  github_url: string;
}
