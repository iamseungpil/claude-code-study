export interface Env {
  DB: D1Database;
  JWT_SECRET: string;
  ADMIN_USER_IDS: string;
}

export interface User {
  user_id: string;
  full_name: string;
  first_name: string;
  last_name: string;
  password_hash: string | null;
  profile_image: string | null;
  role: string;
  registered_at: string;
}

export interface Challenge {
  week: number;
  status: string;
  start_time: string | null;
  end_time: string | null;
  started_by: string | null;
}

export interface PersonalTimer {
  user_id: string;
  week: number;
  started_at: string;
  status: string;
}

export interface Submission {
  id: number;
  user_id: string;
  week: number;
  submission_number: number;
  github_url: string;
  submitted_at: string;
  elapsed_seconds: number;
  elapsed_minutes: number;
  personal_start_time: string | null;
  global_start_time: string | null;
  created_at: string;
}

export interface Evaluation {
  id: number;
  user_id: string;
  week: number;
  submission_number: number;
  rubric_score: number;
  time_rank: number;
  time_rank_bonus: number;
  total_score: number;
  feedback: string;
  strengths: string | null;
  improvements: string | null;
  status: string;
  evaluated_at: string | null;
  evaluated_by: string | null;
}

export interface JwtPayload {
  user_id: string;
  full_name: string;
  first_name: string;
  last_name: string;
  role: string;
}
