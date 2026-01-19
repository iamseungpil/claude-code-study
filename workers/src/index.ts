import { Hono } from 'hono';
import { cors } from 'hono/cors';
import type { Env, UserPayload, RegisterRequest, LoginRequest, SubmitRequest } from './types';
import { createToken, verifyToken, extractToken } from './services/jwt';
import { hashPassword, verifyPassword } from './services/password';

const app = new Hono<{ Bindings: Env }>();

// CORS middleware
app.use('*', cors({
  origin: ['http://localhost:8003', 'https://claude-code-study.pages.dev'],
  credentials: true,
}));

// Helper: Get current user from JWT
async function getCurrentUser(c: any): Promise<UserPayload | null> {
  const authHeader = c.req.header('Authorization');
  const token = extractToken(authHeader);
  if (!token) return null;
  return verifyToken(token, c.env.JWT_SECRET);
}

// Helper: Require authentication
async function requireAuth(c: any): Promise<UserPayload> {
  const user = await getCurrentUser(c);
  if (!user) {
    throw new Error('Unauthorized');
  }
  return user;
}

// Health check
app.get('/api/health', (c) => {
  return c.json({ status: 'ok', timestamp: new Date().toISOString(), version: '1.0.0' });
});

// ============================================================
// AUTH ROUTES
// ============================================================

app.post('/api/auth/register', async (c) => {
  try {
    const body = await c.req.json<RegisterRequest>();
    const { user_id, password, first_name, last_name } = body;

    if (!user_id || !password || !first_name || !last_name) {
      return c.json({ error: 'Missing required fields' }, 400);
    }

    // Check if user exists
    const existing = await c.env.DB.prepare(
      'SELECT user_id FROM users WHERE user_id = ?'
    ).bind(user_id).first();

    if (existing) {
      return c.json({ error: 'User already exists' }, 400);
    }

    const password_hash = await hashPassword(password);
    const full_name = `${first_name} ${last_name}`;
    const role = user_id === 'iamseungpil' ? 'admin' : 'participant';
    const registered_at = new Date().toISOString();

    await c.env.DB.prepare(`
      INSERT INTO users (user_id, full_name, first_name, last_name, password_hash, role, registered_at)
      VALUES (?, ?, ?, ?, ?, ?, ?)
    `).bind(user_id, full_name, first_name, last_name, password_hash, role, registered_at).run();

    return c.json({ message: 'User registered successfully', user_id });
  } catch (error: any) {
    return c.json({ error: error.message }, 500);
  }
});

app.post('/api/auth/login', async (c) => {
  try {
    const body = await c.req.json<LoginRequest>();
    const { user_id, password } = body;

    const user = await c.env.DB.prepare(
      'SELECT * FROM users WHERE user_id = ?'
    ).bind(user_id).first();

    if (!user) {
      return c.json({ error: 'Invalid credentials' }, 401);
    }

    const isValid = await verifyPassword(password, user.password_hash as string);
    if (!isValid) {
      return c.json({ error: 'Invalid credentials' }, 401);
    }

    const payload: UserPayload = {
      user_id: user.user_id as string,
      full_name: user.full_name as string,
      first_name: user.first_name as string,
      last_name: user.last_name as string,
      role: user.role as 'admin' | 'participant',
      exp: Math.floor(Date.now() / 1000) + 24 * 60 * 60, // 24 hours
    };

    const token = await createToken(payload, c.env.JWT_SECRET);

    return c.json({
      access_token: token,
      token_type: 'bearer',
      user: {
        user_id: user.user_id,
        full_name: user.full_name,
        role: user.role,
        profile_image: user.profile_image,
      },
    });
  } catch (error: any) {
    return c.json({ error: error.message }, 500);
  }
});

app.get('/api/auth/me', async (c) => {
  try {
    const currentUser = await getCurrentUser(c);
    if (!currentUser) {
      return c.json({ error: 'Not authenticated' }, 401);
    }

    const user = await c.env.DB.prepare(
      'SELECT user_id, full_name, first_name, last_name, role, profile_image, registered_at FROM users WHERE user_id = ?'
    ).bind(currentUser.user_id).first();

    if (!user) {
      return c.json({ error: 'User not found' }, 404);
    }

    return c.json(user);
  } catch (error: any) {
    return c.json({ error: error.message }, 500);
  }
});

// ============================================================
// CHALLENGE ROUTES
// ============================================================

app.get('/api/challenges/status', async (c) => {
  try {
    const challenges = await c.env.DB.prepare(
      'SELECT week, status, start_time, end_time FROM challenges ORDER BY week'
    ).all();

    return c.json(challenges.results);
  } catch (error: any) {
    return c.json({ error: error.message }, 500);
  }
});

app.get('/api/challenge/:week/status', async (c) => {
  try {
    const week = parseInt(c.req.param('week'));
    const challenge = await c.env.DB.prepare(
      'SELECT * FROM challenges WHERE week = ?'
    ).bind(week).first();

    if (!challenge) {
      return c.json({ error: 'Challenge not found' }, 404);
    }

    return c.json(challenge);
  } catch (error: any) {
    return c.json({ error: error.message }, 500);
  }
});

app.post('/api/challenge/:week/start-personal', async (c) => {
  try {
    const user = await requireAuth(c);
    const week = parseInt(c.req.param('week'));

    // Check if already started
    const existing = await c.env.DB.prepare(
      'SELECT * FROM personal_challenges WHERE user_id = ? AND week = ?'
    ).bind(user.user_id, week).first();

    if (existing) {
      return c.json({
        message: 'Already started',
        started_at: existing.started_at,
        status: existing.status,
      });
    }

    const started_at = new Date().toISOString();

    await c.env.DB.prepare(`
      INSERT INTO personal_challenges (user_id, week, started_at, status)
      VALUES (?, ?, ?, 'in_progress')
    `).bind(user.user_id, week, started_at).run();

    return c.json({ message: 'Challenge started', started_at, status: 'in_progress' });
  } catch (error: any) {
    if (error.message === 'Unauthorized') {
      return c.json({ error: 'Unauthorized' }, 401);
    }
    return c.json({ error: error.message }, 500);
  }
});

app.get('/api/challenge/:week/my-status', async (c) => {
  try {
    const user = await requireAuth(c);
    const week = parseInt(c.req.param('week'));

    const personal = await c.env.DB.prepare(
      'SELECT * FROM personal_challenges WHERE user_id = ? AND week = ?'
    ).bind(user.user_id, week).first();

    if (!personal) {
      return c.json({ status: 'not_started' });
    }

    const started_at = new Date(personal.started_at as string);
    const elapsed_seconds = (Date.now() - started_at.getTime()) / 1000;

    return c.json({
      status: personal.status,
      started_at: personal.started_at,
      elapsed_seconds,
      elapsed_minutes: elapsed_seconds / 60,
    });
  } catch (error: any) {
    if (error.message === 'Unauthorized') {
      return c.json({ error: 'Unauthorized' }, 401);
    }
    return c.json({ error: error.message }, 500);
  }
});

// ============================================================
// SUBMISSION ROUTES
// ============================================================

app.post('/api/submissions/submit', async (c) => {
  try {
    const user = await requireAuth(c);
    const body = await c.req.json<SubmitRequest>();
    const { week, github_url } = body;

    // Get personal challenge start time
    const personal = await c.env.DB.prepare(
      'SELECT * FROM personal_challenges WHERE user_id = ? AND week = ?'
    ).bind(user.user_id, week).first();

    if (!personal) {
      return c.json({ error: 'Challenge not started' }, 400);
    }

    const started_at = new Date(personal.started_at as string);
    const submitted_at = new Date();
    const elapsed_seconds = (submitted_at.getTime() - started_at.getTime()) / 1000;

    // Check if already submitted
    const existing = await c.env.DB.prepare(
      'SELECT * FROM submissions WHERE user_id = ? AND week = ?'
    ).bind(user.user_id, week).first();

    if (existing) {
      return c.json({ error: 'Already submitted' }, 400);
    }

    await c.env.DB.prepare(`
      INSERT INTO submissions (user_id, week, github_url, submitted_at, elapsed_seconds, status)
      VALUES (?, ?, ?, ?, ?, 'submitted')
    `).bind(user.user_id, week, github_url, submitted_at.toISOString(), elapsed_seconds).run();

    // Update personal challenge status
    await c.env.DB.prepare(
      'UPDATE personal_challenges SET status = ? WHERE user_id = ? AND week = ?'
    ).bind('submitted', user.user_id, week).run();

    // Create pending evaluation
    await c.env.DB.prepare(`
      INSERT INTO evaluations (user_id, week, status)
      VALUES (?, ?, 'pending')
    `).bind(user.user_id, week).run();

    return c.json({
      message: 'Submission received',
      elapsed_seconds,
      elapsed_minutes: elapsed_seconds / 60,
    });
  } catch (error: any) {
    if (error.message === 'Unauthorized') {
      return c.json({ error: 'Unauthorized' }, 401);
    }
    return c.json({ error: error.message }, 500);
  }
});

app.get('/api/submissions/:week', async (c) => {
  try {
    const week = parseInt(c.req.param('week'));

    const submissions = await c.env.DB.prepare(`
      SELECT s.*, u.full_name
      FROM submissions s
      JOIN users u ON s.user_id = u.user_id
      WHERE s.week = ?
      ORDER BY s.submitted_at
    `).bind(week).all();

    return c.json(submissions.results);
  } catch (error: any) {
    return c.json({ error: error.message }, 500);
  }
});

// ============================================================
// EVALUATION ROUTES
// ============================================================

app.get('/api/evaluations/:week/:participant_id', async (c) => {
  try {
    const week = parseInt(c.req.param('week'));
    const participant_id = c.req.param('participant_id');

    const evaluation = await c.env.DB.prepare(
      'SELECT * FROM evaluations WHERE user_id = ? AND week = ?'
    ).bind(participant_id, week).first();

    if (!evaluation) {
      return c.json({ error: 'Evaluation not found' }, 404);
    }

    // Parse JSON fields
    const result = {
      ...evaluation,
      breakdown: evaluation.breakdown_json ? JSON.parse(evaluation.breakdown_json as string) : null,
      strengths: evaluation.strengths_json ? JSON.parse(evaluation.strengths_json as string) : null,
      improvements: evaluation.improvements_json ? JSON.parse(evaluation.improvements_json as string) : null,
    };

    return c.json(result);
  } catch (error: any) {
    return c.json({ error: error.message }, 500);
  }
});

// ============================================================
// LEADERBOARD ROUTES
// ============================================================

app.get('/api/leaderboard/season', async (c) => {
  try {
    const results = await c.env.DB.prepare(`
      SELECT
        u.user_id,
        u.full_name,
        u.profile_image,
        SUM(COALESCE(e.total_score, 0)) as total_score,
        COUNT(CASE WHEN e.status = 'completed' THEN 1 END) as completed_weeks
      FROM users u
      LEFT JOIN evaluations e ON u.user_id = e.user_id AND e.status = 'completed'
      GROUP BY u.user_id, u.full_name, u.profile_image
      ORDER BY total_score DESC
    `).all();

    return c.json(results.results);
  } catch (error: any) {
    return c.json({ error: error.message }, 500);
  }
});

app.get('/api/leaderboard/:week', async (c) => {
  try {
    const week = parseInt(c.req.param('week'));

    const results = await c.env.DB.prepare(`
      SELECT
        u.user_id,
        u.full_name,
        u.profile_image,
        e.rubric_score,
        e.time_rank,
        e.time_rank_bonus,
        e.total_score,
        e.status as evaluation_status,
        s.elapsed_minutes
      FROM users u
      JOIN submissions s ON u.user_id = s.user_id AND s.week = ?
      LEFT JOIN evaluations e ON u.user_id = e.user_id AND e.week = ?
      ORDER BY e.total_score DESC NULLS LAST, s.elapsed_seconds ASC
    `).bind(week, week).all();

    return c.json(results.results);
  } catch (error: any) {
    return c.json({ error: error.message }, 500);
  }
});

// ============================================================
// ADMIN ROUTES
// ============================================================

app.get('/api/admin/users', async (c) => {
  try {
    const user = await requireAuth(c);
    if (user.role !== 'admin') {
      return c.json({ error: 'Forbidden' }, 403);
    }

    const users = await c.env.DB.prepare(
      'SELECT user_id, full_name, first_name, last_name, role, profile_image, registered_at FROM users'
    ).all();

    return c.json(users.results);
  } catch (error: any) {
    if (error.message === 'Unauthorized') {
      return c.json({ error: 'Unauthorized' }, 401);
    }
    return c.json({ error: error.message }, 500);
  }
});

app.post('/api/admin/challenge/:week/start', async (c) => {
  try {
    const user = await requireAuth(c);
    if (user.role !== 'admin') {
      return c.json({ error: 'Forbidden' }, 403);
    }

    const week = parseInt(c.req.param('week'));
    const start_time = new Date().toISOString();

    await c.env.DB.prepare(
      'UPDATE challenges SET status = ?, start_time = ?, started_by = ? WHERE week = ?'
    ).bind('started', start_time, user.user_id, week).run();

    return c.json({ message: 'Challenge started', week, start_time });
  } catch (error: any) {
    if (error.message === 'Unauthorized') {
      return c.json({ error: 'Unauthorized' }, 401);
    }
    return c.json({ error: error.message }, 500);
  }
});

app.post('/api/admin/challenge/:week/end', async (c) => {
  try {
    const user = await requireAuth(c);
    if (user.role !== 'admin') {
      return c.json({ error: 'Forbidden' }, 403);
    }

    const week = parseInt(c.req.param('week'));
    const end_time = new Date().toISOString();

    await c.env.DB.prepare(
      'UPDATE challenges SET status = ?, end_time = ? WHERE week = ?'
    ).bind('ended', end_time, week).run();

    return c.json({ message: 'Challenge ended', week, end_time });
  } catch (error: any) {
    if (error.message === 'Unauthorized') {
      return c.json({ error: 'Unauthorized' }, 401);
    }
    return c.json({ error: error.message }, 500);
  }
});

export default app;
