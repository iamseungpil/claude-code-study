import { Hono } from 'hono';
import { cors } from 'hono/cors';
import type { Env, AppVariables } from './types';
import { authMiddleware } from './middleware/auth';

import health from './routes/health';
import auth from './routes/auth';
import challenges from './routes/challenges';
import timer from './routes/timer';
import submissions from './routes/submissions';
import evaluations from './routes/evaluations';
import admin from './routes/admin';
import leaderboard from './routes/leaderboard';

const app = new Hono<{ Bindings: Env; Variables: AppVariables }>();

// ---------- CORS ----------
app.use(
  '*',
  cors({
    origin: [
      'http://localhost:8003',
      'http://127.0.0.1:8003',
      'http://localhost:8787',
      'http://127.0.0.1:8787',
      'http://localhost:5500',
      'http://127.0.0.1:5500',
      'https://claude-code-study.pages.dev',
    ],
    allowMethods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
    allowHeaders: ['Content-Type', 'Authorization', 'X-Admin-Key'],
    credentials: true,
  }),
);

// ---------- Auth middleware (extracts JWT if present) ----------
app.use('/api/*', authMiddleware);

// ---------- Error handler ----------
app.onError((err, c) => {
  if (err.message === 'NOT_AUTHENTICATED') {
    return c.json({ detail: 'Not authenticated' }, 401);
  }
  if (err.message === 'ADMIN_REQUIRED') {
    return c.json({ detail: 'Admin access required' }, 403);
  }
  console.error('Unhandled error:', err);
  return c.json({ detail: 'Internal server error' }, 500);
});

// ---------- Mount routes ----------
app.route('/', health);
app.route('/', auth);
app.route('/', challenges);
app.route('/', timer);
app.route('/', submissions);
app.route('/', evaluations);
app.route('/', admin);
app.route('/', leaderboard);

export default app;
