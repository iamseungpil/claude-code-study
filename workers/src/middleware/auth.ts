import type { Context, Next } from 'hono';
import type { Env, JwtPayload, AppVariables } from '../types';
import { timingSafeEqual } from 'hono/utils/buffer';
import { verifyToken } from '../lib/jwt';

/** Extract and verify JWT from Authorization header. Sets c.set('user', payload). */
export async function authMiddleware(c: Context<{ Bindings: Env; Variables: AppVariables }>, next: Next) {
  const authHeader = c.req.header('Authorization');
  if (!authHeader?.startsWith('Bearer ')) {
    c.set('user', null);
    return next();
  }
  const token = authHeader.slice(7);
  const payload = await verifyToken(token, c.env.JWT_SECRET);
  c.set('user', payload);
  return next();
}

/** Require a valid JWT — 401 if missing/invalid. */
export function requireAuth(c: Context<{ Bindings: Env; Variables: AppVariables }>): JwtPayload {
  const user = c.get('user');
  if (!user) {
    throw new Error('NOT_AUTHENTICATED');
  }
  return user;
}

/** Require admin role — 403 if not admin. Supports X-Admin-Key header for CLI scripts. */
export async function requireAdmin(c: Context<{ Bindings: Env; Variables: AppVariables }>): Promise<JwtPayload> {
  // API key auth (CLI scripts)
  const apiKey = c.req.header('X-Admin-Key');
  const expectedKey = c.env.ADMIN_API_KEY;
  if (apiKey && expectedKey && await timingSafeEqual(apiKey, expectedKey)) {
    return {
      user_id: 'cli-admin',
      full_name: 'CLI Admin',
      first_name: 'CLI',
      last_name: 'Admin',
      role: 'admin',
    };
  }

  // JWT auth
  const user = requireAuth(c);
  if (user.role !== 'admin') {
    throw new Error('ADMIN_REQUIRED');
  }
  return user;
}
