import type { Context, Next } from 'hono';
import type { Env, JwtPayload } from '../types';
import { verifyToken } from '../lib/jwt';

/** Extract and verify JWT from Authorization header. Sets c.set('user', payload). */
export async function authMiddleware(c: Context<{ Bindings: Env; Variables: { user: JwtPayload } }>, next: Next) {
  const authHeader = c.req.header('Authorization');
  if (!authHeader?.startsWith('Bearer ')) {
    c.set('user', null as unknown as JwtPayload);
    return next();
  }
  const token = authHeader.slice(7);
  const payload = await verifyToken(token, c.env.JWT_SECRET);
  c.set('user', payload as unknown as JwtPayload);
  return next();
}

/** Require a valid JWT — 401 if missing/invalid. */
export function requireAuth(c: Context<{ Bindings: Env; Variables: { user: JwtPayload } }>): JwtPayload {
  const user = c.get('user');
  if (!user) {
    throw new Error('NOT_AUTHENTICATED');
  }
  return user;
}

/** Require admin role — 403 if not admin. */
export function requireAdmin(c: Context<{ Bindings: Env; Variables: { user: JwtPayload } }>): JwtPayload {
  const user = requireAuth(c);
  if (user.role !== 'admin') {
    throw new Error('ADMIN_REQUIRED');
  }
  return user;
}
