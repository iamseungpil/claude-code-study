import { Hono } from 'hono';
import type { Env, JwtPayload, User } from '../types';
import { hashPassword, verifyPassword } from '../lib/crypto';
import { signToken } from '../lib/jwt';
import { requireAuth } from '../middleware/auth';

const PARTICIPANT_ID_RE = /^[a-zA-Z0-9_-]{3,30}$/;
const NAME_RE = /^[a-zA-Z\uAC00-\uD7A3\s]{1,50}$/;

const app = new Hono<{ Bindings: Env; Variables: { user: JwtPayload } }>();

// ---------- Register ----------
app.post('/api/auth/register', async (c) => {
  const body = await c.req.json<{
    user_id: string;
    first_name: string;
    last_name: string;
    password: string;
  }>();

  // Validation
  if (!PARTICIPANT_ID_RE.test(body.user_id)) {
    return c.json({ detail: 'User ID must be 3-30 characters (letters, numbers, _, -)' }, 400);
  }
  if (!NAME_RE.test(body.first_name) || !NAME_RE.test(body.last_name)) {
    return c.json({ detail: 'Name must contain only letters and spaces (1-50 characters)' }, 400);
  }
  if (!body.password || body.password.length < 4) {
    return c.json({ detail: 'Password must be at least 4 characters' }, 400);
  }

  const fullName = `${body.first_name.trim()} ${body.last_name.trim()}`;

  // Check duplicate user_id (case-insensitive)
  const existing = await c.env.DB.prepare(
    'SELECT user_id FROM users WHERE LOWER(user_id) = LOWER(?)',
  )
    .bind(body.user_id)
    .first();
  if (existing) {
    return c.json({ detail: 'User ID already exists' }, 400);
  }

  // Check duplicate full_name
  const existingName = await c.env.DB.prepare(
    'SELECT user_id FROM users WHERE LOWER(full_name) = LOWER(?)',
  )
    .bind(fullName)
    .first();
  if (existingName) {
    return c.json({ detail: 'User already exists' }, 400);
  }

  // Profile image lookup
  const profileImage = await lookupProfileImage(body.first_name, body.last_name);

  // Determine role
  const adminIds = new Set(
    (c.env.ADMIN_USER_IDS || 'iamseungpil').split(',').map((s) => s.trim().toLowerCase()),
  );
  const role = adminIds.has(body.user_id.toLowerCase()) ? 'admin' : 'participant';

  const passwordHash = await hashPassword(body.password);
  const registeredAt = new Date().toISOString();

  await c.env.DB.prepare(
    `INSERT INTO users (user_id, full_name, first_name, last_name, password_hash, profile_image, role, registered_at)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
  )
    .bind(body.user_id, fullName, body.first_name.trim(), body.last_name.trim(), passwordHash, profileImage, role, registeredAt)
    .run();

  const token = await signToken(
    { user_id: body.user_id, full_name: fullName, first_name: body.first_name.trim(), last_name: body.last_name.trim(), role },
    c.env.JWT_SECRET,
  );

  return c.json({
    status: 'registered',
    user_id: body.user_id,
    full_name: fullName,
    profile_image: profileImage,
    token,
  });
});

// ---------- Login ----------
app.post('/api/auth/login', async (c) => {
  const body = await c.req.json<{ user_id: string; password: string }>();

  const user = await c.env.DB.prepare('SELECT * FROM users WHERE LOWER(user_id) = LOWER(?)')
    .bind(body.user_id)
    .first<User>();

  if (!user || !user.password_hash) {
    return c.json({ detail: 'Invalid credentials' }, 401);
  }

  const valid = await verifyPassword(body.password, user.password_hash);
  if (!valid) {
    return c.json({ detail: 'Invalid credentials' }, 401);
  }

  const token = await signToken(
    {
      user_id: user.user_id,
      full_name: user.full_name,
      first_name: user.first_name,
      last_name: user.last_name,
      role: user.role,
    },
    c.env.JWT_SECRET,
  );

  return c.json({
    status: 'logged_in',
    user_id: user.user_id,
    full_name: user.full_name,
    first_name: user.first_name,
    last_name: user.last_name,
    profile_image: user.profile_image,
    role: user.role,
    token,
  });
});

// ---------- Me ----------
app.get('/api/auth/me', async (c) => {
  const jwtUser = requireAuth(c);

  const user = await c.env.DB.prepare('SELECT * FROM users WHERE LOWER(user_id) = LOWER(?)')
    .bind(jwtUser.user_id)
    .first<User>();

  if (!user) {
    return c.json({ detail: 'Not authenticated' }, 401);
  }

  return c.json({
    user_id: user.user_id,
    full_name: user.full_name,
    first_name: user.first_name,
    last_name: user.last_name,
    profile_image: user.profile_image,
    role: user.role,
    registered_at: user.registered_at,
  });
});

// ---------- Helper ----------
async function lookupProfileImage(firstName: string, lastName: string): Promise<string | null> {
  const fullCombined = `${firstName}${lastName}`.toLowerCase().replace(/\s/g, '');
  const firstOnly = firstName.toLowerCase().replace(/\s/g, '');
  const variants = [...new Set([fullCombined, firstOnly])];

  for (const name of variants) {
    for (const ext of ['png', 'jpg', 'jpeg']) {
      const url = `https://sundong.kim/assets/img/members/${name}.${ext}`;
      try {
        const res = await fetch(url, { method: 'HEAD' });
        if (res.ok) return url;
      } catch {
        continue;
      }
    }
  }
  return null;
}

export default app;
