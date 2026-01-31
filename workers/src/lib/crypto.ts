/**
 * Password hashing: PBKDF2 for new users, bcrypt-edge for verifying existing bcrypt hashes.
 */
import { compareSync } from 'bcrypt-edge';

const PBKDF2_ITERATIONS = 100_000;
const SALT_LENGTH = 16;
const KEY_LENGTH = 32;

function toHex(buf: ArrayBuffer | Uint8Array): string {
  const bytes = buf instanceof Uint8Array ? buf : new Uint8Array(buf);
  return [...bytes].map((b) => b.toString(16).padStart(2, '0')).join('');
}

function fromHex(hex: string): Uint8Array {
  const bytes = new Uint8Array(hex.length / 2);
  for (let i = 0; i < hex.length; i += 2) {
    bytes[i / 2] = parseInt(hex.substring(i, i + 2), 16);
  }
  return bytes;
}

/** Hash a password using PBKDF2 (Web Crypto). Returns "pbkdf2$<salt>$<hash>". */
export async function hashPassword(password: string): Promise<string> {
  const salt = crypto.getRandomValues(new Uint8Array(SALT_LENGTH));
  const key = await crypto.subtle.importKey(
    'raw',
    new TextEncoder().encode(password),
    'PBKDF2',
    false,
    ['deriveBits'],
  );
  const derived = await crypto.subtle.deriveBits(
    { name: 'PBKDF2', salt: salt.buffer as ArrayBuffer, iterations: PBKDF2_ITERATIONS, hash: 'SHA-256' },
    key,
    KEY_LENGTH * 8,
  );
  return `pbkdf2$${toHex(salt)}$${toHex(derived)}`;
}

/** Verify password against either a bcrypt hash ($2b$...) or a PBKDF2 hash (pbkdf2$...). */
export async function verifyPassword(password: string, stored: string): Promise<boolean> {
  if (stored.startsWith('$2b$') || stored.startsWith('$2a$')) {
    // Existing bcrypt hash — use bcrypt-edge WASM
    return compareSync(password, stored);
  }

  if (stored.startsWith('pbkdf2$')) {
    const [, saltHex, hashHex] = stored.split('$');
    const salt = fromHex(saltHex);
    const key = await crypto.subtle.importKey(
      'raw',
      new TextEncoder().encode(password),
      'PBKDF2',
      false,
      ['deriveBits'],
    );
    const derived = await crypto.subtle.deriveBits(
      { name: 'PBKDF2', salt: salt.buffer as ArrayBuffer, iterations: PBKDF2_ITERATIONS, hash: 'SHA-256' },
      key,
      KEY_LENGTH * 8,
    );
    return toHex(derived) === hashHex;
  }

  return false;
}
