// Password hashing using Web Crypto API (PBKDF2)
// bcrypt is not available in Workers, so we use PBKDF2 instead

const ITERATIONS = 100000;
const KEY_LENGTH = 256;

export async function hashPassword(password: string): Promise<string> {
  const salt = crypto.getRandomValues(new Uint8Array(16));
  const encoder = new TextEncoder();

  const keyMaterial = await crypto.subtle.importKey(
    'raw',
    encoder.encode(password),
    'PBKDF2',
    false,
    ['deriveBits']
  );

  const hash = await crypto.subtle.deriveBits(
    {
      name: 'PBKDF2',
      hash: 'SHA-256',
      salt: salt,
      iterations: ITERATIONS,
    },
    keyMaterial,
    KEY_LENGTH
  );

  const saltB64 = btoa(String.fromCharCode(...salt));
  const hashB64 = btoa(String.fromCharCode(...new Uint8Array(hash)));

  return `pbkdf2$${saltB64}$${hashB64}`;
}

export async function verifyPassword(password: string, storedHash: string): Promise<boolean> {
  try {
    // Support both old bcrypt hashes and new PBKDF2 hashes
    if (storedHash.startsWith('$2b$') || storedHash.startsWith('$2a$')) {
      // Legacy bcrypt hash - can't verify in Workers
      // User needs to reset password
      return false;
    }

    if (!storedHash.startsWith('pbkdf2$')) {
      return false;
    }

    const parts = storedHash.split('$');
    if (parts.length !== 3) return false;

    const [, saltB64, expectedHashB64] = parts;
    const salt = new Uint8Array([...atob(saltB64)].map(c => c.charCodeAt(0)));

    const encoder = new TextEncoder();
    const keyMaterial = await crypto.subtle.importKey(
      'raw',
      encoder.encode(password),
      'PBKDF2',
      false,
      ['deriveBits']
    );

    const hash = await crypto.subtle.deriveBits(
      {
        name: 'PBKDF2',
        hash: 'SHA-256',
        salt: salt,
        iterations: ITERATIONS,
      },
      keyMaterial,
      KEY_LENGTH
    );

    const hashB64 = btoa(String.fromCharCode(...new Uint8Array(hash)));

    return hashB64 === expectedHashB64;
  } catch {
    return false;
  }
}
