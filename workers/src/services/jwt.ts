import type { UserPayload } from '../types';

// Simple JWT implementation using Web Crypto API
// Note: This is a basic implementation. Consider using a library for production.

async function base64UrlEncode(data: Uint8Array): Promise<string> {
  const base64 = btoa(String.fromCharCode(...data));
  return base64.replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

async function base64UrlDecode(str: string): Promise<Uint8Array> {
  const base64 = str.replace(/-/g, '+').replace(/_/g, '/');
  const padded = base64 + '='.repeat((4 - base64.length % 4) % 4);
  const binary = atob(padded);
  return new Uint8Array([...binary].map(c => c.charCodeAt(0)));
}

async function sign(payload: string, secret: string): Promise<string> {
  const encoder = new TextEncoder();
  const key = await crypto.subtle.importKey(
    'raw',
    encoder.encode(secret),
    { name: 'HMAC', hash: 'SHA-256' },
    false,
    ['sign']
  );
  const signature = await crypto.subtle.sign('HMAC', key, encoder.encode(payload));
  return base64UrlEncode(new Uint8Array(signature));
}

async function verify(payload: string, signature: string, secret: string): Promise<boolean> {
  const expectedSignature = await sign(payload, secret);
  return signature === expectedSignature;
}

export async function createToken(payload: UserPayload, secret: string): Promise<string> {
  const header = { alg: 'HS256', typ: 'JWT' };
  const headerB64 = await base64UrlEncode(new TextEncoder().encode(JSON.stringify(header)));
  const payloadB64 = await base64UrlEncode(new TextEncoder().encode(JSON.stringify(payload)));
  const signature = await sign(`${headerB64}.${payloadB64}`, secret);
  return `${headerB64}.${payloadB64}.${signature}`;
}

export async function verifyToken(token: string, secret: string): Promise<UserPayload | null> {
  try {
    const parts = token.split('.');
    if (parts.length !== 3) return null;

    const [headerB64, payloadB64, signature] = parts;

    const isValid = await verify(`${headerB64}.${payloadB64}`, signature, secret);
    if (!isValid) return null;

    const payloadData = await base64UrlDecode(payloadB64);
    const payload = JSON.parse(new TextDecoder().decode(payloadData)) as UserPayload;

    // Check expiration
    if (payload.exp && payload.exp < Date.now() / 1000) {
      return null;
    }

    return payload;
  } catch {
    return null;
  }
}

export function extractToken(authHeader: string | null): string | null {
  if (!authHeader) return null;
  const parts = authHeader.split(' ');
  if (parts.length !== 2 || parts[0] !== 'Bearer') return null;
  return parts[1];
}
