import { SignJWT, jwtVerify } from 'jose';
import type { JwtPayload } from '../types';

const ALG = 'HS256';
const EXPIRATION_HOURS = 24;

function getSecretKey(secret: string): Uint8Array {
  return new TextEncoder().encode(secret);
}

export async function signToken(payload: JwtPayload, secret: string): Promise<string> {
  return new SignJWT({ ...payload })
    .setProtectedHeader({ alg: ALG })
    .setExpirationTime(`${EXPIRATION_HOURS}h`)
    .setIssuedAt()
    .sign(getSecretKey(secret));
}

export async function verifyToken(token: string, secret: string): Promise<JwtPayload | null> {
  try {
    const { payload } = await jwtVerify(token, getSecretKey(secret));
    return payload as unknown as JwtPayload;
  } catch {
    return null;
  }
}
