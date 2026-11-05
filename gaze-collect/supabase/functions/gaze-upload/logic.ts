/**
 * The rules of gaze-upload, separate from Supabase so they can be tested without a project.
 * Every request carries the participant's invite code; nothing here trusts the browser otherwise.
 */

export const CONSENT_VERSION = "2026-11-05";
export const PART_BYTES = 20 * 1024 * 1024;
export const MAX_SESSION_BYTES = 120 * 1024 * 1024;
export const DAILY_CAP = 3;
export const LIGHTINGS = ["daylight", "lamp-front", "ceiling", "dim"];
const IST_OFFSET_MS = 5.5 * 3600 * 1000;

export interface Participant {
  id: string;
  alias: string;
  active: boolean;
  consent_version: string | null;
}

export interface SessionRow {
  id: string;
  participant_id: string;
  client_id: string;
  status: "uploading" | "uploaded" | "synced";
  size_bytes: number;
  part_count: number;
  sha256: string;
  lighting: string;
  created_at: string;
}

export interface Store {
  participantByHash(hash: string): Promise<Participant | null>;
  setConsent(participantId: string, version: string): Promise<void>;
  sessionsSince(participantId: string, sinceIso: string): Promise<SessionRow[]>;
  lastLighting(participantId: string): Promise<string | null>;
  sessionByClientId(participantId: string, clientId: string): Promise<SessionRow | null>;
  sessionById(participantId: string, sessionId: string): Promise<SessionRow | null>;
  createSession(row: Omit<SessionRow, "id" | "status" | "created_at">): Promise<SessionRow>;
  markUploaded(sessionId: string): Promise<void>;
  /** Bytes of sessions still held in the bucket (not yet synced to the owner's PC). */
  storedBytes(): Promise<number>;
  /** part file name -> size, for one session's folder. */
  listParts(path: string): Promise<Map<string, number>>;
  signedUploadUrl(path: string): Promise<string>;
}

export class HttpError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

export async function hashCode(code: string): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(code.trim().toUpperCase()));
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

/** Start of the current day in India, where participants are, as an ISO time. */
export function startOfToday(nowMs: number): string {
  const local = nowMs + IST_OFFSET_MS;
  return new Date(local - (local % 86400000) - IST_OFFSET_MS).toISOString();
}

const partName = (index: number) => `part-${String(index).padStart(3, "0")}`;
const folder = (participantId: string, sessionId: string) => `${participantId}/${sessionId}`;

async function participantFor(store: Store, code: unknown): Promise<Participant> {
  if (typeof code !== "string" || code.trim().length < 6 || code.length > 64) throw new HttpError(401, "Invalid code");
  const participant = await store.participantByHash(await hashCode(code));
  if (!participant || !participant.active) throw new HttpError(401, "Invalid code");
  return participant;
}

function requireConsent(participant: Participant) {
  if (participant.consent_version !== CONSENT_VERSION) throw new HttpError(403, "Consent required");
}

async function completedToday(store: Store, participantId: string, nowMs: number): Promise<number> {
  const rows = await store.sessionsSince(participantId, startOfToday(nowMs));
  return rows.filter((r) => r.status !== "uploading").length;
}

async function uploadPlan(store: Store, session: SessionRow) {
  const path = folder(session.participant_id, session.id);
  const existing = await store.listParts(path);
  const parts: { index: number; url: string }[] = [];
  for (let index = 0; index < session.part_count; index++) {
    const name = partName(index);
    const expected = index < session.part_count - 1 ? PART_BYTES : session.size_bytes - PART_BYTES * index;
    if (existing.get(name) !== expected) parts.push({ index, url: await store.signedUploadUrl(`${path}/${name}`) });
  }
  return { sessionId: session.id, status: session.status, parts };
}

// deno-lint-ignore no-explicit-any
export async function handle(store: Store, body: any, nowMs: number, maxStoredBytes: number): Promise<unknown> {
  const participant = await participantFor(store, body?.code);
  switch (body?.action) {
    case "status":
      return {
        alias: participant.alias,
        consented: participant.consent_version === CONSENT_VERSION,
        consentVersion: CONSENT_VERSION,
        today: await completedToday(store, participant.id, nowMs),
        cap: DAILY_CAP,
        lastLighting: await store.lastLighting(participant.id),
      };

    case "consent":
      if (body.version !== CONSENT_VERSION) throw new HttpError(400, "Outdated consent text");
      await store.setConsent(participant.id, CONSENT_VERSION);
      return { consented: true };

    case "begin": {
      requireConsent(participant);
      const { clientId, size, sha256, lighting } = body;
      if (typeof clientId !== "string" || !/^[\w-]{8,64}$/.test(clientId)) throw new HttpError(400, "Bad session id");
      if (!Number.isInteger(size) || size <= 0 || size > MAX_SESSION_BYTES) throw new HttpError(413, "Session too large");
      if (typeof sha256 !== "string" || !/^[0-9a-f]{64}$/.test(sha256)) throw new HttpError(400, "Bad checksum");
      if (!LIGHTINGS.includes(lighting)) throw new HttpError(400, "Unknown lighting");

      const known = await store.sessionByClientId(participant.id, clientId);
      if (known) {
        if (known.sha256 !== sha256 || known.size_bytes !== size) throw new HttpError(409, "Session changed");
        return uploadPlan(store, known); // a retry: only the parts still missing
      }
      if ((await completedToday(store, participant.id, nowMs)) >= DAILY_CAP) {
        throw new HttpError(429, `Daily limit reached (${DAILY_CAP} sessions). Thank you! Please come back tomorrow.`);
      }
      if ((await store.storedBytes()) + size > maxStoredBytes) {
        throw new HttpError(507, "Storage is full for now. Your session is safe in this browser and will upload later.");
      }
      const session = await store.createSession({
        participant_id: participant.id,
        client_id: clientId,
        size_bytes: size,
        part_count: Math.ceil(size / PART_BYTES),
        sha256,
        lighting,
      });
      return uploadPlan(store, session);
    }

    case "complete": {
      requireConsent(participant);
      if (typeof body.sessionId !== "string") throw new HttpError(400, "Bad session id");
      const session = await store.sessionById(participant.id, body.sessionId);
      if (!session) throw new HttpError(404, "Unknown session");
      if (session.status !== "uploading") return { status: session.status };
      const plan = await uploadPlan(store, session);
      if (plan.parts.length) throw new HttpError(409, `${plan.parts.length} part(s) missing`);
      await store.markUploaded(session.id);
      return { status: "uploaded" };
    }

    default:
      throw new HttpError(400, "Unknown action");
  }
}
