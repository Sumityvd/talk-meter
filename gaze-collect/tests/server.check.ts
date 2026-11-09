import { beforeEach, describe, expect, it } from "vitest";

import {
  CONSENT_VERSION, MAX_SESSION_BYTES, NEW_PARTICIPANTS_PER_HOUR, PART_BYTES, handle, hashCode, startOfToday,
  type Participant, type SessionRow, type Store,
} from "../supabase/functions/gaze-upload/logic";
import { CONSENT_VERSION as SITE_CONSENT_VERSION } from "../src/lib/protocol";

const NOW = Date.parse("2026-11-05T10:00:00Z");
const LIMIT = 700 * 1024 * 1024;
const SHA = "a".repeat(64);

class MemoryStore implements Store {
  participants = new Map<string, Participant>();
  sessions: SessionRow[] = [];
  parts = new Map<string, Map<string, number>>();
  signed: string[] = [];

  created: string[] = [];
  takenAliases = new Set<string>();
  participantByHash = async (hash: string) => this.participants.get(hash) ?? null;
  createParticipant = async (row: { alias: string; code_hash: string; consent_version: string }) => {
    const aliases = [...this.participants.values()].map((p) => p.alias);
    if (this.participants.has(row.code_hash) || aliases.includes(row.alias) || this.takenAliases.has(row.alias)) return null;
    const participant = { id: `p${this.participants.size + 1}`, alias: row.alias, active: true, consent_version: row.consent_version };
    this.participants.set(row.code_hash, participant);
    this.created.push(this.now);
    return participant;
  };
  now = new Date(NOW).toISOString();
  participantsCreatedSince = async (since: string) => this.created.filter((t) => t >= since).length;
  setConsent = async (id: string, version: string) => {
    for (const p of this.participants.values()) if (p.id === id) p.consent_version = version;
  };
  sessionsSince = async (id: string, since: string) =>
    this.sessions.filter((s) => s.participant_id === id && s.created_at >= since);
  lastLighting = async (id: string) =>
    this.sessions.filter((s) => s.participant_id === id && s.status !== "uploading").at(-1)?.lighting ?? null;
  sessionByClientId = async (id: string, clientId: string) =>
    this.sessions.find((s) => s.participant_id === id && s.client_id === clientId) ?? null;
  sessionById = async (id: string, sessionId: string) =>
    this.sessions.find((s) => s.participant_id === id && s.id === sessionId) ?? null;
  createSession = async (row: Omit<SessionRow, "id" | "status" | "created_at">) => {
    const session: SessionRow = { ...row, id: `s${this.sessions.length}`, status: "uploading", created_at: new Date(NOW).toISOString() };
    this.sessions.push(session);
    return session;
  };
  markUploaded = async (id: string) => {
    this.sessions.find((s) => s.id === id)!.status = "uploaded";
  };
  storedBytes = async () => this.sessions.filter((s) => s.status !== "synced").reduce((sum, s) => sum + s.size_bytes, 0);
  listParts = async (path: string) => this.parts.get(path) ?? new Map<string, number>();
  signedUploadUrl = async (path: string) => {
    this.signed.push(path);
    return `https://signed/${path}`;
  };
}

describe("gaze-upload rules", () => {
  let store: MemoryStore;
  const CODE = "ABCDEF2345";
  const call = (body: Record<string, unknown>, nowMs = NOW, limit = LIMIT) => handle(store, { code: CODE, ...body }, nowMs, limit);
  const begin = (clientId = "client-0001", size = 30 * 1024 * 1024) =>
    call({ action: "begin", clientId, size, sha256: SHA, lighting: "daylight" }) as Promise<{ sessionId: string; parts: { index: number; url: string }[] }>;
  const fail = (promise: Promise<unknown>) => promise.then(() => { throw new Error("expected failure"); }, (e) => e);

  beforeEach(async () => {
    store = new MemoryStore();
    store.participants.set(await hashCode(CODE), { id: "p1", alias: "P01", active: true, consent_version: CONSENT_VERSION });
  });

  it("refuses wrong, short and deactivated codes", async () => {
    expect((await fail(handle(store, { code: "WRONGCODE1", action: "status" }, NOW, LIMIT))).status).toBe(401);
    expect((await fail(handle(store, { code: "abc", action: "status" }, NOW, LIMIT))).status).toBe(401);
    expect((await fail(handle(store, { action: "status" }, NOW, LIMIT))).status).toBe(401);
    (await store.participantByHash(await hashCode(CODE)))!.active = false;
    expect((await fail(call({ action: "status" }))).status).toBe(401);
  });

  it("lets anyone join by agreeing, with a private code and an anonymous ID", async () => {
    const register = (body: Record<string, unknown>) => handle(store, { action: "register", ...body }, NOW, LIMIT);
    expect((await fail(register({ version: "2026-11-05", adult: true }))).status).toBe(400);
    expect((await fail(register({ version: CONSENT_VERSION }))).status).toBe(400);
    const joined = await register({ version: CONSENT_VERSION, adult: true }) as { code: string; alias: string };
    expect(joined.code).toMatch(/^[A-Z2-9]{20}$/);
    expect(joined.alias).toMatch(/^V[A-Z2-9]{6}$/);
    const status = await handle(store, { action: "status", code: joined.code }, NOW, LIMIT);
    expect(status).toMatchObject({ alias: joined.alias, consented: true, today: 0 });
    const plan = await handle(store, { action: "begin", code: joined.code, clientId: "client-0001", size: 10, sha256: SHA, lighting: "dim" }, NOW, LIMIT);
    expect(plan).toMatchObject({ sessionId: "s0" });
    expect(store.sessions[0].participant_id).not.toBe("p1"); // not filed under anyone else
  });

  it("limits new participants per hour across the site, and retries a taken ID", async () => {
    const register = (nowMs = NOW) => handle(store, { action: "register", version: CONSENT_VERSION, adult: true }, nowMs, LIMIT);
    const real = crypto.getRandomValues.bind(crypto);
    let calls = 0;
    // The first alias drawn is taken: registration must try again rather than fail.
    crypto.getRandomValues = (<T extends ArrayBufferView>(array: T) => (calls++ === 1 ? (array as unknown as Uint8Array).fill(0) : real(array))) as typeof crypto.getRandomValues;
    store.takenAliases.add("VAAAAAA");
    try {
      expect(await register()).toMatchObject({ alias: expect.not.stringMatching(/^VAAAAAA$/) });
    } finally {
      crypto.getRandomValues = real;
    }
    for (let i = 1; i < NEW_PARTICIPANTS_PER_HOUR; i++) await register();
    expect((await fail(register())).status).toBe(429);
    store.now = new Date(NOW + 2 * 3600 * 1000).toISOString();
    expect(await register(NOW + 2 * 3600 * 1000)).toHaveProperty("code");
  });

  it("uses the same consent version as the site's text", () => {
    expect(SITE_CONSENT_VERSION).toBe(CONSENT_VERSION);
  });

  it("accepts a code in any letter case", async () => {
    expect(await handle(store, { code: " abcdef2345 ", action: "status" }, NOW, LIMIT)).toMatchObject({ alias: "P01" });
  });

  it("requires consent before any upload", async () => {
    await store.setConsent("p1", "");
    expect((await fail(begin())).status).toBe(403);
    expect((await fail(call({ action: "consent", version: "old" }))).status).toBe(400);
    await call({ action: "consent", version: CONSENT_VERSION });
    expect((await begin()).parts).toHaveLength(2);
  });

  it("rejects oversized and malformed sessions", async () => {
    expect((await fail(begin("client-0001", MAX_SESSION_BYTES + 1))).status).toBe(413);
    expect((await fail(call({ action: "begin", clientId: "../../etc", size: 10, sha256: SHA, lighting: "daylight" }))).status).toBe(400);
    expect((await fail(call({ action: "begin", clientId: "client-0001", size: 10, sha256: "x", lighting: "daylight" }))).status).toBe(400);
    expect((await fail(call({ action: "begin", clientId: "client-0001", size: 10, sha256: SHA, lighting: "disco" }))).status).toBe(400);
    expect((await fail(call({ action: "nonsense" }))).status).toBe(400);
  });

  it("only completes once every part has arrived at its exact size", async () => {
    const { sessionId } = await begin();
    expect((await fail(call({ action: "complete", sessionId }))).status).toBe(409);
    const path = `p1/${sessionId}`;
    store.parts.set(path, new Map([["part-000", PART_BYTES], ["part-001", 1]]));
    expect((await fail(call({ action: "complete", sessionId }))).status).toBe(409);
    store.parts.set(path, new Map([["part-000", PART_BYTES], ["part-001", 10 * 1024 * 1024]]));
    expect(await call({ action: "complete", sessionId })).toEqual({ status: "uploaded" });
    expect(await call({ action: "complete", sessionId })).toEqual({ status: "uploaded" });
  });

  it("resumes: a retry asks only for missing parts and creates no second session", async () => {
    const first = await begin();
    store.parts.set(`p1/${first.sessionId}`, new Map([["part-000", PART_BYTES]]));
    const retry = await begin();
    expect(retry.sessionId).toBe(first.sessionId);
    expect(retry.parts.map((p) => p.index)).toEqual([1]);
    expect(store.sessions).toHaveLength(1);
    expect((await fail(call({ action: "begin", clientId: "client-0001", size: 5, sha256: SHA, lighting: "daylight" }))).status).toBe(409);
  });

  it("has no daily limit, and counts finished sessions by Indian day", async () => {
    for (let i = 0; i < 5; i++) {
      await begin(`client-000${i}`);
      store.sessions[i].status = "uploaded";
    }
    const sixth = await begin("client-0009");
    expect(sixth.sessionId).toBe("s5");
    expect(store.sessions).toHaveLength(6);
    const status = await call({ action: "status" });
    expect(status).toMatchObject({ today: 5 });
    expect(status).not.toHaveProperty("cap");
    const tomorrow = NOW + 24 * 3600 * 1000;
    expect(await call({ action: "status" }, tomorrow)).toMatchObject({ today: 0 });
    expect(startOfToday(Date.parse("2026-11-05T18:29:00Z"))).toBe("2026-11-04T18:30:00.000Z");
    expect(startOfToday(Date.parse("2026-11-05T18:31:00Z"))).toBe("2026-11-05T18:30:00.000Z");
  });

  it("stops new sessions when storage is nearly full, keeping them for later", async () => {
    await begin();
    expect((await fail(call({ action: "begin", clientId: "client-0002", size: 20, sha256: SHA, lighting: "dim" }, NOW, 30 * 1024 * 1024))).status).toBe(507);
  });

  it("can't reach another participant's session", async () => {
    const { sessionId } = await begin();
    const other = "ZZZZZZ2345";
    store.participants.set(await hashCode(other), { id: "p2", alias: "P02", active: true, consent_version: CONSENT_VERSION });
    expect((await fail(handle(store, { code: other, action: "complete", sessionId }, NOW, LIMIT))).status).toBe(404);
  });

  it("signs upload paths inside the participant's own folder", async () => {
    const { sessionId } = await begin();
    expect(store.signed).toEqual([`p1/${sessionId}/part-000`, `p1/${sessionId}/part-001`]);
  });
});
