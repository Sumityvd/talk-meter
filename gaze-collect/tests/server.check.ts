import { beforeEach, describe, expect, it } from "vitest";

import {
  CONSENT_VERSION, DAILY_CAP, MAX_SESSION_BYTES, PART_BYTES, handle, hashCode, startOfToday,
  type Participant, type SessionRow, type Store,
} from "../supabase/functions/gaze-upload/logic";

const NOW = Date.parse("2026-11-05T10:00:00Z");
const LIMIT = 700 * 1024 * 1024;
const SHA = "a".repeat(64);

class MemoryStore implements Store {
  participants = new Map<string, Participant>();
  sessions: SessionRow[] = [];
  parts = new Map<string, Map<string, number>>();
  signed: string[] = [];

  participantByHash = async (hash: string) => this.participants.get(hash) ?? null;
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

  it("enforces the daily cap by Indian day, on finished sessions", async () => {
    for (let i = 0; i < DAILY_CAP; i++) {
      await begin(`client-000${i}`);
      store.sessions[i].status = "uploaded";
    }
    expect((await fail(begin("client-0009"))).status).toBe(429);
    await begin("client-0000"); // a retry of a known session still works
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
