// gaze-upload: the only way the website reaches the database and bucket (see logic.ts for the rules).
// Deploy with: supabase functions deploy gaze-upload --no-verify-jwt
// Secrets: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are provided by Supabase. Optional: GAZE_MAX_STORED_MB,
// GAZE_ALLOWED_ORIGIN (the site's address; without it any website could call this function).

import { createClient } from "npm:@supabase/supabase-js@2";

import { handle, HttpError, type Store } from "./logic.ts";

const BUCKET = "gaze-sessions";
const db = createClient(Deno.env.get("SUPABASE_URL")!, Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!, {
  auth: { persistSession: false },
});
const MAX_STORED = Number(Deno.env.get("GAZE_MAX_STORED_MB") ?? "700") * 1024 * 1024;
const cors = {
  "Access-Control-Allow-Origin": Deno.env.get("GAZE_ALLOWED_ORIGIN") ?? "*",
  "Access-Control-Allow-Headers": "authorization, apikey, content-type",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
};

// deno-lint-ignore no-explicit-any
function ok<T>({ data, error }: { data: T; error: any }): T {
  if (error) throw new Error(error.message);
  return data;
}

const store: Store = {
  participantByHash: async (hash) =>
    ok(await db.from("gaze_participants").select("id, alias, active, consent_version").eq("code_hash", hash).maybeSingle()),
  setConsent: async (id, version) => {
    ok(await db.from("gaze_participants").update({ consent_version: version, consent_at: new Date().toISOString() }).eq("id", id));
  },
  sessionsSince: async (id, since) =>
    ok(await db.from("gaze_sessions").select("*").eq("participant_id", id).gte("created_at", since)),
  lastLighting: async (id) => {
    const rows = ok(await db.from("gaze_sessions").select("lighting").eq("participant_id", id).neq("status", "uploading")
      .order("created_at", { ascending: false }).limit(1));
    return rows?.[0]?.lighting ?? null;
  },
  sessionByClientId: async (id, clientId) =>
    ok(await db.from("gaze_sessions").select("*").eq("participant_id", id).eq("client_id", clientId).maybeSingle()),
  sessionById: async (id, sessionId) =>
    ok(await db.from("gaze_sessions").select("*").eq("participant_id", id).eq("id", sessionId).maybeSingle()),
  createSession: async (row) => ok(await db.from("gaze_sessions").insert(row).select().single()),
  markUploaded: async (id) => {
    ok(await db.from("gaze_sessions").update({ status: "uploaded", uploaded_at: new Date().toISOString() }).eq("id", id));
  },
  storedBytes: async () => {
    const rows = ok(await db.from("gaze_sessions").select("size_bytes").neq("status", "synced"));
    return (rows ?? []).reduce((sum: number, r: { size_bytes: number }) => sum + Number(r.size_bytes), 0);
  },
  listParts: async (path) => {
    const files = ok(await db.storage.from(BUCKET).list(path, { limit: 100 }));
    // deno-lint-ignore no-explicit-any
    return new Map((files ?? []).map((f: any) => [f.name, Number(f.metadata?.size ?? -1)]));
  },
  signedUploadUrl: async (path) => {
    const signed = ok(await db.storage.from(BUCKET).createSignedUploadUrl(path, { upsert: true }));
    return signed.signedUrl;
  },
};

Deno.serve(async (request) => {
  if (request.method === "OPTIONS") return new Response("ok", { headers: cors });
  const reply = (status: number, body: unknown) =>
    new Response(JSON.stringify(body), { status, headers: { ...cors, "Content-Type": "application/json" } });
  if (request.method !== "POST") return reply(405, { error: "POST only" });
  try {
    return reply(200, await handle(store, await request.json(), Date.now(), MAX_STORED));
  } catch (error) {
    if (error instanceof HttpError) return reply(error.status, { error: error.message });
    console.error(error);
    return reply(500, { error: "Server error" });
  }
});
