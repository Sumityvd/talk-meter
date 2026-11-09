/** Talks to the gaze-upload edge function and keeps finished sessions safe until they're uploaded. */

const PART_BYTES = 20 * 1024 * 1024; // same as gaze-upload/logic.ts
const URL_BASE = import.meta.env.VITE_SUPABASE_URL as string | undefined;
const ANON_KEY = import.meta.env.VITE_SUPABASE_ANON_KEY as string | undefined;

/** Development only (`?offline=1` with `npm run dev`): no server, the finished file stays in the browser. */
export const OFFLINE = import.meta.env.DEV && new URLSearchParams(location.search).has("offline");
export const CONFIGURED = OFFLINE || (!!URL_BASE && !!ANON_KEY);

export interface Status {
  alias: string;
  consented: boolean;
  consentVersion: string;
  today: number;
  lastLighting: string | null;
}

export interface PendingSession {
  clientId: string;
  file: Blob;
  sha256: string;
  lighting: string;
}

const offlineStatus: Status = { alias: "TEST", consented: true, consentVersion: "offline", today: 0, lastLighting: null };

async function call<T>(body: Record<string, unknown>): Promise<T> {
  const response = await fetch(`${URL_BASE}/functions/v1/gaze-upload`, {
    method: "POST",
    headers: { "Content-Type": "application/json", apikey: ANON_KEY!, Authorization: `Bearer ${ANON_KEY}` },
    body: JSON.stringify(body),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error ?? `Server error ${response.status}`);
  return data as T;
}

export const getStatus = (code: string): Promise<Status> =>
  OFFLINE ? Promise.resolve(offlineStatus) : call<Status>({ action: "status", code });

export const giveConsent = (code: string, version: string) =>
  OFFLINE ? Promise.resolve() : call({ action: "consent", code, version });

export async function uploadSession(code: string, session: PendingSession, onProgress: (fraction: number) => void) {
  if (OFFLINE) {
    (window as unknown as { __gazeSession: Blob }).__gazeSession = session.file;
    onProgress(1);
    return;
  }
  const plan = await call<{ sessionId: string; status: string; parts: { index: number; url: string }[] }>({
    action: "begin", code, clientId: session.clientId, size: session.file.size, sha256: session.sha256, lighting: session.lighting,
  });
  const totalParts = Math.ceil(session.file.size / PART_BYTES);
  let done = totalParts - plan.parts.length;
  onProgress(done / totalParts);
  for (const part of plan.parts) {
    const chunk = session.file.slice(part.index * PART_BYTES, (part.index + 1) * PART_BYTES, "application/octet-stream");
    for (let attempt = 1; ; attempt++) {
      try {
        const response = await fetch(part.url, { method: "PUT", headers: { "Content-Type": "application/octet-stream", "x-upsert": "true" }, body: chunk });
        if (!response.ok) throw new Error(`Upload failed (${response.status})`);
        break;
      } catch (error) {
        if (attempt >= 4) throw error;
        await new Promise((r) => setTimeout(r, 2000 * attempt));
      }
    }
    onProgress(++done / totalParts);
  }
  await call({ action: "complete", code, sessionId: plan.sessionId });
}

// A finished session waits in IndexedDB, so closing the tab or losing the network can't lose it.
const openDb = () =>
  new Promise<IDBDatabase>((resolve, reject) => {
    const request = indexedDB.open("gaze-collect", 1);
    request.onupgradeneeded = () => request.result.createObjectStore("pending", { keyPath: "clientId" });
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });

async function withStore<T>(mode: IDBTransactionMode, run: (store: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  const db = await openDb();
  return new Promise<T>((resolve, reject) => {
    const request = run(db.transaction("pending", mode).objectStore("pending"));
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  }).finally(() => db.close());
}

export const savePending = (session: PendingSession) => withStore("readwrite", (s) => s.put(session));
export const removePending = (clientId: string) => withStore("readwrite", (s) => s.delete(clientId));
export const loadPending = () => withStore<PendingSession[]>("readonly", (s) => s.getAll());
