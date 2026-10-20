import { useCallback, useEffect, useRef, useState } from "react";
import type { FaceLandmarker } from "@mediapipe/tasks-vision";

import { downloadText, minutesCsv, secondsCsv } from "./lib/csv";
import {
  ACTIVITIES,
  SessionRecorder,
  type Activity,
  type SecondRecord,
} from "./lib/recorder";
import { closeCamera, createLandmarker, openCamera, startLoop } from "./lib/tracker";

type Status = "idle" | "loading" | "recording" | "stopped" | "error";

const STATE_TEXT: Record<SecondRecord["state"], string> = {
  talking: "Talking",
  smiling: "Smiling",
  quiet: "Quiet",
  "no-face": "No face",
  sparse: "Too few frames",
};

const pct = (x: number) => `${Math.round(x * 100)}%`;
const stamp = (iso: string) => iso.replace(/[:.]/g, "-").slice(0, 19);

export default function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const landmarkerRef = useRef<FaceLandmarker | null>(null);
  const stopLoopRef = useRef<(() => void) | null>(null);
  const recorderRef = useRef<SessionRecorder | null>(null);
  const labelRef = useRef<Activity>("reading");

  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState<string | null>(null);
  const [label, setLabel] = useState<Activity>("reading");
  const [last, setLast] = useState<SecondRecord | null>(null);
  const [sessionStart, setSessionStart] = useState<string | null>(null);
  const [, setTick] = useState(0);

  const chooseLabel = useCallback((a: Activity) => {
    labelRef.current = a;
    setLabel(a);
  }, []);

  const start = useCallback(async () => {
    const video = videoRef.current;
    if (!video) return;
    setError(null);
    setStatus("loading");
    try {
      streamRef.current = await openCamera(video);
      landmarkerRef.current ??= await createLandmarker();
      const rec = new SessionRecorder();
      recorderRef.current = rec;
      setSessionStart(new Date().toISOString());
      setLast(null);
      stopLoopRef.current = startLoop(video, landmarkerRef.current, (f) => {
        const closed = rec.addFrame(f, labelRef.current);
        if (closed) {
          setLast(closed);
          setTick((n) => n + 1);
        }
      });
      setStatus("recording");
    } catch (e) {
      closeCamera(video, streamRef.current);
      streamRef.current = null;
      setError(e instanceof Error ? e.message : String(e));
      setStatus("error");
    }
  }, []);

  const stop = useCallback(() => {
    stopLoopRef.current?.();
    stopLoopRef.current = null;
    recorderRef.current?.finish();
    if (videoRef.current) closeCamera(videoRef.current, streamRef.current);
    streamRef.current = null;
    setStatus("stopped");
    setTick((n) => n + 1);
  }, []);

  // Number keys 1–8 pick the activity label, for hands-free scripted recordings.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const i = Number(e.key) - 1;
      if (i >= 0 && i < ACTIVITIES.length) chooseLabel(ACTIVITIES[i]);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [chooseLabel]);

  // Data lives only in this tab: warn before it is thrown away.
  useEffect(() => {
    const onUnload = (e: BeforeUnloadEvent) => {
      if (recorderRef.current?.seconds.length) e.preventDefault();
    };
    window.addEventListener("beforeunload", onUnload);
    return () => window.removeEventListener("beforeunload", onUnload);
  }, []);

  useEffect(() => () => stopLoopRef.current?.(), []);

  const rec = recorderRef.current;
  const minutes = rec?.minutes ?? [];
  const recording = status === "recording";
  const hasData = !!rec?.seconds.length;

  return (
    <main className="wrap">
      <header>
        <h1>talk-meter</h1>
        <p className="sub">
          Measures how much of each minute you spend talking, smiling or quiet, from your
          face alone. No audio is recorded, and nothing leaves this browser.
        </p>
      </header>

      <section className="grid">
        <div className="card">
          <video ref={videoRef} className="video" playsInline muted />
          <div className="row">
            {recording ? (
              <button className="primary" onClick={stop}>Stop</button>
            ) : (
              <button className="primary" onClick={start} disabled={status === "loading"}>
                {status === "loading" ? "Loading…" : hasData ? "Start new session" : "Start"}
              </button>
            )}
            {recording && last && (
              <span className={`chip ${last.state}`}>{STATE_TEXT[last.state]}</span>
            )}
          </div>
          {error && <p className="error">Couldn't start: {error}</p>}
          {recording && (
            <p className="hint">Keep this tab visible — browsers slow down hidden tabs.</p>
          )}
        </div>

        <div className="card">
          <h2>What are you doing?</h2>
          <p className="hint">
            Pick the activity as you go (or press 1–8). Each second is stored with it, so the
            detector can be checked against what really happened.
          </p>
          <div className="labels">
            {ACTIVITIES.map((a, i) => (
              <button
                key={a}
                className={a === label ? "label active" : "label"}
                onClick={() => chooseLabel(a)}
              >
                <kbd>{i + 1}</kbd> {a}
              </button>
            ))}
          </div>
        </div>
      </section>

      <section className="card">
        <div className="row spread">
          <h2>Minutes</h2>
          <div className="row">
            <button
              disabled={!hasData || !sessionStart}
              onClick={() =>
                rec && sessionStart &&
                downloadText(`talk-meter-minutes-${stamp(sessionStart)}.csv`, minutesCsv(sessionStart, rec.minutes))
              }
            >
              Download minutes CSV
            </button>
            <button
              disabled={!hasData || !sessionStart}
              onClick={() =>
                rec && sessionStart &&
                downloadText(`talk-meter-seconds-${stamp(sessionStart)}.csv`, secondsCsv(sessionStart, rec.seconds))
              }
            >
              Download seconds CSV
            </button>
          </div>
        </div>
        {minutes.length === 0 ? (
          <p className="hint">
            {recording ? "The first minute appears after 60 seconds." : "No minutes yet."}
          </p>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Minute</th>
                  <th>Activity</th>
                  <th>Talking</th>
                  <th>Smiling</th>
                  <th>Quiet seconds</th>
                  <th>Face visible</th>
                  <th>fps</th>
                </tr>
              </thead>
              <tbody>
                {minutes.map((m) => (
                  <tr key={m.minute}>
                    <td>{m.minute + 1}</td>
                    <td>{m.label}</td>
                    <td>{pct(m.talk_frac)}</td>
                    <td>{pct(m.smile_frac)}</td>
                    <td>{m.quiet_s} / {m.seconds}</td>
                    <td>{pct(m.faceRatio)}</td>
                    <td>{m.fps}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </main>
  );
}
