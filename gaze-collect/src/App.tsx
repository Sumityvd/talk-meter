import { useCallback, useEffect, useRef, useState } from "react";
import type { FaceLandmarker } from "@mediapipe/tasks-vision";

import {
  CONFIGURED, OFFLINE, ServerError, getStatus, giveConsent, loadPending, register, removePending, savePending, uploadSession,
  type PendingSession, type Status,
} from "./lib/backend";
import { CalibrationAborted, CalibrationRunner, type View } from "./lib/calibration";
import { checks, measure, rating, type Check, type Posture } from "./lib/posture";
import {
  BLINK_THRESHOLD, CAMERA_ABOVE_MM, CARD_WIDTH_MM, CONSENT_VERSION, LIGHTINGS, PLAUSIBLE_INCHES, SCREEN_INCHES, diagonalInches, hfovFor,
  mmPerCssFromInches, type CameraPosition,
} from "./lib/protocol";
import { buildSessionFile, sha256Hex, type Manifest } from "./lib/sessionFile";
import { createLandmarker, listCameras, measureBrightness, openCamera, startFrames, type VideoFrame } from "./lib/tracker";

interface Device {
  deviceId: string;
  label: string;
  position: CameraPosition;
  mmPerCss: number;
  /** How the screen size was given: a diagonal in inches, or "card". */
  screenChoice: number | "card";
  cardPx: number;
  dpr: number;
}

/** A saved device, unless it's from before screen size was checked (those must be set up again). */
const readDevice = () => {
  const device = read<Device>("gaze-device");
  return device?.mmPerCss ? device : null;
};

type Stage = "loading" | "consent" | "setup" | "home" | "starting" | "positioning" | "calibrating" | "saving";

const READY_SECONDS = 2;
const COUNTDOWN_SECONDS = 3;
const CONFIRM_SECONDS = 5;

const read = <T,>(key: string): T | null => {
  try {
    return JSON.parse(localStorage.getItem(key) ?? "null") as T | null;
  } catch {
    return null;
  }
};

export default function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const landmarkerRef = useRef<FaceLandmarker | null>(null);
  const stopFramesRef = useRef<(() => void) | null>(null);
  const frameHandler = useRef<((f: VideoFrame) => void) | null>(null);
  const sendingRef = useRef(false);

  const [stage, setStage] = useState<Stage>("loading");
  const [code, setCode] = useState(read<string>("gaze-code") ?? "");
  const [device, setDevice] = useState<Device | null>(readDevice());
  const [status, setStatus] = useState<Status | null>(null);
  const [lighting, setLighting] = useState<string>("");
  const [baseline, setBaseline] = useState<{ posture: Posture; warnings: string[]; expected: string } | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [upload, setUpload] = useState<{ fraction: number; error?: string } | null>(null);

  const refresh = useCallback(async (activeCode: string) => {
    const next = await getStatus(activeCode);
    setStatus(next);
    return next;
  }, []);

  const sendPending = useCallback(async (activeCode: string) => {
    if (sendingRef.current) return; // one upload loop at a time; it picks up anything saved meanwhile
    sendingRef.current = true;
    try {
      for (let pending = await loadPending(); pending.length; pending = await loadPending()) {
        setUpload({ fraction: 0 });
        await uploadSession(activeCode, pending[0], (fraction) => setUpload({ fraction }));
        await removePending(pending[0].clientId);
        setUpload(null);
      }
      await refresh(activeCode).catch(() => undefined);
    } catch (error) {
      setUpload({ fraction: 0, error: error instanceof Error ? error.message : String(error) });
    } finally {
      sendingRef.current = false;
    }
  }, [refresh]);

  // Start: a new browser sees the consent text; a known one gets its status, which decides the next screen.
  // A finished session from an earlier visit resumes uploading.
  useEffect(() => {
    if (!CONFIGURED) return;
    const activeCode = OFFLINE ? "OFFLINE" : code;
    if (!activeCode) {
      setStage("consent");
      return;
    }
    refresh(activeCode)
      .then((next) => {
        setCode(activeCode);
        if (!next.consented) setStage("consent");
        else if (!readDevice()) setStage("setup");
        else {
          setStage("home");
          void sendPending(activeCode);
        }
      })
      .catch((error) => {
        // Only a refused code starts over. A network problem must not give this person a second ID.
        if (error instanceof ServerError && error.status === 401) {
          localStorage.removeItem("gaze-code");
          setCode("");
          setStage("consent");
        } else {
          setMessage("Couldn't reach the server. Please check your connection and reload the page.");
        }
      });
    // Runs once on load; later changes go through the screens' own handlers.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const stopCamera = useCallback(() => {
    stopFramesRef.current?.();
    stopFramesRef.current = null;
    frameHandler.current = null;
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
  }, []);

  const backHome = useCallback((text: string | null) => {
    stopCamera();
    if (document.fullscreenElement) void document.exitFullscreen();
    setMessage(text);
    setStage("home");
  }, [stopCamera]);

  // Must run inside the click: browsers only allow full screen from a user gesture.
  const startSession = useCallback(async () => {
    if (!device) return;
    if (Math.abs(device.dpr - window.devicePixelRatio) > 0.01) {
      setMessage("Your browser zoom or display scaling changed, so please measure your screen again.");
      setStage("setup");
      return;
    }
    setMessage(null);
    setStage("starting");
    try {
      await document.documentElement.requestFullscreen();
      streamRef.current = await openCamera(videoRef.current!, device.deviceId);
      landmarkerRef.current ??= await createLandmarker();
      stopFramesRef.current = startFrames(videoRef.current!, landmarkerRef.current, (f) => frameHandler.current?.(f));
      setStage("positioning");
    } catch (error) {
      backHome(`Could not start: ${error instanceof Error ? error.message : String(error)}. Check the camera isn't in use by another app.`);
    }
  }, [device, backHome]);

  // Leaving full screen or the tab invalidates dot positions and timing, so it ends the session.
  useEffect(() => {
    if (stage !== "positioning" && stage !== "calibrating") return;
    const voidSession = () => {
      if (!document.fullscreenElement || document.hidden) {
        backHome("The session stopped because you left full screen or the tab. Nothing was saved. Please try again.");
      }
    };
    document.addEventListener("fullscreenchange", voidSession);
    document.addEventListener("visibilitychange", voidSession);
    return () => {
      document.removeEventListener("fullscreenchange", voidSession);
      document.removeEventListener("visibilitychange", voidSession);
    };
  }, [stage, backHome]);

  const finishRecording = useCallback(async (file: Blob) => {
    setStage("saving");
    const session: PendingSession = { clientId: crypto.randomUUID(), file, sha256: await sha256Hex(file), lighting };
    await savePending(session);
    backHome("Session recorded. Thank you!");
    await sendPending(code);
  }, [lighting, code, backHome, sendPending]);

  const videoClass = stage === "positioning" ? "video preview" : "video hidden";
  const fullScreen = stage === "positioning" || stage === "calibrating" || stage === "starting" || stage === "saving";

  return (
    <div className={fullScreen ? "stage" : "wrap"}>
      {/* Always mounted and never display:none: browsers stop delivering frames to a hidden video. */}
      <video ref={videoRef} className={videoClass} muted playsInline />

      {!CONFIGURED && <p className="error">This site isn't configured yet (missing Supabase settings).</p>}

      {!fullScreen && (
        <>
          <h1>Gaze data collection</h1>
          <p className="sub">Help train a webcam eye tracker. A session takes about 2 minutes: you look at dots on the screen.</p>
          {message && <p className="notice">{message}</p>}
        </>
      )}

      {stage === "loading" && <p>Loading ...</p>}
      {stage === "consent" && (
        <Consent onAgree={async () => {
          let activeCode = code;
          if (activeCode) {
            await giveConsent(activeCode, CONSENT_VERSION); // the text changed since this browser last agreed
          } else {
            activeCode = (await register(CONSENT_VERSION)).code;
            localStorage.setItem("gaze-code", JSON.stringify(activeCode));
            setCode(activeCode);
          }
          await refresh(activeCode);
          if (device) {
            setStage("home");
            void sendPending(activeCode);
          } else {
            setStage("setup");
          }
        }} />
      )}
      {stage === "setup" && (
        <Setup video={videoRef} initial={device} onDone={(next) => {
          localStorage.setItem("gaze-device", JSON.stringify(next));
          setDevice(next);
          stopCamera();
          setMessage(null);
          setStage("home");
        }} stream={streamRef} />
      )}
      {stage === "home" && status && (
        <Home
          status={status}
          lighting={lighting}
          onLighting={setLighting}
          upload={upload}
          onRetry={() => void sendPending(code)}
          onStart={() => void startSession()}
          onSetup={() => setStage("setup")}
        />
      )}
      {stage === "starting" && <p className="center-text">Starting the camera ...</p>}
      {stage === "saving" && <p className="center-text">Saving your session ...</p>}
      {stage === "positioning" && device && (
        <Positioning
          video={videoRef}
          frames={frameHandler}
          hfovDeg={hfovFor(device.label)}
          onReady={(posture, warnings, expected) => {
            setBaseline({ posture, warnings, expected });
            setStage("calibrating");
          }}
        />
      )}
      {stage === "calibrating" && device && baseline && (
        <Calibrating
          video={videoRef}
          frames={frameHandler}
          device={device}
          lighting={lighting}
          consentVersion={status?.consentVersion ?? ""}
          baseline={baseline}
          onFinished={(file) => void finishRecording(file)}
          onFailed={(text) => backHome(text)}
        />
      )}
    </div>
  );
}


function Consent({ onAgree }: { onAgree: () => Promise<void> }) {
  const [adult, setAdult] = useState(false);
  const [agree, setAgree] = useState(false);
  const [error, setError] = useState<string | null>(null);
  return (
    <div className="card">
      <h2>What you're agreeing to</h2>
      <ul>
        <li><b>What's collected:</b> during each session only, images of your face from your webcam, face landmark points, where each dot was, your screen size, camera name and the lighting you choose. No audio. The camera is off the rest of the time.</li>
        <li><b>Why:</b> to train and test a webcam gaze-tracking model, for non-commercial research only.</li>
        <li><b>Where it goes:</b> it's uploaded to a private Supabase store, then moved to the organiser's own computer and deleted from Supabase. Your name isn't collected; you're identified only by a random participant ID for this browser.</li>
        <li><b>Your choice:</b> taking part is voluntary and you can stop at any time. Your participant ID is shown on the next screens; to have your data deleted, give it to the person who shared this site with you.</li>
        <li><b>Not public:</b> your images won't be published or shared. Models trained from them may be.</li>
      </ul>
      <label><input type="checkbox" checked={adult} onChange={(e) => setAdult(e.target.checked)} /> I am 18 or older.</label>
      <label><input type="checkbox" checked={agree} onChange={(e) => setAgree(e.target.checked)} /> I agree to my data being collected and used as described above.</label>
      <button className="primary" disabled={!adult || !agree} onClick={() => onAgree().catch((e) => setError(String(e.message ?? e)))}>I agree</button>
      {error && <p className="error">{error}</p>}
    </div>
  );
}

function Setup({ video, stream, initial, onDone }: {
  video: React.RefObject<HTMLVideoElement>;
  stream: React.MutableRefObject<MediaStream | null>;
  initial: Device | null;
  onDone: (device: Device) => void;
}) {
  const [cameras, setCameras] = useState<MediaDeviceInfo[]>([]);
  const [deviceId, setDeviceId] = useState(initial?.deviceId ?? "");
  const [position, setPosition] = useState<CameraPosition | "other" | "">(initial?.position ?? "");
  const [screenChoice, setScreenChoice] = useState<number | "card" | "">(initial?.screenChoice ?? "");
  const [cardPx, setCardPx] = useState(initial?.cardPx ?? 324);
  const [error, setError] = useState<string | null>(null);

  const [cssWidth, cssHeight] = [window.screen.width, window.screen.height];
  const mmPerCss = screenChoice === "card" ? CARD_WIDTH_MM / cardPx
    : screenChoice === "" ? 0 : mmPerCssFromInches(screenChoice, cssWidth, cssHeight);
  const inches = diagonalInches(mmPerCss, cssWidth, cssHeight);
  const plausible = inches >= PLAUSIBLE_INCHES[0] && inches <= PLAUSIBLE_INCHES[1];

  useEffect(() => {
    let active = true;
    (async () => {
      try {
        stream.current?.getTracks().forEach((t) => t.stop());
        stream.current = await openCamera(video.current!, deviceId || undefined);
        const found = await listCameras();
        if (!active) return;
        setCameras(found);
        const current = stream.current.getVideoTracks()[0].getSettings().deviceId;
        if (!deviceId && current) setDeviceId(current);
      } catch (e) {
        if (active) setError(`Camera problem: ${e instanceof Error ? e.message : String(e)}. Please allow camera access.`);
      }
    })();
    return () => {
      active = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [deviceId]);

  return (
    <div className="card">
      <h2>Set up this computer (once)</h2>
      <h3>1. How big is the screen you'll use?</h3>
      <p>The size in inches, measured corner to corner. Laptops are usually 14 or 15.6 inches; it's in the computer's specifications.</p>
      <select aria-label="Screen size" value={screenChoice} onChange={(e) => setScreenChoice(e.target.value === "card" ? "card" : e.target.value === "" ? "" : Number(e.target.value))}>
        <option value="">Choose ...</option>
        {SCREEN_INCHES.map((size) => <option key={size} value={size}>{size} inches</option>)}
        <option value="card">I don't know: measure with a card</option>
      </select>
      {screenChoice === "card" && (
        <>
          <p>Hold a bank or ID card flat against the screen, long side level. Move the slider until the blue box is exactly as wide as the card.</p>
          <div className="card-box" style={{ width: cardPx, height: cardPx * 0.63 }} />
          <input type="range" min={150} max={680} value={cardPx} onChange={(e) => setCardPx(Number(e.target.value))} aria-label="Card width" />
        </>
      )}
      {screenChoice !== "" && (plausible ? (
        <p className="muted">Your screen: about {inches.toFixed(1)} inches ({(mmPerCss * cssWidth / 10).toFixed(1)} x {(mmPerCss * cssHeight / 10).toFixed(1)} cm). If that looks wrong, please correct it: the data depends on it.</p>
      ) : (
        <p className="error">That would make your screen {inches.toFixed(1)} inches, which can't be right. Check that the box matches the card's long side exactly.</p>
      ))}

      <h3>2. Choose your camera</h3>
      <select value={deviceId} onChange={(e) => setDeviceId(e.target.value)}>
        {cameras.map((c) => <option key={c.deviceId} value={c.deviceId}>{c.label || "Camera"}</option>)}
      </select>

      <h3>3. Where is the camera?</h3>
      <label><input type="radio" name="position" checked={position === "built-in"} onChange={() => setPosition("built-in")} /> Built into the laptop screen, top centre</label>
      <label><input type="radio" name="position" checked={position === "clip-on"} onChange={() => setPosition("clip-on")} /> External, clipped on top of the screen, centred</label>
      <label><input type="radio" name="position" checked={position === "other"} onChange={() => setPosition("other")} /> Somewhere else</label>
      {position === "other" && <p className="error">The data needs the camera at the top centre of the screen you'll use. Please move it there.</p>}

      {error && <p className="error">{error}</p>}
      <button
        className="primary"
        disabled={!deviceId || !plausible || (position !== "built-in" && position !== "clip-on")}
        onClick={() => onDone({
          deviceId,
          label: cameras.find((c) => c.deviceId === deviceId)?.label ?? "",
          position: position as CameraPosition,
          mmPerCss,
          screenChoice: screenChoice as number | "card",
          cardPx,
          dpr: window.devicePixelRatio,
        })}
      >Save</button>
    </div>
  );
}

function Home({ status, lighting, onLighting, upload, onRetry, onStart, onSetup }: {
  status: Status;
  lighting: string;
  onLighting: (id: string) => void;
  upload: { fraction: number; error?: string } | null;
  onRetry: () => void;
  onStart: () => void;
  onSetup: () => void;
}) {
  const uploading = !!upload && !upload.error;
  return (
    <div className="card">
      <h2>{status.today === 0 ? "No sessions yet today" : `Today: ${status.today} session${status.today === 1 ? "" : "s"} done, thank you!`}</h2>
      {upload && (
        <div className="upload">
          {upload.error ? (
            <>
              <p className="error">Upload didn't finish: {upload.error} Your session is safe in this browser.</p>
              <button onClick={onRetry}>Try the upload again</button>
            </>
          ) : (
            <>
              <p>Uploading your session: {Math.round(upload.fraction * 100)}%. Please keep this tab open.</p>
              <progress value={upload.fraction} max={1} />
            </>
          )}
        </div>
      )}
      <h3>What's your lighting right now?</h3>
      <p>Different lighting each time helps most.</p>
      {LIGHTINGS.map((l) => (
        <label key={l.id}>
          <input type="radio" name="lighting" checked={lighting === l.id} onChange={() => onLighting(l.id)} /> {l.label}
          {status.lastLighting === l.id && <span className="muted"> (you used this last time)</span>}
        </label>
      ))}
      <p className="muted">The session runs in full screen. Keep this tab in front until it finishes.</p>
      <button className="primary" disabled={!lighting || uploading} onClick={onStart}>Start session</button>
      <p className="muted">Your participant ID: <b>{status.alias}</b>. To have your data deleted at any time, give this ID to the person who shared this site with you.</p>
      <p><button className="link" onClick={onSetup}>Changed computer, camera or display? Set up again</button></p>
    </div>
  );
}

function Positioning({ video, frames, hfovDeg, onReady }: {
  video: React.RefObject<HTMLVideoElement>;
  frames: React.MutableRefObject<((f: VideoFrame) => void) | null>;
  hfovDeg: number;
  onReady: (posture: Posture, warnings: string[], expected: string) => void;
}) {
  const [items, setItems] = useState<Check[]>(checks(null));
  const [ready, setReady] = useState(0);
  const [countdown, setCountdown] = useState<number | null>(null);
  const [confirm, setConfirm] = useState<number | null>(null);
  const latest = useRef<{ posture: Posture | null; items: Check[] }>({ posture: null, items: checks(null) });

  useEffect(() => {
    const good: Posture[] = [];
    let goodSince: number | null = null;
    let countdownStart: number | null = null;
    frames.current = (frame) => {
      const v = video.current!;
      const posture = frame.landmarks
        ? measure(frame.landmarks, v.videoWidth, v.videoHeight, hfovDeg, frame.blink <= BLINK_THRESHOLD, measureBrightness(v, frame.landmarks))
        : null;
      const current = checks(posture);
      latest.current = { posture, items: current };
      setItems(current);
      if (posture && current.every((c) => c.ok)) {
        goodSince ??= frame.nowMs;
        good.push(posture);
      } else {
        goodSince = countdownStart = null;
        good.length = 0;
      }
      const fraction = goodSince === null ? 0 : Math.min(1, (frame.nowMs - goodSince) / (READY_SECONDS * 1000));
      setReady(fraction);
      if (fraction >= 1) countdownStart ??= frame.nowMs;
      if (countdownStart === null) {
        setCountdown(null);
        return;
      }
      const remaining = COUNTDOWN_SECONDS - (frame.nowMs - countdownStart) / 1000;
      if (remaining <= 0) {
        const median = (key: keyof Posture) => good.map((g) => g[key]).sort((a, b) => a - b)[good.length >> 1];
        const result = Object.fromEntries((Object.keys(good[0]) as (keyof Posture)[]).map((k) => [k, median(k)])) as unknown as Posture;
        frames.current = null;
        onReady(result, [], "best");
      } else setCountdown(Math.ceil(remaining));
    };
    return () => {
      frames.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const startAnyway = useCallback(() => {
    const { posture, items: current } = latest.current;
    if (!posture) return;
    if (current.every((c) => c.ok)) onReady(posture, [], "best");
    else setConfirm(CONFIRM_SECONDS);
  }, [onReady]);

  const proceed = useCallback(() => {
    const { posture, items: current } = latest.current;
    if (!posture) {
      setConfirm(null);
      return;
    }
    const issues = current.filter((c) => !c.ok);
    onReady(posture, issues.map((c) => c.label), rating(issues));
  }, [onReady]);

  useEffect(() => {
    if (confirm === null) return;
    if (confirm <= 0) {
      proceed();
      return;
    }
    const timer = setTimeout(() => setConfirm(confirm - 1), 1000);
    return () => clearTimeout(timer);
  }, [confirm, proceed]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.code === "Space") {
        e.preventDefault();
        if (confirm === null) startAnyway();
        else proceed();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [confirm, startAnyway, proceed]);

  const issues = items.filter((c) => !c.ok);
  const hasFace = items.length > 1;
  if (confirm !== null) {
    return (
      <div className="confirm">
        <h2>Starting with expected accuracy: {rating(issues).toUpperCase()}</h2>
        {issues.map((c) => (
          <p key={c.label}><b>{c.label}:</b> {c.impact}</p>
        ))}
        <p>Starting in {confirm} ...</p>
        <button className="primary" onClick={proceed}>Start now</button>{" "}
        <button onClick={() => setConfirm(null)}>Back to positioning</button>
      </div>
    );
  }
  return (
    <div className="positioning">
      <h2>Get into position</h2>
      <div className="oval-space"><div className={`oval ${issues.length ? "" : "good"}`} /></div>
      <ul className="checks">
        {items.map((c) => (
          <li key={c.label}>
            <span className={`light ${c.ok ? "green" : "red"}`} /> {c.label}
            {c.hint && <span className="hint"> {c.hint}</span>}
          </li>
        ))}
      </ul>
      <progress value={ready} max={1} />
      {countdown !== null ? (
        <p className="good-text">Good position - starting in {countdown}</p>
      ) : !hasFace ? (
        <p className="hint">Calibration needs your face in view.</p>
      ) : issues.length ? (
        <p>
          <button onClick={startAnyway}>Start anyway (SPACE)</button>{" "}
          <span className="hint">Expected accuracy: {rating(issues)} ({issues.length} issue{issues.length > 1 ? "s" : ""})</span>
        </p>
      ) : (
        <p className="good-text">Good - hold still</p>
      )}
    </div>
  );
}

function Calibrating({ video, frames, device, lighting, consentVersion, baseline, onFinished, onFailed }: {
  video: React.RefObject<HTMLVideoElement>;
  frames: React.MutableRefObject<((f: VideoFrame) => void) | null>;
  device: Device;
  lighting: string;
  consentVersion: string;
  baseline: { posture: Posture; warnings: string[]; expected: string };
  onFinished: (file: Blob) => void;
  onFailed: (message: string) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current!;
    const v = video.current!;
    const cssWidth = window.innerWidth;
    const cssHeight = window.innerHeight;
    const dpr = window.devicePixelRatio;
    const screenPx: [number, number] = [Math.round(cssWidth * dpr), Math.round(cssHeight * dpr)];
    const { mmPerCss } = device;
    const hfov = hfovFor(device.label);
    canvas.width = screenPx[0];
    canvas.height = screenPx[1];
    const runner = new CalibrationRunner(v, screenPx, hfov, baseline.posture);
    frames.current = (frame) => runner.onFrame(frame);
    let animation = requestAnimationFrame(function paint() {
      draw(canvas, runner.view(), dpr);
      animation = requestAnimationFrame(paint);
    });
    let cancelled = false;

    runner.run().then((recording) => {
      if (cancelled) return;
      const manifest: Manifest = {
        format: 1,
        created: new Date().toISOString(),
        lighting,
        consent_version: consentVersion,
        camera_name: device.label,
        camera_position: device.position,
        hfov_deg: hfov,
        camera_above_mm: CAMERA_ABOVE_MM[device.position],
        screen_px: screenPx,
        screen_mm: [cssWidth * mmPerCss, cssHeight * mmPerCss],
        screen_source: device.screenChoice === "card" ? "card" : `${device.screenChoice} inches`,
        device_pixel_ratio: dpr,
        frame_size: [v.videoWidth, v.videoHeight],
        posture: baseline.posture,
        warnings: baseline.warnings,
        expected_accuracy: baseline.expected,
        group_pass: recording.groupPass,
        duration_s: recording.durationS,
        frames: recording.frames,
      };
      if (recording.frames.length < 100) {
        onFailed("Too few usable frames were recorded (is your face in view and well lit?). Nothing was saved. Please try again.");
      } else onFinished(buildSessionFile(manifest, recording.landmarks, recording.crops));
    }).catch((error) => {
      if (!cancelled && !(error instanceof CalibrationAborted)) onFailed(`The session stopped: ${error}`);
    });

    return () => {
      cancelled = true;
      runner.abort();
      frames.current = null;
      cancelAnimationFrame(animation);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return <canvas ref={canvasRef} className="calibration" />;
}

function draw(canvas: HTMLCanvasElement, view: View, dpr: number) {
  const c = canvas.getContext("2d")!;
  const { width, height } = canvas;
  c.fillStyle = "#000";
  c.fillRect(0, 0, width, height);
  c.textAlign = "center";
  c.fillStyle = "#e6e6e6";
  if (view.title) {
    c.font = `${22 * dpr}px system-ui, sans-serif`;
    view.title.forEach((line, i) => c.fillText(line, width / 2, height / 3 + i * 38 * dpr));
  }
  if (view.dot) {
    const x = view.dot[0] * width;
    const y = view.dot[1] * height;
    const progress = Math.min(1, view.progress ?? 0);
    c.fillStyle = "#ffc83c";
    c.beginPath();
    c.arc(x, y, (28 - 18 * progress) * dpr * 0.8, 0, 2 * Math.PI);
    c.fill();
    c.fillStyle = "#000";
    c.beginPath();
    c.arc(x, y, 3.2 * dpr, 0, 2 * Math.PI);
    c.fill();
    if (view.ring) {
      c.strokeStyle = view.ring.done ? "#5adc5a" : "#fabe28";
      c.lineWidth = 3.2 * dpr;
      c.beginPath();
      c.arc(x, y, 32 * dpr, -Math.PI / 2, -Math.PI / 2 + 2 * Math.PI * view.ring.fraction);
      c.stroke();
    }
    if (view.dotText) {
      c.fillStyle = view.ring?.done ? "#5adc5a" : "#fabe28";
      c.font = `${18 * dpr}px system-ui, sans-serif`;
      const half = c.measureText(view.dotText).width / 2 + 10;
      const textX = Math.min(Math.max(x, half), width - half);
      c.fillText(view.dotText, textX, y < height - 110 * dpr ? y + 66 * dpr : y - 52 * dpr);
    }
  }
  if (view.faceLost) {
    c.fillStyle = "#e6e6e6";
    c.font = `${17 * dpr}px system-ui, sans-serif`;
    c.fillText("Face not found - look at the screen", width / 2, height - 48 * dpr);
  }
}
