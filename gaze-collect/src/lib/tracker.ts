/** Webcam -> MediaPipe FaceLandmarker, as in talk-meter. Only the camera is requested, never audio. */

import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

import { MAX_FPS } from "./protocol";
import type { Point3 } from "./posture";

// MUST match the installed @mediapipe/tasks-vision version (package.json pins it exactly).
const WASM_BASE = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
// Used when the browser doesn't report a frame's capture time (same value as the tracker's pursuit lag).
const FALLBACK_LAG_MS = 80;

export interface VideoFrame {
  landmarks: Point3[] | null;
  blink: number;
  /** performance.now() clock. */
  captureMs: number;
  nowMs: number;
}

export async function openCamera(video: HTMLVideoElement, deviceId?: string): Promise<MediaStream> {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 1280 }, height: { ideal: 720 }, ...(deviceId ? { deviceId: { exact: deviceId } } : {}) },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
  return stream;
}

export async function listCameras(): Promise<MediaDeviceInfo[]> {
  return (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput");
}

export async function createLandmarker(): Promise<FaceLandmarker> {
  const fileset = await FilesetResolver.forVisionTasks(WASM_BASE);
  const options = (delegate: "GPU" | "CPU") => ({
    baseOptions: { modelAssetPath: MODEL_URL, delegate },
    runningMode: "VIDEO" as const,
    numFaces: 1,
    outputFaceBlendshapes: true,
  });
  try {
    return await FaceLandmarker.createFromOptions(fileset, options("GPU"));
  } catch {
    return await FaceLandmarker.createFromOptions(fileset, options("CPU"));
  }
}

type Bs = { categoryName?: string; score?: number };
const score = (bs: Bs[], name: string) => bs.find((c) => c.categoryName === name)?.score ?? 0;

/** Calls onFrame for new video frames, at most MAX_FPS. Returns a stop function. */
export function startFrames(video: HTMLVideoElement, landmarker: FaceLandmarker, onFrame: (f: VideoFrame) => void) {
  let stopped = false;
  let lastProcessed = -Infinity;
  let lastTimestamp = 0;
  const minimumGap = 1000 / MAX_FPS - 5;

  const step = (_now: number, metadata: VideoFrameCallbackMetadata) => {
    if (stopped) return;
    const nowMs = performance.now();
    if (nowMs - lastProcessed >= minimumGap && video.readyState >= 2) {
      lastProcessed = nowMs;
      lastTimestamp = Math.max(lastTimestamp + 1, Math.round(nowMs));
      const result = landmarker.detectForVideo(video, lastTimestamp);
      const landmarks = result.faceLandmarks?.[0];
      const bs = (result.faceBlendshapes?.[0]?.categories ?? []) as Bs[];
      onFrame({
        landmarks: landmarks?.length ? landmarks : null,
        blink: Math.max(score(bs, "eyeBlinkLeft"), score(bs, "eyeBlinkRight")),
        captureMs: metadata.captureTime ?? nowMs - FALLBACK_LAG_MS,
        nowMs,
      });
    }
    video.requestVideoFrameCallback(step);
  };
  video.requestVideoFrameCallback(step);
  return () => {
    stopped = true;
  };
}

/** Mean grey level inside and outside the face, from a small copy of the frame. */
export function measureBrightness(video: HTMLVideoElement, landmarks: Point3[]) {
  const width = 160;
  const height = Math.max(1, Math.round((width * video.videoHeight) / video.videoWidth));
  const canvas = new OffscreenCanvas(width, height);
  const context = canvas.getContext("2d", { willReadFrequently: true })!;
  context.drawImage(video, 0, 0, width, height);
  const { data } = context.getImageData(0, 0, width, height);
  const xs = landmarks.map((p) => p.x * width);
  const ys = landmarks.map((p) => p.y * height);
  const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  let face = 0, faceCount = 0, background = 0, backgroundCount = 0;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 4;
      const grey = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
      if (x >= x0 && x <= x1 && y >= y0 && y <= y1) {
        face += grey;
        faceCount++;
      } else {
        background += grey;
        backgroundCount++;
      }
    }
  }
  const faceMean = faceCount ? face / faceCount : 0;
  return { face: faceMean, background: backgroundCount ? background / backgroundCount : faceMean };
}
