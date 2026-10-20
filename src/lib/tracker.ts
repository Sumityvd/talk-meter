/**
 * Webcam → MediaPipe FaceLandmarker → one FrameSample per video frame.
 *
 * Runs on the main thread. Browsers throttle hidden tabs, so keep the tab
 * visible while recording. Only the camera is requested — never audio.
 */

import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

import type { FrameSample } from "./recorder";
import { mouthAspectRatio } from "./speech";

// MUST match the installed @mediapipe/tasks-vision version (package.json pins
// it exactly): a mismatched WASM runtime refuses to boot.
const WASM_BASE = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

export async function openCamera(video: HTMLVideoElement): Promise<MediaStream> {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
  return stream;
}

export function closeCamera(video: HTMLVideoElement, stream: MediaStream | null): void {
  stream?.getTracks().forEach((t) => t.stop());
  video.srcObject = null;
}

export async function createLandmarker(): Promise<FaceLandmarker> {
  const fileset = await FilesetResolver.forVisionTasks(WASM_BASE);
  const options = (delegate: "GPU" | "CPU") => ({
    baseOptions: { modelAssetPath: MODEL_URL, delegate },
    runningMode: "VIDEO" as const,
    numFaces: 1,
    outputFaceBlendshapes: true,
    minFaceDetectionConfidence: 0.5,
    minFacePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
  try {
    return await FaceLandmarker.createFromOptions(fileset, options("GPU"));
  } catch {
    return await FaceLandmarker.createFromOptions(fileset, options("CPU"));
  }
}

type Bs = { categoryName?: string; score?: number };
const score = (bs: Bs[], name: string) => bs.find((c) => c.categoryName === name)?.score ?? 0;

/** Start the per-frame loop. Returns a stop function. */
export function startLoop(
  video: HTMLVideoElement,
  landmarker: FaceLandmarker,
  onFrame: (f: FrameSample) => void,
): () => void {
  let stopped = false;
  let lastVideoTime = -1;
  const rvfc = typeof video.requestVideoFrameCallback === "function";

  const schedule = () => {
    if (stopped) return;
    if (rvfc) video.requestVideoFrameCallback(step);
    else requestAnimationFrame(step);
  };

  const step = () => {
    if (stopped) return;
    if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
      lastVideoTime = video.currentTime;
      const now = performance.now();
      const res = landmarker.detectForVideo(video, now);
      const lms = res.faceLandmarks?.[0];
      if (lms && lms.length) {
        const bs = (res.faceBlendshapes?.[0]?.categories ?? []) as Bs[];
        onFrame({
          t: now / 1000,
          face: true,
          mar: mouthAspectRatio(lms),
          smile: 0.5 * (score(bs, "mouthSmileLeft") + score(bs, "mouthSmileRight")),
        });
      } else {
        onFrame({ t: now / 1000, face: false, mar: 0, smile: 0 });
      }
    }
    schedule();
  };

  schedule();
  return () => {
    stopped = true;
  };
}
