/**
 * One recorded session as a single file, read by sync/sync_sessions.py:
 *   "GZC1" | u32 manifest byte length (little endian) | manifest JSON
 *   | Float32 landmarks (frames x 478 x 2, pixels, little endian) | the frames' JPEG crops, back to back
 * The file is cut into fixed-size parts for upload and joined again on the PC.
 */

import type { Posture } from "./posture";

export const LANDMARKS = 478;

export interface FrameRecord {
  /** Capture time, seconds from the start of the calibration. */
  t: number;
  /** Where the dot was, in physical screen pixels. */
  x: number;
  y: number;
  /** Target number (>= 0), or a moving-dot segment (< 0), as in the tracker's `groups`. */
  group: number;
  blink: number;
  crop: [number, number, number, number];
  jpeg_bytes: number;
}

export interface Manifest {
  format: 1;
  created: string;
  lighting: string;
  consent_version: string;
  camera_name: string;
  camera_position: string;
  hfov_deg: number;
  camera_above_mm: number;
  screen_px: [number, number];
  screen_mm: [number, number];
  /** "15.6 inches" (the participant's choice) or "card" (measured against a card). */
  screen_source: string;
  device_pixel_ratio: number;
  frame_size: [number, number];
  posture: Posture;
  warnings: string[];
  expected_accuracy: string;
  /** For each target group >= 0: 0 = head still, 1 = guided head movement. */
  group_pass: number[];
  duration_s: number;
  frames: FrameRecord[];
}

export function buildSessionFile(manifest: Manifest, landmarks: Float32Array, crops: Blob[]): Blob {
  const json = new TextEncoder().encode(JSON.stringify(manifest));
  const header = new ArrayBuffer(8);
  new Uint8Array(header).set(new TextEncoder().encode("GZC1"), 0);
  new DataView(header).setUint32(4, json.length, true);
  const landmarkBytes = new Uint8Array(landmarks.buffer, landmarks.byteOffset, landmarks.byteLength);
  return new Blob([header, json, landmarkBytes as BlobPart, ...crops], { type: "application/octet-stream" });
}

export async function sha256Hex(blob: Blob): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", await blob.arrayBuffer());
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}
