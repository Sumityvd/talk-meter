/**
 * The positioning checks, ported from positioning.py. Only guidance for the participant: the real head pose used
 * for training is computed later by the Python pipeline. Head angles come from the landmarks' depth values,
 * because the browser has no solvePnP.
 */

import type { HeadMove } from "./protocol";

export interface Point3 {
  x: number;
  y: number;
  z: number;
}

export interface Posture {
  distance_cm: number;
  centre_x: number;
  centre_y: number;
  yaw: number;
  pitch: number;
  roll: number;
  interpupil_px: number;
  face_brightness: number;
  background_brightness: number;
  eyes_open: number;
}

export interface Check {
  label: string;
  ok: boolean;
  hint: string;
  impact: string;
  severity: number;
}

const INTERPUPIL_MM = 63;
const DISTANCE_CM: [number, number] = [45, 75];
const REFERENCE_DISTANCE_CM = 55;
const CENTRE_X_TOLERANCE = 0.12;
const FACE_Y_RANGE: [number, number] = [0.28, 0.62];
const YAW_LIMIT = 12;
const ROLL_LIMIT = 8;
const FACE_BRIGHTNESS: [number, number] = [60, 210];
const BACKLIGHT_MARGIN = 45;
const BACKLIT_FACE_MAX = 75;
const MIN_INTERPUPIL_PX = 40;
const degrees = (radians: number) => (radians * 180) / Math.PI;

export const focalPx = (frameWidth: number, hfovDeg: number) =>
  frameWidth / 2 / Math.tan((hfovDeg * Math.PI) / 360);

const mean = (points: Point3[]) => {
  const sum = points.reduce((s, p) => ({ x: s.x + p.x, y: s.y + p.y, z: s.z + p.z }), { x: 0, y: 0, z: 0 });
  return { x: sum.x / points.length, y: sum.y / points.length, z: sum.z / points.length };
};

/** Face crop for upload: square around the face, wide enough to contain the training patch at any head pose. */
export function cropBox(landmarks: Point3[], frameWidth: number, frameHeight: number) {
  const left = mean(landmarks.slice(468, 473));
  const right = mean(landmarks.slice(473, 478));
  const ipd = Math.hypot((left.x - right.x) * frameWidth, (left.y - right.y) * frameHeight);
  const centre = mean(landmarks);
  const side = Math.min(Math.ceil(ipd * 3.4), frameWidth, frameHeight);
  const x = Math.round(Math.min(Math.max(centre.x * frameWidth - side / 2, 0), frameWidth - side));
  const y = Math.round(Math.min(Math.max(centre.y * frameHeight - side / 2, 0), frameHeight - side));
  return { x, y, width: side, height: side };
}

export function measure(
  landmarks: Point3[],
  frameWidth: number,
  frameHeight: number,
  hfovDeg: number,
  eyesOpen: boolean,
  brightness: { face: number; background: number },
): Posture {
  const left = mean(landmarks.slice(468, 473));
  const right = mean(landmarks.slice(473, 478));
  const ipd = Math.hypot((left.x - right.x) * frameWidth, (left.y - right.y) * frameHeight);
  const centre = mean(landmarks);
  const a = landmarks[33]; // eye corners, image left and right
  const b = landmarks[263];
  const top = landmarks[10]; // forehead, chin
  const bottom = landmarks[152];
  const ratio = frameHeight / frameWidth;
  return {
    distance_cm: (focalPx(frameWidth, hfovDeg) * INTERPUPIL_MM) / Math.max(ipd, 1) / 10,
    centre_x: centre.x,
    centre_y: centre.y,
    yaw: degrees(Math.atan2(b.z - a.z, b.x - a.x)),
    pitch: degrees(Math.atan2(bottom.z - top.z, (bottom.y - top.y) * ratio)),
    roll: degrees(Math.atan2((b.y - a.y) * ratio, b.x - a.x)),
    interpupil_px: ipd,
    face_brightness: brightness.face,
    background_brightness: brightness.background,
    eyes_open: eyesOpen ? 1 : 0,
  };
}

export function checks(m: Posture | null): Check[] {
  if (!m) {
    return [{
      label: "Face",
      ok: false,
      hint: "No face found - sit in front of the camera and look at the screen",
      impact: "No face: calibration cannot run",
      severity: 3,
    }];
  }
  const result: Check[] = [{ label: "Face", ok: true, hint: "", impact: "", severity: 0 }];
  const add = (label: string, ok: boolean, hint: string, impact: string, severity: number) =>
    result.push({ label, ok, hint: ok ? "" : hint, impact: ok ? "" : impact, severity: ok ? 0 : severity });

  const d = m.distance_cm;
  const ratio = d / REFERENCE_DISTANCE_CM;
  add(
    `Distance ${d.toFixed(0)} cm`,
    d >= DISTANCE_CM[0] && d <= DISTANCE_CM[1],
    `${d > DISTANCE_CM[1] ? "Move closer" : "Move back"} (now ${d.toFixed(0)} cm, best ${DISTANCE_CM[0]}-${DISTANCE_CM[1]} cm)`,
    ratio > 1
      ? `Your eyes look smaller to the camera: roughly ${ratio.toFixed(1)}x the error and more jitter`
      : "Too close: your head can leave the camera view during the head-movement part",
    ratio > 1.4 ? 2 : 1,
  );
  // The camera image isn't mirrored: a face right of centre means the person sits too far to their left.
  const dx = m.centre_x - 0.5;
  add(
    "Centred left/right",
    Math.abs(dx) <= CENTRE_X_TOLERANCE,
    dx > 0 ? "Move a little to your right" : "Move a little to your left",
    "Your head may leave the camera view when it moves",
    1,
  );
  add(
    "Camera aim",
    m.centre_y >= FACE_Y_RANGE[0] && m.centre_y <= FACE_Y_RANGE[1],
    m.centre_y < FACE_Y_RANGE[0] ? "Tilt the camera up a little" : "Tilt the camera down a little",
    "The camera sees your eyes at a steep angle: less accurate, mostly up/down",
    1,
  );
  add(
    "Head straight",
    Math.abs(m.yaw) <= YAW_LIMIT && Math.abs(m.roll) <= ROLL_LIMIT,
    Math.abs(m.yaw) > YAW_LIMIT ? "Face the screen straight on" : "Keep your head level (don't tilt it sideways)",
    "A turned or tilted head hides part of one eye: less accurate, mostly left/right",
    1,
  );
  const face = m.face_brightness;
  const backlit = m.background_brightness - face > BACKLIGHT_MARGIN && face < BACKLIT_FACE_MAX;
  const tooBright = face > FACE_BRIGHTNESS[1];
  add(
    "Lighting",
    !backlit && !tooBright && face >= FACE_BRIGHTNESS[0],
    backlit
      ? "Bright light behind you - face a lamp or window instead"
      : tooBright
        ? "Too much light on your face - dim it or move away from direct light"
        : "Your face is too dark - add light in front of you",
    tooBright ? "Washed-out eyes: less accurate" : "Dark, noisy eye detail: less accurate",
    tooBright ? 1 : 2,
  );
  const open = m.eyes_open > 0.5;
  add(
    "Eyes visible",
    open && m.interpupil_px >= MIN_INTERPUPIL_PX,
    open ? "Your eyes are too small in the image - move closer" : "Open your eyes and look at the screen",
    open ? "Eyes too small in the image: much less accurate" : "Calibration needs open eyes on each dot",
    2,
  );
  return result;
}

export function rating(items: Check[]): string {
  const total = items.reduce((sum, item) => sum + item.severity, 0);
  return total === 0 ? "best" : total === 1 ? "slightly reduced" : total <= 3 ? "reduced" : "much reduced";
}

/** How far (0..1+) the head has moved in the asked way, against the position at the start. */
export function headMoveProgress(move: HeadMove, m: Posture | null, baseline: Posture, frameWidth: number, hfovDeg: number) {
  if (!m) return 0;
  const wrap = (angle: number) => ((angle + 540) % 360) - 180;
  let change: number;
  switch (move.kind) {
    case "turn":
      change = Math.abs(wrap(m.yaw - baseline.yaw));
      break;
    case "tilt":
      change = Math.abs(wrap(m.pitch - baseline.pitch));
      break;
    case "closer":
      change = baseline.distance_cm - m.distance_cm;
      break;
    case "back":
      change = m.distance_cm - baseline.distance_cm;
      break;
    default: // sideways shift, cm at the face: image offset * distance / focal length
      change = (Math.abs(m.centre_x - baseline.centre_x) * frameWidth * m.distance_cm) / focalPx(frameWidth, hfovDeg);
  }
  return Math.max(0, change / move.amount);
}
