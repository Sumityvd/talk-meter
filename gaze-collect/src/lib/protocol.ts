/**
 * The calibration, matching `webcam_gaze.py --calibration quick` (constants of the same name there), so web
 * sessions are recorded like the tracker's own. All positions here are fractions of the screen.
 */

/** The version of the consent text in App.tsx. Change it with the text; the server must have the same value. */
export const CONSENT_VERSION = "2026-11-09";

export const MOVE_SECONDS = 0.45;
export const SETTLE_SECONDS = 0.55;
export const COLLECT_SECONDS = 1.3;
export const CALIBRATION_MARGIN = 0.06;
export const DOT_GRID: [number, number] = [4, 4];
export const HEAD_GRID: [number, number] = [4, 3];
export const HEAD_MARGIN = 0.12;
export const HEAD_DOTS = 6;
export const HEAD_MOVE_TIMEOUT_SECONDS = 8;
export const PURSUIT_SECONDS = 20;
export const PURSUIT_LEAD_SECONDS = 1.5;
export const PURSUIT_SKIP_SECONDS = 0.3;
export const PURSUIT_PERIODS: [number, number] = [13, 9];
export const BLINK_THRESHOLD = 0.45;
export const MAX_FPS = 15;
export const MIN_SAMPLES = 8;
export const HEAD_MOVE_FRAMES = Math.round(COLLECT_SECONDS * MAX_FPS);

export const CARD_WIDTH_MM = 85.6;
export const SCREEN_INCHES = [13.3, 14, 15.6, 16, 17.3, 21.5, 24, 27, 32];
// Outside this, the measurement is wrong rather than the screen unusual. Labels depend directly on screen size:
// a misjudged card (a "9 inch" laptop) nearly doubled the error on 2026-11-05.
export const PLAUSIBLE_INCHES: [number, number] = [11, 45];

const MM_PER_INCH = 25.4;

/** Millimetres per CSS pixel for a screen of this diagonal. */
export const mmPerCssFromInches = (inches: number, cssWidth: number, cssHeight: number) =>
  (inches * MM_PER_INCH) / Math.hypot(cssWidth, cssHeight);

export const diagonalInches = (mmPerCss: number, cssWidth: number, cssHeight: number) =>
  (mmPerCss * Math.hypot(cssWidth, cssHeight)) / MM_PER_INCH;
export const CAMERA_ABOVE_MM = { "built-in": 8, "clip-on": 15 } as const;
export type CameraPosition = keyof typeof CAMERA_ABOVE_MM;
export const DEFAULT_HFOV_DEG = 65;
export const KNOWN_HFOV_DEG: Record<string, number> = { "lifecam hd-6000": 59 };

export const LIGHTINGS = [
  { id: "daylight", label: "Daylight from a window" },
  { id: "lamp-front", label: "Lamp in front of me" },
  { id: "ceiling", label: "Ceiling light" },
  { id: "dim", label: "Dim room" },
] as const;

export const HEAD_MOVES = [
  { text: "Turn your head to the LEFT", kind: "turn", amount: 8 },
  { text: "Lean CLOSER to the screen", kind: "closer", amount: 5 },
  { text: "Tilt your chin UP", kind: "tilt", amount: 6 },
  { text: "Shift your head to the RIGHT (don't turn it)", kind: "shift", amount: 4 },
  { text: "Turn your head to the RIGHT", kind: "turn", amount: 8 },
  { text: "Lean BACK from the screen", kind: "back", amount: 5 },
  { text: "Tilt your chin DOWN", kind: "tilt", amount: 6 },
  { text: "Shift your head to the LEFT (don't turn it)", kind: "shift", amount: 4 },
] as const;
export type HeadMove = (typeof HEAD_MOVES)[number];

/** Grid of screen fractions in a fixed random order, so dots jump around rather than sweep. */
export function gridTargets(columns: number, rows: number, margin: number, seed: number): [number, number][] {
  const axis = (n: number) => Array.from({ length: n }, (_, i) => margin + ((1 - 2 * margin) * i) / (n - 1));
  const targets: [number, number][] = [];
  for (const y of axis(rows)) for (const x of axis(columns)) targets.push([x, y]);
  let state = seed;
  const random = () => {
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  for (let i = targets.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [targets[i], targets[j]] = [targets[j], targets[i]];
  }
  return targets;
}

export function pursuitPosition(t: number): [number, number] {
  return [
    0.5 + 0.42 * Math.sin((2 * Math.PI * t) / PURSUIT_PERIODS[0]),
    0.5 + 0.4 * Math.sin((2 * Math.PI * t) / PURSUIT_PERIODS[1] + Math.PI / 2),
  ];
}

export function hfovFor(cameraLabel: string): number {
  const name = cameraLabel.toLowerCase();
  const known = Object.keys(KNOWN_HFOV_DEG).find((key) => name.includes(key));
  return known ? KNOWN_HFOV_DEG[known] : DEFAULT_HFOV_DEG;
}
