/**
 * Talking / smiling / quiet, measured from the face alone — no audio.
 *
 * Speech opens and closes the mouth at 2–7 Hz (about five syllables a
 * second); a yawn is one slow gape of ~6 s. So a talking second shows the
 * mouth-aspect ratio (MAR) swinging back and forth, while a yawning second
 * sits wide open. Each second is classified on its own; a minute is the share
 * of measured seconds in each state.
 *
 * The thresholds are starting values: the per-second CSV exports every raw
 * statistic so they can be re-tuned offline from labelled recordings.
 */

export interface Point {
  x: number;
  y: number;
}

// MediaPipe FaceMesh landmark indices for the mouth.
export const MAR_IDX = {
  upperInner: 13,
  lowerInner: 14,
  leftCorner: 78,
  rightCorner: 308,
  upperOuter: 0,
  lowerOuter: 17,
} as const;

export const THRESHOLDS = {
  minFramesPerSecond: 3,  // below this a second can't show motion
  marRange: 0.08,         // min peak-to-trough MAR within the second
  minCrossings: 2,        // ≥ one open-close cycle around the median
  yawnMar: 0.6,           // "wide open" bar...
  yawnFrac: 0.5,          // ...held for half the second = yawning, not talking
  smileScore: 0.5,        // mean mouthSmileLeft/Right blendshape
} as const;

const dist = (a: Point, b: Point) => Math.hypot(a.x - b.x, a.y - b.y);

/** Mouth openness relative to mouth width; ~0.1 closed, >0.6 yawning. */
export function mouthAspectRatio(lm: ReadonlyArray<Point>): number {
  const ui = lm[MAR_IDX.upperInner];
  const li = lm[MAR_IDX.lowerInner];
  const lc = lm[MAR_IDX.leftCorner];
  const rc = lm[MAR_IDX.rightCorner];
  const uo = lm[MAR_IDX.upperOuter];
  const lo = lm[MAR_IDX.lowerOuter];
  if (!ui || !li || !lc || !rc || !uo || !lo) return 0;
  const h = dist(lc, rc);
  return h === 0 ? 0 : (dist(ui, li) + dist(uo, lo)) / (2 * h);
}

/** Raw statistics of one second of face frames, plus its classification. */
export interface SecondStats {
  frames: number;     // face frames in the second
  marMin: number;
  marMax: number;
  marMedian: number;
  marRange: number;
  crossings: number;  // sign changes of (MAR − median)
  wideFrac: number;   // share of frames with MAR above the yawn bar
  smileMean: number;
  talking: boolean;
  smiling: boolean;
}

/**
 * Classify one second from its face frames (in time order). Returns null when
 * there are too few frames to see mouth motion at all.
 */
export function classifySecond(
  frames: ReadonlyArray<{ mar: number; smile: number }>,
): SecondStats | null {
  const n = frames.length;
  if (n < THRESHOLDS.minFramesPerSecond) return null;

  const mars = frames.map((f) => f.mar);
  let smileSum = 0;
  let wide = 0;
  for (const f of frames) {
    smileSum += f.smile;
    if (f.mar > THRESHOLDS.yawnMar) wide++;
  }
  const marMedian = median(mars);
  let crossings = 0;
  let prevSign = 0;
  for (const m of mars) {
    const sign = Math.sign(m - marMedian);
    if (sign === 0) continue;
    if (prevSign !== 0 && sign !== prevSign) crossings++;
    prevSign = sign;
  }
  const marMin = Math.min(...mars);
  const marMax = Math.max(...mars);
  const marRange = marMax - marMin;
  const wideFrac = wide / n;
  const smileMean = smileSum / n;

  return {
    frames: n,
    marMin,
    marMax,
    marMedian,
    marRange,
    crossings,
    wideFrac,
    smileMean,
    talking:
      wideFrac < THRESHOLDS.yawnFrac &&
      marRange >= THRESHOLDS.marRange &&
      crossings >= THRESHOLDS.minCrossings,
    smiling: smileMean >= THRESHOLDS.smileScore,
  };
}

/** Per-window summary: shares of measured seconds in each state. */
export interface WindowSpeech {
  talk_frac: number;  // share of measured seconds with speech-like motion
  smile_frac: number; // share of measured seconds spent smiling
  quiet_s: number;    // measured seconds that were neither
  seconds: number;    // seconds measured (each needs enough face frames)
}

export function summarizeSeconds(
  seconds: ReadonlyArray<SecondStats | null>,
): WindowSpeech {
  let measured = 0;
  let talking = 0;
  let smiling = 0;
  let quiet = 0;
  for (const s of seconds) {
    if (!s) continue;
    measured++;
    if (s.talking) talking++;
    if (s.smiling) smiling++;
    if (!s.talking && !s.smiling) quiet++;
  }
  const frac = (x: number) => (measured ? Math.round((x / measured) * 1000) / 1000 : 0);
  return {
    talk_frac: frac(talking),
    smile_frac: frac(smiling),
    quiet_s: quiet,
    seconds: measured,
  };
}

/**
 * Summarise a run of face frames (time-ordered, `t` in seconds) by binning
 * them into whole seconds.
 */
export function measureWindowSpeech(
  frames: ReadonlyArray<{ t: number; mar: number; smile: number }>,
): WindowSpeech {
  const stats: Array<SecondStats | null> = [];
  let i = 0;
  while (i < frames.length) {
    const bin = Math.floor(frames[i].t);
    let j = i;
    while (j < frames.length && Math.floor(frames[j].t) === bin) j++;
    stats.push(classifySecond(frames.slice(i, j)));
    i = j;
  }
  return summarizeSeconds(stats);
}

function median(a: number[]): number {
  if (!a.length) return 0;
  const s = [...a].sort((x, y) => x - y);
  const mid = s.length >> 1;
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}
