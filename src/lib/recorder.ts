/**
 * Turns the per-frame stream into per-second and per-minute records.
 *
 * Frames are binned by whole seconds of session time. Each second keeps its
 * raw mouth statistics and the activity label the user had selected, so a
 * labelled recording can be used to re-tune the thresholds offline. A minute
 * closes when the first second of the next minute arrives, or on finish().
 */

import { classifySecond, summarizeSeconds, type SecondStats } from "./speech";

export const ACTIVITIES = [
  "reading",
  "typing",
  "listening",
  "speaking",
  "laughing",
  "chewing",
  "yawning",
  "other",
] as const;
export type Activity = (typeof ACTIVITIES)[number];

export type SecondState = "talking" | "smiling" | "quiet" | "no-face" | "sparse";

export interface FrameSample {
  t: number;      // seconds (monotonic clock)
  face: boolean;
  mar: number;
  smile: number;
}

export interface SecondRecord {
  second: number;        // index from session start
  label: Activity;       // majority label over the second's frames
  frames: number;        // all frames, with or without a face
  faceFrames: number;
  state: SecondState;
  stats: SecondStats | null;
}

export interface MinuteRecord {
  minute: number;
  startSecond: number;
  endSecond: number;     // inclusive
  label: Activity;       // majority label over its seconds
  talk_frac: number;
  smile_frac: number;
  quiet_s: number;
  seconds: number;       // measured seconds
  faceRatio: number;
  fps: number;
}

// A trailing partial minute shorter than this is dropped on finish().
const MIN_PARTIAL_MINUTE_S = 10;

export class SessionRecorder {
  readonly seconds: SecondRecord[] = [];
  readonly minutes: MinuteRecord[] = [];

  private t0: number | null = null;
  private curSecond = -1;
  private buf: Array<{ face: boolean; mar: number; smile: number; label: Activity }> = [];
  private curMinute: number | null = null;
  private minuteSeconds: SecondRecord[] = [];

  /** Add one frame. Returns the second it closed, if this frame started a new one. */
  addFrame(f: FrameSample, label: Activity): SecondRecord | null {
    if (this.t0 === null) this.t0 = f.t;
    const sec = Math.floor(f.t - this.t0);
    let closed: SecondRecord | null = null;
    if (sec !== this.curSecond) {
      closed = this.closeSecond();
      this.curSecond = sec;
    }
    this.buf.push({ face: f.face, mar: f.mar, smile: f.smile, label });
    return closed;
  }

  /** Close the open second and minute. Call when recording stops. */
  finish(): void {
    this.closeSecond();
    if (this.minuteSeconds.length >= MIN_PARTIAL_MINUTE_S) this.closeMinute();
    this.minuteSeconds = [];
    this.curMinute = null;
  }

  private closeSecond(): SecondRecord | null {
    if (!this.buf.length) return null;
    const faceFrames = this.buf.filter((b) => b.face);
    const stats = classifySecond(faceFrames);
    const state: SecondState = !faceFrames.length
      ? "no-face"
      : !stats
        ? "sparse"
        : stats.talking
          ? "talking"
          : stats.smiling
            ? "smiling"
            : "quiet";
    const rec: SecondRecord = {
      second: this.curSecond,
      label: majority(this.buf.map((b) => b.label)),
      frames: this.buf.length,
      faceFrames: faceFrames.length,
      state,
      stats,
    };
    this.buf = [];

    const minute = Math.floor(rec.second / 60);
    if (this.curMinute !== null && minute !== this.curMinute) this.closeMinute();
    this.curMinute = minute;
    this.minuteSeconds.push(rec);
    this.seconds.push(rec);
    return rec;
  }

  private closeMinute(): void {
    const secs = this.minuteSeconds;
    this.minuteSeconds = [];
    if (!secs.length || this.curMinute === null) return;
    const speech = summarizeSeconds(secs.map((s) => s.stats));
    const frames = secs.reduce((a, s) => a + s.frames, 0);
    const faceFrames = secs.reduce((a, s) => a + s.faceFrames, 0);
    this.minutes.push({
      minute: this.curMinute,
      startSecond: secs[0].second,
      endSecond: secs[secs.length - 1].second,
      label: majority(secs.map((s) => s.label)),
      ...speech,
      faceRatio: frames ? Math.round((faceFrames / frames) * 1000) / 1000 : 0,
      fps: Math.round((frames / secs.length) * 10) / 10,
    });
  }
}

function majority<T>(xs: T[]): T {
  const counts = new Map<T, number>();
  let best = xs[0];
  let bestN = 0;
  for (const x of xs) {
    const n = (counts.get(x) ?? 0) + 1;
    counts.set(x, n);
    if (n > bestN) {
      best = x;
      bestN = n;
    }
  }
  return best;
}
