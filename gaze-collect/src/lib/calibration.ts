/**
 * Runs one calibration: still dots, guided head moves, moving dot. Mirrors collect_targets, collect_head_moves and
 * collect_pursuit in webcam_gaze.py. The page draws what `view()` returns every animation frame.
 */

import { cropBox, headMoveProgress, measure, type Point3, type Posture } from "./posture";
import {
  BLINK_THRESHOLD, COLLECT_SECONDS, DOT_GRID, HEAD_DOTS, HEAD_GRID, HEAD_MARGIN, HEAD_MOVES, HEAD_MOVE_FRAMES,
  HEAD_MOVE_TIMEOUT_SECONDS, CALIBRATION_MARGIN, MIN_SAMPLES, MOVE_SECONDS, PURSUIT_LEAD_SECONDS, PURSUIT_SECONDS,
  PURSUIT_SKIP_SECONDS, SETTLE_SECONDS, gridTargets, pursuitPosition,
} from "./protocol";
import { LANDMARKS, type FrameRecord } from "./sessionFile";
import type { VideoFrame } from "./tracker";

export interface View {
  /** Screen fractions. */
  dot?: [number, number];
  /** 0..1, the dot shrinks as it fills. */
  progress?: number;
  /** Ring around the dot (guided head moves). */
  ring?: { fraction: number; done: boolean };
  dotText?: string;
  title?: string[];
  faceLost?: boolean;
}

export interface Recording {
  frames: FrameRecord[];
  landmarks: Float32Array;
  crops: Blob[];
  groupPass: number[];
  durationS: number;
}

interface Pending {
  record: Omit<FrameRecord, "jpeg_bytes">;
  points: Float32Array;
  crop: Promise<Blob>;
}

export class CalibrationAborted extends Error {}

const smooth = (k: number) => k * k * (3 - 2 * k);
const lerp = (a: [number, number], b: [number, number], k: number): [number, number] => [a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k];

export class CalibrationRunner {
  view: () => View = () => ({});
  private handler: ((f: VideoFrame) => void) | null = null;
  private pending: Pending[] = [];
  private groupPass: number[] = [];
  private aborted = false;
  private faceLost = false;
  private startMs = 0;

  constructor(
    private video: HTMLVideoElement,
    private screenPx: [number, number],
    private hfovDeg: number,
    private baseline: Posture,
  ) {}

  onFrame(frame: VideoFrame) {
    this.faceLost = !frame.landmarks;
    this.handler?.(frame);
  }

  abort() {
    this.aborted = true;
  }

  private sleep(seconds: number): Promise<void> {
    return new Promise((resolve, reject) => {
      const end = performance.now() + seconds * 1000;
      const tick = () => {
        if (this.aborted) reject(new CalibrationAborted());
        else if (performance.now() >= end) resolve();
        else setTimeout(tick, 20);
      };
      tick();
    });
  }

  private capture(frame: VideoFrame, target: [number, number], group: number): Pending {
    const landmarks = frame.landmarks as Point3[];
    const frameWidth = this.video.videoWidth;
    const frameHeight = this.video.videoHeight;
    const box = cropBox(landmarks, frameWidth, frameHeight);
    const canvas = new OffscreenCanvas(box.width, box.height);
    canvas.getContext("2d")!.drawImage(this.video, box.x, box.y, box.width, box.height, 0, 0, box.width, box.height);
    const points = new Float32Array(LANDMARKS * 2);
    for (let i = 0; i < LANDMARKS; i++) {
      points[2 * i] = landmarks[i].x * frameWidth;
      points[2 * i + 1] = landmarks[i].y * frameHeight;
    }
    return {
      record: {
        t: (frame.captureMs - this.startMs) / 1000,
        x: target[0] * this.screenPx[0],
        y: target[1] * this.screenPx[1],
        group,
        blink: frame.blink,
        crop: [box.x, box.y, box.width, box.height],
      },
      points,
      crop: canvas.convertToBlob({ type: "image/jpeg", quality: 0.92 }),
    };
  }

  private usable = (frame: VideoFrame) => !!frame.landmarks && frame.blink <= BLINK_THRESHOLD;

  private async announce(lines: string[], seconds: number) {
    const end = performance.now() + seconds * 1000;
    this.view = () => ({ title: [...lines, "", `Starting in ${Math.ceil((end - performance.now()) / 1000)} ...`] });
    this.handler = null;
    await this.sleep(seconds);
  }

  private async stillDots() {
    const queue = gridTargets(DOT_GRID[0], DOT_GRID[1], CALIBRATION_MARGIN, 7);
    const retries = new Map<[number, number], number>();
    let previous: [number, number] = [0.5, 0.5];
    while (queue.length) {
      const target = queue.shift()!;
      const group = this.groupPass.length;
      const samples: Pending[] = [];
      const start = performance.now();
      const total = MOVE_SECONDS + SETTLE_SECONDS + COLLECT_SECONDS;
      const from = previous;
      this.view = () => {
        const elapsed = (performance.now() - start) / 1000;
        return elapsed < MOVE_SECONDS
          ? { dot: lerp(from, target, smooth(elapsed / MOVE_SECONDS)), progress: 0, faceLost: this.faceLost }
          : { dot: target, progress: (elapsed - MOVE_SECONDS) / (total - MOVE_SECONDS), faceLost: this.faceLost };
      };
      this.handler = (frame) => {
        if ((frame.nowMs - start) / 1000 >= MOVE_SECONDS + SETTLE_SECONDS && this.usable(frame)) {
          samples.push(this.capture(frame, target, group));
        }
      };
      await this.sleep(total);
      previous = target;
      if (samples.length >= MIN_SAMPLES) {
        this.pending.push(...samples);
        this.groupPass.push(0);
      } else if ((retries.get(target) ?? 0) < 2) {
        retries.set(target, (retries.get(target) ?? 0) + 1);
        queue.push(target);
      }
    }
  }

  private async headMoves() {
    const targets = gridTargets(HEAD_GRID[0], HEAD_GRID[1], HEAD_MARGIN, 11).slice(0, HEAD_DOTS);
    let previous: [number, number] = [0.5, 0.5];
    for (const [number, target] of targets.entries()) {
      const move = HEAD_MOVES[number % HEAD_MOVES.length];
      const group = this.groupPass.length;
      const samples: Pending[] = [];
      const start = performance.now();
      const from = previous;
      let fraction = 0;
      this.view = () => {
        const elapsed = (performance.now() - start) / 1000;
        const dot = elapsed < MOVE_SECONDS ? lerp(from, target, smooth(elapsed / MOVE_SECONDS)) : target;
        const done = fraction >= 1;
        return {
          dot,
          progress: samples.length / HEAD_MOVE_FRAMES,
          ring: { fraction: Math.min(1, fraction), done },
          dotText: done ? "Hold it - keep looking at the dot" : move.text,
          faceLost: this.faceLost,
        };
      };
      await new Promise<void>((resolve, reject) => {
        this.handler = (frame) => {
          const elapsed = (frame.nowMs - start) / 1000;
          const posture = frame.landmarks
            ? measure(frame.landmarks, this.video.videoWidth, this.video.videoHeight, this.hfovDeg, frame.blink <= BLINK_THRESHOLD, { face: 0, background: 0 })
            : null;
          fraction = headMoveProgress(move, posture, this.baseline, this.video.videoWidth, this.hfovDeg);
          if (elapsed >= MOVE_SECONDS && fraction >= 1 && this.usable(frame)) samples.push(this.capture(frame, target, group));
          if (samples.length >= HEAD_MOVE_FRAMES || elapsed > MOVE_SECONDS + HEAD_MOVE_TIMEOUT_SECONDS) resolve();
        };
        const watch = () => {
          if (this.aborted) reject(new CalibrationAborted());
          else if ((performance.now() - start) / 1000 > MOVE_SECONDS + HEAD_MOVE_TIMEOUT_SECONDS + 1) resolve();
          else setTimeout(watch, 100);
        };
        watch();
      });
      previous = target;
      if (samples.length >= MIN_SAMPLES) {
        this.pending.push(...samples);
        this.groupPass.push(1);
      }
    }
  }

  private async pursuit() {
    const start = performance.now();
    const clock = (now: number) => (now - start) / 1000 - PURSUIT_LEAD_SECONDS;
    this.view = () => ({ dot: pursuitPosition(Math.max(0, clock(performance.now()))), progress: 0.5, faceLost: this.faceLost });
    this.handler = (frame) => {
      const t = clock(frame.captureMs); // where the dot was when this frame was exposed
      if (t > PURSUIT_SKIP_SECONDS && this.usable(frame)) {
        this.pending.push(this.capture(frame, pursuitPosition(t), -1 - Math.floor(t / 2)));
      }
    };
    await this.sleep(PURSUIT_LEAD_SECONDS + PURSUIT_SECONDS);
  }

  async run(): Promise<Recording> {
    this.startMs = performance.now();
    await this.announce(["Calibration", "Look at the black centre of each dot until it shrinks.", "Keep your head natural and fairly still."], 3);
    await this.stillDots();
    await this.announce([
      "Part 2: head movement",
      "Keep your EYES on each dot and move your HEAD the way the text under the dot asks.",
      "The ring fills as you move. When it turns green, hold still a moment.",
    ], 7);
    await this.headMoves();
    await this.announce(["Part 3: follow the moving dot", `Follow the dot smoothly with your eyes for ${PURSUIT_SECONDS} seconds.`], 5);
    await this.pursuit();
    this.handler = null;
    this.view = () => ({ title: ["Done - saving your session ..."] });

    const crops = await Promise.all(this.pending.map((p) => p.crop));
    const landmarks = new Float32Array(this.pending.length * LANDMARKS * 2);
    this.pending.forEach((p, i) => landmarks.set(p.points, i * LANDMARKS * 2));
    return {
      frames: this.pending.map((p, i) => ({ ...p.record, jpeg_bytes: crops[i].size })),
      landmarks,
      crops,
      groupPass: this.groupPass,
      durationS: (performance.now() - this.startMs) / 1000,
    };
  }
}
