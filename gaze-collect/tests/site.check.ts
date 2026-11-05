import { describe, expect, it } from "vitest";

import { checks, cropBox, headMoveProgress, measure, rating, type Point3, type Posture } from "../src/lib/posture";
import {
  CALIBRATION_MARGIN, DOT_GRID, HEAD_MOVES, MAX_FPS, gridTargets, hfovFor, pursuitPosition,
} from "../src/lib/protocol";
import { LANDMARKS, buildSessionFile, type Manifest } from "../src/lib/sessionFile";

const good: Posture = {
  distance_cm: 55, centre_x: 0.5, centre_y: 0.45, yaw: 2, pitch: -10, roll: 1,
  interpupil_px: 130, face_brightness: 120, background_brightness: 130, eyes_open: 1,
};

/** A flat frontal face, pupils 0.1 of the frame width apart. */
function face(): Point3[] {
  const points: Point3[] = Array.from({ length: LANDMARKS }, () => ({ x: 0.5, y: 0.5, z: 0 }));
  for (let i = 468; i < 473; i++) points[i] = { x: 0.45, y: 0.45, z: 0 };
  for (let i = 473; i < 478; i++) points[i] = { x: 0.55, y: 0.45, z: 0 };
  points[33] = { x: 0.42, y: 0.45, z: 0 };
  points[263] = { x: 0.58, y: 0.45, z: 0 };
  points[10] = { x: 0.5, y: 0.3, z: 0 };
  points[152] = { x: 0.5, y: 0.7, z: 0 };
  return points;
}

describe("calibration protocol matches webcam_gaze.py --calibration quick", () => {
  it("has the tracker's dot positions", () => {
    const targets = gridTargets(DOT_GRID[0], DOT_GRID[1], CALIBRATION_MARGIN, 7);
    expect(targets).toHaveLength(16);
    // numpy: np.linspace(0.06, 0.94, 4)
    const expected = [0.06, 0.3533333, 0.6466667, 0.94];
    for (const value of expected) {
      expect(targets.filter(([x]) => Math.abs(x - value) < 1e-6)).toHaveLength(4);
      expect(targets.filter(([, y]) => Math.abs(y - value) < 1e-6)).toHaveLength(4);
    }
  });

  it("keeps a fixed dot order", () => {
    expect(gridTargets(4, 4, 0.06, 7)).toEqual(gridTargets(4, 4, 0.06, 7));
    expect(gridTargets(4, 4, 0.06, 7)).not.toEqual(gridTargets(4, 4, 0.06, 11));
  });

  it("uses the tracker's moving-dot path", () => {
    // Values computed with webcam_gaze.pursuit_position(t, 1, 1).
    expect(pursuitPosition(0)[0]).toBeCloseTo(0.5, 9);
    expect(pursuitPosition(0)[1]).toBeCloseTo(0.9, 9);
    expect(pursuitPosition(13 / 4)[0]).toBeCloseTo(0.92, 9);
    expect(pursuitPosition(9 / 2)[1]).toBeCloseTo(0.1, 9);
  });

  it("knows the LifeCam and defaults for others", () => {
    expect(hfovFor("Microsoft® LifeCam HD-6000 for Notebooks")).toBe(59);
    expect(hfovFor("HD User Facing")).toBe(65);
  });

  it("cycles the tracker's head moves", () => {
    expect(HEAD_MOVES.map((m) => m.kind).slice(0, 6)).toEqual(["turn", "closer", "tilt", "shift", "turn", "back"]);
    expect(MAX_FPS).toBe(15);
  });
});

describe("positioning", () => {
  it("is green in a good position", () => {
    const items = checks(good);
    expect(items.every((c) => c.ok)).toBe(true);
    expect(rating(items)).toBe("best");
  });

  it("gives hints and scales the distance cost", () => {
    const far = checks({ ...good, distance_cm: 88 }).find((c) => c.label.startsWith("Distance"))!;
    expect(far.hint).toContain("Move closer");
    expect(far.impact).toContain("1.6x");
    expect(checks({ ...good, centre_x: 0.8 }).find((c) => c.label.startsWith("Centred"))!.hint).toContain("right");
    expect(checks({ ...good, centre_y: 0.1 }).find((c) => c.label === "Camera aim")!.hint).toContain("up");
  });

  it("only calls it backlit when the face itself is dark", () => {
    const lighting = (face_brightness: number, background_brightness: number) =>
      checks({ ...good, face_brightness, background_brightness }).find((c) => c.label === "Lighting")!;
    expect(lighting(92, 160).ok).toBe(true); // measured fine on 2026-10-23
    expect(lighting(70, 160).hint).toContain("behind you");
    expect(lighting(40, 50).hint).toContain("too dark");
  });

  it("rates accumulated problems", () => {
    expect(rating(checks(null))).toBe("reduced");
    const bad = checks({ ...good, distance_cm: 90, face_brightness: 40, background_brightness: 40, yaw: 30 });
    expect(rating(bad)).toBe("much reduced");
  });

  it("estimates distance from the pupil gap", () => {
    const m = measure(face(), 1280, 720, 59, true, { face: 100, background: 100 });
    expect(m.interpupil_px).toBeCloseTo(128, 5);
    // focal 1131 px x 63 mm / 128 px = 55.7 cm
    expect(m.distance_cm).toBeCloseTo(55.7, 0);
    expect(m.yaw).toBeCloseTo(0, 5);
    expect(m.roll).toBeCloseTo(0, 5);
  });

  it("makes crops large enough for the training patch, inside the frame", () => {
    const box = cropBox(face(), 1280, 720);
    expect(box.width).toBe(Math.ceil(128 * 3.4));
    expect(box.x).toBeGreaterThanOrEqual(0);
    expect(box.y).toBeGreaterThanOrEqual(0);
    expect(box.x + box.width).toBeLessThanOrEqual(1280);
    expect(box.y + box.height).toBeLessThanOrEqual(720);
    const edge = face().map((p) => ({ ...p, x: p.x - 0.42 }));
    expect(cropBox(edge, 1280, 720).x).toBe(0);
  });

  it("measures guided head movement", () => {
    const turn = HEAD_MOVES[0];
    expect(headMoveProgress(turn, { ...good, yaw: good.yaw + 8 }, good, 1280, 59)).toBeCloseTo(1, 5);
    expect(headMoveProgress(turn, good, good, 1280, 59)).toBe(0);
    expect(headMoveProgress(HEAD_MOVES[1], { ...good, distance_cm: 50 }, good, 1280, 59)).toBeCloseTo(1, 5);
    expect(headMoveProgress(HEAD_MOVES[5], { ...good, distance_cm: 50 }, good, 1280, 59)).toBe(0);
    expect(headMoveProgress(turn, null, good, 1280, 59)).toBe(0);
  });
});

describe("session file", () => {
  it("has the layout sync_sessions.py reads", async () => {
    const manifest = { format: 1, frames: [{ t: 1, x: 2, y: 3, group: 0, blink: 0, crop: [0, 0, 2, 2], jpeg_bytes: 3 }] } as unknown as Manifest;
    const landmarks = new Float32Array(LANDMARKS * 2).fill(7);
    const bytes = new Uint8Array(await buildSessionFile(manifest, landmarks, [new Blob([new Uint8Array([9, 8, 7])])]).arrayBuffer());
    expect(new TextDecoder().decode(bytes.slice(0, 4))).toBe("GZC1");
    const size = new DataView(bytes.buffer).getUint32(4, true);
    expect(JSON.parse(new TextDecoder().decode(bytes.slice(8, 8 + size))).format).toBe(1);
    expect(new Float32Array(bytes.slice(8 + size, 8 + size + LANDMARKS * 8).buffer)[0]).toBe(7);
    expect([...bytes.slice(-3)]).toEqual([9, 8, 7]);
  });
});
