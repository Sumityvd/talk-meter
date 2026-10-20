import { describe, expect, it } from "vitest";

import { classifySecond, measureWindowSpeech, mouthAspectRatio, MAR_IDX } from "../speech";

const FPS = 15;
type Frame = { t: number; mar: number; smile: number };

function frames(seconds: number, mar: (t: number) => number, smile: (t: number) => number = () => 0): Frame[] {
  const out: Frame[] = [];
  for (let i = 0; i < seconds * FPS; i++) {
    const t = i / FPS;
    out.push({ t, mar: mar(t), smile: smile(t) });
  }
  return out;
}

// Speech: the mouth opens and closes about five times a second.
const speaking = (t: number) => 0.2 + 0.12 * Math.sin(2 * Math.PI * 5 * t);
const still = () => 0.12;

describe("mouthAspectRatio", () => {
  it("is mouth opening over width", () => {
    const lm = Array.from({ length: 478 }, () => ({ x: 0, y: 0 }));
    lm[MAR_IDX.leftCorner] = { x: 0, y: 0 };
    lm[MAR_IDX.rightCorner] = { x: 1, y: 0 };
    lm[MAR_IDX.upperInner] = { x: 0.5, y: -0.1 };
    lm[MAR_IDX.lowerInner] = { x: 0.5, y: 0.1 };
    lm[MAR_IDX.upperOuter] = { x: 0.5, y: -0.2 };
    lm[MAR_IDX.lowerOuter] = { x: 0.5, y: 0.2 };
    expect(mouthAspectRatio(lm)).toBeCloseTo(0.3, 6);
  });

  it("is 0 when landmarks are missing", () => {
    expect(mouthAspectRatio([])).toBe(0);
  });
});

describe("classifySecond", () => {
  it("needs at least three frames", () => {
    expect(classifySecond([{ mar: 0.2, smile: 0 }, { mar: 0.3, smile: 0 }])).toBeNull();
  });

  it("flags an oscillating mouth as talking", () => {
    const s = classifySecond(frames(1, speaking));
    expect(s?.talking).toBe(true);
  });

  it("does not flag a wide-open (yawning) second as talking", () => {
    const s = classifySecond(frames(1, (t) => 0.7 + 0.1 * Math.sin(2 * Math.PI * 5 * t)));
    expect(s?.talking).toBe(false);
  });
});

describe("measureWindowSpeech", () => {
  it("calls a minute of speech talking", () => {
    const s = measureWindowSpeech(frames(60, speaking));
    expect(s.seconds).toBe(60);
    expect(s.talk_frac).toBeGreaterThan(0.9);
  });

  it("calls a still face quiet", () => {
    expect(measureWindowSpeech(frames(60, still))).toEqual({
      talk_frac: 0,
      smile_frac: 0,
      quiet_s: 60,
      seconds: 60,
    });
  });

  it("does not mistake a six-second yawn for talking", () => {
    const yawn = (t: number) => (t >= 20 && t < 26 ? 0.75 + 0.02 * Math.sin(t) : 0.12);
    expect(measureWindowSpeech(frames(60, yawn)).talk_frac).toBe(0);
  });

  it("measures smiling separately from quiet", () => {
    const s = measureWindowSpeech(frames(60, still, (t) => (t < 30 ? 0.8 : 0)));
    expect(s.smile_frac).toBeCloseTo(0.5, 1);
    expect(s.quiet_s).toBe(30);
  });
});
