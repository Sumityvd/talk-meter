import { describe, expect, it } from "vitest";

import { minutesCsv, secondsCsv, toCsv } from "../csv";
import { SessionRecorder, type Activity } from "../recorder";

const FPS = 15;
const speaking = (t: number) => 0.2 + 0.12 * Math.sin(2 * Math.PI * 5 * t);
const still = () => 0.12;

/** Drive `seconds` of frames starting at t0, labelled `label`. */
function drive(
  rec: SessionRecorder,
  t0: number,
  seconds: number,
  mar: (t: number) => number,
  label: Activity,
  face = true,
) {
  for (let i = 0; i < seconds * FPS; i++) {
    const t = t0 + i / FPS;
    rec.addFrame({ t, face, mar: face ? mar(t) : 0, smile: 0 }, label);
  }
}

describe("SessionRecorder", () => {
  it("closes one record per second and one per minute", () => {
    const rec = new SessionRecorder();
    drive(rec, 100, 60, still, "reading");
    drive(rec, 160, 60, speaking, "speaking");
    rec.finish();
    expect(rec.seconds).toHaveLength(120);
    expect(rec.minutes).toHaveLength(2);
    expect(rec.minutes[0]).toMatchObject({ minute: 0, label: "reading", talk_frac: 0, quiet_s: 60 });
    expect(rec.minutes[1].label).toBe("speaking");
    expect(rec.minutes[1].talk_frac).toBeGreaterThan(0.9);
  });

  it("stores the label and state on every second", () => {
    const rec = new SessionRecorder();
    drive(rec, 0, 3, speaking, "speaking");
    rec.finish();
    expect(rec.seconds.every((s) => s.label === "speaking" && s.state === "talking")).toBe(true);
  });

  it("marks seconds without a face", () => {
    const rec = new SessionRecorder();
    drive(rec, 0, 2, still, "other", false);
    rec.finish();
    expect(rec.seconds.map((s) => s.state)).toEqual(["no-face", "no-face"]);
    expect(rec.minutes).toHaveLength(0); // under the 10 s partial-minute floor
  });

  it("keeps a trailing partial minute of 10 seconds or more", () => {
    const rec = new SessionRecorder();
    drive(rec, 0, 75, still, "typing");
    rec.finish();
    expect(rec.minutes).toHaveLength(2);
    expect(rec.minutes[1]).toMatchObject({ minute: 1, startSecond: 60, endSecond: 74 });
  });

  it("closes a minute across a gap (hidden tab)", () => {
    const rec = new SessionRecorder();
    drive(rec, 0, 30, still, "reading");
    drive(rec, 200, 30, speaking, "speaking"); // 170 s later
    rec.finish();
    expect(rec.minutes.map((m) => m.minute)).toEqual([0, 3]);
  });
});

describe("CSV export", () => {
  it("escapes commas and quotes", () => {
    expect(toCsv(["a", "b"], [['x,y', 'say "hi"']])).toBe('a,b\n"x,y","say ""hi"""\n');
  });

  it("writes one row per second and per minute", () => {
    const rec = new SessionRecorder();
    drive(rec, 0, 60, still, "reading");
    rec.finish();
    const start = "2026-10-20T09:00:00.000Z";
    expect(secondsCsv(start, rec.seconds).trim().split("\n")).toHaveLength(61);
    const minutes = minutesCsv(start, rec.minutes).trim().split("\n");
    expect(minutes).toHaveLength(2);
    expect(minutes[1]).toContain(",reading,0,0,60,60,1,15");
  });
});
