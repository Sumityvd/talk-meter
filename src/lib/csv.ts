import type { MinuteRecord, SecondRecord } from "./recorder";

type Cell = string | number | boolean | null | undefined;

export function toCsv(header: string[], rows: Cell[][]): string {
  const esc = (v: Cell) => {
    if (v === null || v === undefined) return "";
    const s = typeof v === "number" ? (Number.isFinite(v) ? String(round(v)) : "") : String(v);
    return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  return [header, ...rows].map((r) => r.map(esc).join(",")).join("\n") + "\n";
}

const round = (v: number) => Math.round(v * 10000) / 10000;

export function secondsCsv(sessionStart: string, seconds: ReadonlyArray<SecondRecord>): string {
  return toCsv(
    [
      "session_start",
      "second",
      "label",
      "state",
      "frames",
      "face_frames",
      "mar_min",
      "mar_max",
      "mar_median",
      "mar_range",
      "crossings",
      "wide_frac",
      "smile_mean",
    ],
    seconds.map((s) => [
      sessionStart,
      s.second,
      s.label,
      s.state,
      s.frames,
      s.faceFrames,
      s.stats?.marMin,
      s.stats?.marMax,
      s.stats?.marMedian,
      s.stats?.marRange,
      s.stats?.crossings,
      s.stats?.wideFrac,
      s.stats?.smileMean,
    ]),
  );
}

export function minutesCsv(sessionStart: string, minutes: ReadonlyArray<MinuteRecord>): string {
  return toCsv(
    [
      "session_start",
      "minute",
      "start_second",
      "end_second",
      "label",
      "talk_frac",
      "smile_frac",
      "quiet_s",
      "measured_seconds",
      "face_ratio",
      "fps",
    ],
    minutes.map((m) => [
      sessionStart,
      m.minute,
      m.startSecond,
      m.endSecond,
      m.label,
      m.talk_frac,
      m.smile_frac,
      m.quiet_s,
      m.seconds,
      m.faceRatio,
      m.fps,
    ]),
  );
}

export function downloadText(filename: string, text: string): void {
  const url = URL.createObjectURL(new Blob([text], { type: "text/csv;charset=utf-8" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
