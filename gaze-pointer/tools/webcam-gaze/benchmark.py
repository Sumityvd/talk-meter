"""Engine-neutral accuracy benchmark: measures wherever the *system cursor* goes while you look at targets.

Any gaze engine that moves the Windows cursor can be compared on identical terms:
GazeFlow's GazePointer, or webcam_gaze.py with --output cursor.

Pass 1: 16 targets (4x4), head in your normal posture.
Pass 2: 9 targets (3x3) after shifting your head ~5 cm (tests head-movement robustness).

Usage:
  py -3.12 benchmark.py --label gazeflow            # GazeFlow running and calibrated
  py -3.12 benchmark.py --label ours-landmarks      # webcam_gaze.py --output cursor --engine landmarks
  py -3.12 benchmark.py --summary                   # median of all runs per label
Results: bench/<time>_<label>.json
"""

import argparse
import ctypes
import json
import time
from ctypes import wintypes
from pathlib import Path

import numpy as np

from webcam_gaze import (BENCH_DIR, BENCH_PASSES, BENCH_TIMING, Stage, benchmark_grid, draw_target, save_bench,
                         score_targets, screen_info, wait_for_space)

PROFILE_DIR = Path(__file__).resolve().parent / "profiles"
MOVE_SECONDS, SETTLE_SECONDS, COLLECT_SECONDS = BENCH_TIMING
SAMPLE_HZ = 60


def measured_distance_cm():
    """Eye-to-camera distance from the last cnn calibration, else 60 cm."""
    try:
        validation = json.loads((PROFILE_DIR / "calibration_cnn.json").read_text())["validation"]
        return round(float(validation["measured_distance_cm"]), 1)
    except (OSError, KeyError, TypeError, ValueError):
        return 60.0


def cursor_position():
    point = wintypes.POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(point))
    return point.x, point.y


def run_pass(stage, targets):
    """Returns list of (target, cursor samples[N, 2]) or None on ESC."""
    results = []
    previous = (stage.width / 2, stage.height / 2)
    for target in targets:
        samples = []
        start = time.monotonic()
        total = MOVE_SECONDS + SETTLE_SECONDS + COLLECT_SECONDS
        while (elapsed := time.monotonic() - start) < total:
            if elapsed < MOVE_SECONDS:
                k = elapsed / MOVE_SECONDS
                k = k * k * (3 - 2 * k)
                center = (previous[0] + (target[0] - previous[0]) * k, previous[1] + (target[1] - previous[1]) * k)
                progress = 0.0
            else:
                center = target
                progress = (elapsed - MOVE_SECONDS) / (SETTLE_SECONDS + COLLECT_SECONDS)
                if elapsed >= MOVE_SECONDS + SETTLE_SECONDS:
                    samples.append(cursor_position())
            canvas = stage.blank()
            draw_target(canvas, center, progress)
            if stage.show(canvas, max(1, int(1000 / SAMPLE_HZ))) == 27:
                return None
        previous = target
        results.append((np.array(target), np.array(samples, float)))
    return results


def summary():
    runs = {}
    for path in sorted(BENCH_DIR.glob("*.json")):
        data = json.loads(path.read_text())
        runs.setdefault(data["label"], []).append(data)
    if not runs:
        print("no runs in", BENCH_DIR)
        return
    print(f"{'label':<20}{'runs':>5}{'still px':>10}{'still deg':>10}{'moved px':>10}{'moved deg':>10}{'jitter px':>10}")
    for label, items in runs.items():
        def median(key, field):
            values = [r[key][field] for r in items if r.get(key)]
            return float(np.median(values)) if values else float("nan")
        print(f"{label:<20}{len(items):>5}{median('still', 'accuracy_px'):>10.0f}{median('still', 'accuracy_deg'):>10.2f}"
              f"{median('moved', 'accuracy_px'):>10.0f}{median('moved', 'accuracy_deg'):>10.2f}"
              f"{median('still', 'precision_px'):>10.0f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", help="engine name for this run, e.g. gazeflow, ours-cnn")
    parser.add_argument("--distance-cm", type=float, default=measured_distance_cm(),
                        help="eye-to-screen distance for degrees (default: measured in the last cnn calibration)")
    parser.add_argument("--summary", action="store_true", help="print the median of all saved runs per label")
    args = parser.parse_args()
    if args.summary:
        summary()
        return
    if not args.label:
        parser.error("--label is required (or use --summary)")

    width, height, width_mm, height_mm = screen_info()
    mm_per_px = (width_mm / width + height_mm / height) / 2
    stage = Stage(width, height)
    report = {"label": args.label, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
              "screen": [width, height], "distance_cm": args.distance_cm}
    try:
        if not wait_for_space(stage, [
            f"Gaze benchmark: {args.label}",
            "Your gaze engine must be running and moving the mouse cursor.",
            "Look at the black centre of each dot.  Don't touch the mouse.",
            "Pass 1: sit normally.  SPACE to start, ESC to cancel.",
        ]):
            return
        (_, still_grid), (_, moved_grid) = BENCH_PASSES
        still = run_pass(stage, benchmark_grid(width, height, *still_grid))
        if still is None:
            return
        report["still"] = score_targets(still, mm_per_px, args.distance_cm)
        if wait_for_space(stage, [
            "Pass 2: head moved",
            "Shift your head about 5 cm (a hand's width) to one side or up/down, and stay there.",
            "Keep looking at the dots.  SPACE to start, ESC to skip.",
        ]):
            moved = run_pass(stage, benchmark_grid(width, height, *moved_grid))
            if moved:
                report["moved"] = score_targets(moved, mm_per_px, args.distance_cm)
    finally:
        stage.close()

    path = save_bench(report)
    for key in ("still", "moved"):
        if key in report:
            r = report[key]
            print(f"{key:>5}: accuracy {r['accuracy_px']:.0f} px = {r['accuracy_cm']:.1f} cm = {r['accuracy_deg']:.2f} deg, "
                  f"jitter {r['precision_px']:.0f} px")
    print("saved", path)


if __name__ == "__main__":
    main()
