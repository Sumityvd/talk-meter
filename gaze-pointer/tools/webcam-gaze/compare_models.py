"""Compares gaze models in on-screen pixels on the recorded calibrations, the way the tracker uses them:
model output -> the tracker's own personal correction (webcam_gaze.CnnGazeModel) -> screen point.

Models: xgaze (the current one), stagea (webcam fine-tune), personal (stage B: for each sitting, the fold model
that never saw that sitting).
Scores (mean error per dot, px; lower is better):
  same sitting    calibrate on the other dots of that sitting (the tracker's normal start), test on the held-out dot
  next sitting    calibrate in one sitting, test in another, after a one-dot drift fix (reusing a saved calibration)
Each is split into head-still and guided-head-move dots.

Two steps, because torch and mediapipe live in different venvs:
  D:\\gaze-train\\.venv\\Scripts\\python.exe compare_models.py extract --stagea <last.pth> --personal <run dir>
  <tracker venv>\\python.exe compare_models.py score
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
SESSIONS = sorted(d for d in (HERE / "sessions").glob("*") if (d / "session.npz").exists())
FEATURES = Path(r"D:\gaze-train\features")
SCREEN_CENTRE = np.array([960.0, 540.0])


def extract(args):
    import torch

    import train_gaze as t
    from xgaze_model import load_model

    device = torch.device("cuda")
    FEATURES.mkdir(parents=True, exist_ok=True)
    sessions = SESSIONS
    if args.live:
        # Sittings with an accuracy test, none of them used to train the final personal model.
        sessions = live_sessions()
        folds = {f"fold_{d.name}": Path(args.personal) / f"fold_{d.name}.pth" for d in sessions}
        folds = {name: str(path) for name, path in folds.items() if path.exists()}
        models = {"xgaze": None, "stagea": args.stagea, **(folds or {"final": str(Path(args.personal) / "final.pth")})}
    else:
        models = {"xgaze": None, "stagea": args.stagea}
        models.update({f"fold_{d.name}": str(Path(args.personal) / f"fold_{d.name}.pth") for d in SESSIONS})
    for name, weights in models.items():
        model = load_model(weights).to(device).eval()
        for session in sessions:
            data = np.load(session / "session.npz")
            meta = json.loads((session / "meta.json").read_text())
            x, rotations = data["x"], data["rotations"]
            ids = x[:, 6398].astype(int)
            pitchyaw, embedding = [], []
            with torch.no_grad():
                for start in range(0, len(ids), 32):
                    images = []
                    for i in ids[start:start + 32]:
                        image = cv2.imread(str(session / "patches" / f"{i:05d}.jpg")).astype(np.float32) / 255.0
                        images.append(((cv2.cvtColor(image, cv2.COLOR_BGR2RGB) - t.MEAN) / t.STD).transpose(2, 0, 1))
                    batch = torch.from_numpy(np.stack(images)).to(device)
                    gaze, features = model(batch)
                    mirror, _ = model(torch.flip(batch, dims=[3]))
                    # As gaze_cnn.CnnGaze with flip=True: mirror-averaged gaze, unmirrored embedding.
                    pitchyaw.append(torch.stack([(gaze[:, 0] + mirror[:, 0]) / 2,
                                                 (gaze[:, 1] - mirror[:, 1]) / 2], dim=1).cpu().numpy())
                    embedding.append(features.cpu().numpy())
            pitchyaw = np.concatenate(pitchyaw)
            p, y = pitchyaw[:, 0], pitchyaw[:, 1]
            vectors = np.stack([-np.cos(p) * np.sin(y), -np.sin(p), -np.cos(p) * np.cos(y)], axis=1)
            direction = np.einsum("nji,nj->ni", rotations[ids], vectors)   # R^T @ v, camera space
            origin = x[:, 6392:6395]
            hit = origin + direction * (-origin[:, 2:3] / direction[:, 2:3])
            (width_px, height_px), (width_mm, height_mm) = meta["screen_px"], meta["screen_mm"]
            screen = np.stack([(width_mm / 2 - hit[:, 0]) / width_mm * width_px,
                               (hit[:, 1] - meta["camera_above_mm"]) / height_mm * height_px], axis=1)
            np.savez(FEATURES / f"{name}__{session.name}.npz", pitchyaw=pitchyaw.astype(np.float32),
                     embedding=np.concatenate(embedding).astype(np.float32), direction=direction.astype(np.float32),
                     screen=screen.astype(np.float32))
            if name == "xgaze":   # the recorded run used the same model: this must reproduce it closely
                difference = np.degrees(np.abs(pitchyaw - x[:, 22:24])).mean()
                print(f"check {session.name}: recomputed vs recorded pitch/yaw differ by {difference:.2f} deg", flush=True)
        print(f"extracted {name}", flush=True)
        del model
        torch.cuda.empty_cache()


def load_features(name, session):
    """The session's recorded rows with the CNN columns replaced by this model's outputs."""
    data = np.load(session / "session.npz")
    x = data["x"].astype(np.float64)
    f = np.load(FEATURES / f"{name}__{session.name}.npz")
    x[:, 20:22], x[:, 22:24], x[:, 24:2072], x[:, 6395:6398] = f["screen"], f["pitchyaw"], f["embedding"], f["direction"]
    return x, data["y"], data["groups"], data["group_pass"]


def fit(w, x, y):
    return w.CnnGazeModel.fit(x, y, w.MODEL_COMPONENTS, w.MODEL_RIDGE)


def same_sitting(w, name_for, session):
    """Leave-one-dot-out within a sitting. Returns errors per dot pass {0: still, 1: head moved} and jitter."""
    x, y, groups, passes = load_features(name_for(session), session)
    pcas = w.CnnGazeModel.fit_pcas(x, w.MODEL_COMPONENTS)
    r = w.CnnGazeModel.reduce(x, pcas)
    errors, jitter = {0: [], 1: []}, []
    for g in np.unique(groups[groups >= 0]):
        held = groups == g
        model = w.CnnGazeModel.fit_reduced(r[~held], y[~held], w.MODEL_RIDGE, pcas, w.MODEL_COMPONENTS)
        predicted = model.predict_reduced(r[held])
        mean = predicted.mean(axis=0)
        errors[int(passes[g])].append(np.linalg.norm(mean - y[held][0]))
        jitter.append(np.sqrt(np.mean(np.sum((predicted - mean) ** 2, axis=1))))
    return errors, jitter


def next_sitting(w, name_for, train, test):
    """Calibrate in `train`, use in `test` with a one-dot drift fix from the dot nearest the screen centre."""
    name = name_for(test)   # fold models: the one that never saw the test sitting
    x, y, _, _ = load_features(name, train)
    model = fit(w, x, y)
    x, y, groups, passes = load_features(name, test)
    dots = np.unique(groups[groups >= 0])
    means = np.array([model.predict(x[groups == g]).mean(axis=0) for g in dots])
    targets = np.array([y[groups == g][0] for g in dots])
    centre = np.argmin(np.linalg.norm(targets - SCREEN_CENTRE, axis=1))
    offsets = means + (targets[centre] - means[centre]) - targets
    errors = {0: [], 1: []}
    for i, g in enumerate(dots):
        if i != centre:
            errors[int(passes[g])].append(np.linalg.norm(offsets[i]))
    return errors


def live_sessions():
    return [d for d in SESSIONS if np.any(np.load(d / "session.npz")["group_pass"] == 2)]


def score_live(_):
    """Same recorded frames through every model: calibrate on the sitting's calibration dots, score its accuracy
    test (16 dots head still, then the head-moved dots). Also prints how far the head really moved."""
    import webcam_gaze as w

    candidates = {"xgaze (current)": lambda s: "xgaze", "stage A (general)": lambda s: "stagea",
                  "personal, 3 sittings": lambda s: "final"}
    if all((FEATURES / f"fold_{s.name}__{s.name}.npz").exists() for s in live_sessions()):
        candidates["personal, all others"] = lambda s: f"fold_{s.name}"
    print("mean error per dot, px; lower is better")
    totals = {label: {"still": [], "moved": []} for label in candidates}
    for session in live_sessions():
        x0, _, groups0, passes0 = load_features("xgaze", session)
        tests = np.where(passes0 == 2)[0]
        still, moved = tests[:16], tests[16:]
        centre = lambda dots: x0[np.isin(groups0, dots), 6392:6395].mean(axis=0)
        shift = np.linalg.norm(centre(moved) - centre(still)) / 10
        print(f"{session.name}: head moved {shift:.1f} cm between the two test passes")
        for label, name_for in candidates.items():
            x, y, groups, _ = load_features(name_for(session), session)
            train = ~np.isin(groups, tests)
            model = fit(w, x[train], y[train])
            error = lambda dots: [np.linalg.norm(model.predict(x[groups == g]).mean(axis=0) - y[groups == g][0])
                                  for g in dots]
            s, m = error(still), error(moved)
            totals[label]["still"] += s
            if shift >= 2:   # a pass where the head stayed put says nothing about head movement
                totals[label]["moved"] += m
            print(f"  {label:<20} head still {np.mean(s):5.0f}   second pass {np.mean(m):5.0f}", flush=True)
    print("all sittings:")
    for label, t in totals.items():
        moved_score = f"{np.mean(t['moved']):5.0f}" if t["moved"] else "  n/a"
        print(f"  {label:<20} head still {np.mean(t['still']):5.0f}   head really moved {moved_score}")


def score(_):
    import webcam_gaze as w

    candidates = {"xgaze (current)": lambda s: "xgaze", "stage A": lambda s: "stagea",
                  "stage B (personal)": lambda s: f"fold_{s.name}"}
    print(f"{len(SESSIONS)} sittings; mean error per dot, px; lower is better")
    print(f"{'':<20}{'same sitting':^26}{'next sitting + drift dot':^28}")
    print(f"{'model':<20}{'still':>8}{'moved':>8}{'jitter':>8}{'still':>14}{'moved':>10}")
    for label, name_for in candidates.items():
        same, jitters = {0: [], 1: []}, []
        for session in SESSIONS:
            errors, jitter = same_sitting(w, name_for, session)
            same[0] += errors[0]
            same[1] += errors[1]
            jitters += jitter
        nxt = {0: [], 1: []}
        for test in SESSIONS:
            for train in SESSIONS:
                if train != test:
                    errors = next_sitting(w, name_for, train, test)
                    nxt[0] += errors[0]
                    nxt[1] += errors[1]
        print(f"{label:<20}{np.mean(same[0]):8.0f}{np.mean(same[1]):8.0f}{np.mean(jitters):8.0f}"
              f"{np.mean(nxt[0]):14.0f}{np.mean(nxt[1]):10.0f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    e = commands.add_parser("extract")
    e.add_argument("--stagea", required=True)
    e.add_argument("--personal", required=True)
    e.add_argument("--live", action="store_true", help="the sittings with an accuracy test, for score-live")
    e.set_defaults(run=extract)
    commands.add_parser("score").set_defaults(run=score)
    commands.add_parser("score-live").set_defaults(run=score_live)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
