"""Positioning window shown before calibration (and before the drift check on later sessions).

The user sees a mirrored live view of themselves with a face oval and a live checklist: face found,
distance, centred left/right, camera aim (face height in the image), head straight, lighting, eyes open.
Each failing check shows a plain hint ("Move closer", "Tilt the camera up a little", ...). When every
check has passed for READY_SECONDS, a short countdown runs and the measured posture is returned.

On later sessions the posture saved with the calibration is passed in as `baseline`, and the checks
become "match your calibration position": same distance, same place in the image, same head angles.
Sitting differently from the calibration was the largest error source measured (2 cm closer moved
gaze ~250 px), so this is the cheapest accuracy gain available.
"""

import math
import time

import cv2
import numpy as np

from gaze_cnn import FACE_MODEL, MEDIAPIPE_POINTS, camera_matrix

IRISES = ((468, 469, 470, 471, 472), (473, 474, 475, 476, 477))
INTERPUPIL_MM = 63.0          # adult average; distance estimates are within ~10%
READY_SECONDS = 2.0
COUNTDOWN_SECONDS = 3
PREVIEW_WIDTH = 1024

# Limits without a baseline (first calibration) and tolerances against a saved baseline.
DISTANCE_CM = (45.0, 75.0)
CENTRE_X_TOLERANCE = 0.12
FACE_Y_RANGE = (0.28, 0.62)   # face centre height in the image; a little above the middle is natural
YAW_LIMIT, ROLL_LIMIT = 12.0, 8.0
BASELINE_DISTANCE_CM = 5.0
BASELINE_POSITION = 0.06
BASELINE_ANGLE = 6.0
FACE_BRIGHTNESS = (60.0, 210.0)
BACKLIGHT_MARGIN = 45.0
BACKLIT_FACE_MAX = 75.0       # backlight only counts when the face is darker than this
MIN_INTERPUPIL_PX = 40.0
REFERENCE_DISTANCE_CM = 55.0  # distance of the best measured calibrations; farther scales the error up
CONFIRM_SECONDS = 5

GREEN, RED, AMBER, GREY, WHITE = (90, 220, 90), (80, 80, 240), (40, 190, 250), (130, 130, 130), (235, 235, 235)


def measure(frame, info, camera):
    """Posture measurements for one frame, or None without a face."""
    if not info["face"] or info["points"] is None:
        return None
    points = info["points"]
    height, width = frame.shape[:2]
    pupils = [points[list(ring)].mean(axis=0) for ring in IRISES]
    interpupil_px = float(np.linalg.norm(pupils[0] - pupils[1]))
    focal = camera[0, 0]
    distance_cm = focal * INTERPUPIL_MM / max(interpupil_px, 1.0) / 10.0

    ok, rvec, _ = cv2.solvePnP(FACE_MODEL.reshape(-1, 1, 3), points[MEDIAPIPE_POINTS].reshape(-1, 1, 2),
                               camera, None, flags=cv2.SOLVEPNP_EPNP)
    pitch = yaw = roll = 0.0
    if ok:
        pitch, yaw, roll = cv2.RQDecomp3x3(cv2.Rodrigues(rvec)[0])[0]

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    x0, y0 = np.clip(points.min(axis=0).astype(int), 0, [width - 1, height - 1])
    x1, y1 = np.clip(points.max(axis=0).astype(int), 0, [width - 1, height - 1])
    face_region = grey[y0:y1 + 1, x0:x1 + 1]
    mask = np.ones_like(grey, bool)
    mask[y0:y1 + 1, x0:x1 + 1] = False
    face_brightness = float(face_region.mean()) if face_region.size else 0.0
    background = float(grey[mask].mean()) if mask.any() else face_brightness

    centre = points.mean(axis=0)
    return {
        "distance_cm": distance_cm, "centre_x": float(centre[0] / width), "centre_y": float(centre[1] / height),
        "yaw": float(yaw), "pitch": float(pitch), "roll": float(roll), "interpupil_px": interpupil_px,
        "face_brightness": face_brightness, "background_brightness": background, "eyes_open": not info["blink"],
    }


def checks(m, baseline):
    """List of (label, ok, hint). `m` is a measure() result."""
    if m is None:
        return [("Face", False, "No face found - sit in front of the camera and look at the screen")]
    result = [("Face", True, "")]
    d = m["distance_cm"]
    if baseline:
        target = baseline["distance_cm"]
        ok = abs(d - target) <= BASELINE_DISTANCE_CM
        hint = "" if ok else (f"Move closer (now {d:.0f} cm, calibrated at {target:.0f} cm)" if d > target
                              else f"Move back (now {d:.0f} cm, calibrated at {target:.0f} cm)")
    else:
        low, high = DISTANCE_CM
        ok = low <= d <= high
        hint = "" if ok else (f"Move closer (now {d:.0f} cm, best {low:.0f}-{high:.0f} cm)" if d > high
                              else f"Move back (now {d:.0f} cm, best {low:.0f}-{high:.0f} cm)")
    result.append((f"Distance {d:.0f} cm", ok, hint))

    # The camera image is not mirrored: a face right of centre means the user sits too far to their left.
    target_x = baseline["centre_x"] if baseline else 0.5
    tolerance = BASELINE_POSITION if baseline else CENTRE_X_TOLERANCE
    dx = m["centre_x"] - target_x
    ok = abs(dx) <= tolerance
    result.append(("Centred left/right", ok, "" if ok else
                   ("Move a little to your right" if dx > 0 else "Move a little to your left")))

    if baseline:
        dy = m["centre_y"] - baseline["centre_y"]
        ok = abs(dy) <= BASELINE_POSITION
        hint = "" if ok else ("Sit a little higher, or aim the camera as it was at calibration" if dy > 0
                              else "Sit a little lower, or aim the camera as it was at calibration")
    else:
        low, high = FACE_Y_RANGE
        ok = low <= m["centre_y"] <= high
        hint = "" if ok else ("Tilt the camera up a little" if m["centre_y"] < low
                              else "Tilt the camera down a little")
    result.append(("Camera aim", ok, hint))

    if baseline:
        dyaw, dpitch = m["yaw"] - baseline["yaw"], m["pitch"] - baseline["pitch"]
        dyaw = (dyaw + 180) % 360 - 180
        dpitch = (dpitch + 180) % 360 - 180
        ok = abs(dyaw) <= BASELINE_ANGLE and abs(dpitch) <= BASELINE_ANGLE and abs(m["roll"]) <= ROLL_LIMIT
        hint = "" if ok else ("Turn your head back toward the screen centre" if abs(dyaw) > BASELINE_ANGLE else
                              "Raise or lower your chin as at calibration" if abs(dpitch) > BASELINE_ANGLE else
                              "Keep your head level (don't tilt it sideways)")
    else:
        ok = abs(m["yaw"]) <= YAW_LIMIT and abs(m["roll"]) <= ROLL_LIMIT
        hint = "" if ok else ("Face the screen straight on" if abs(m["yaw"]) > YAW_LIMIT
                              else "Keep your head level (don't tilt it sideways)")
    result.append(("Head straight", ok, hint))

    face, background = m["face_brightness"], m["background_brightness"]
    low, high = FACE_BRIGHTNESS
    # A bright wall behind is fine while the face itself is well exposed (face 92 vs background 160 tracked
    # well on 2026-10-23); it only matters once it pushes the camera's exposure down on the face.
    if background - face > BACKLIGHT_MARGIN and face < BACKLIT_FACE_MAX:
        ok, hint = False, "Bright light behind you - face a lamp or window instead"
    elif face < low:
        ok, hint = False, "Your face is too dark - add light in front of you"
    elif face > high:
        ok, hint = False, "Too much light on your face - dim it or move away from direct light"
    else:
        ok, hint = True, ""
    result.append(("Lighting", ok, hint))

    ok = m["eyes_open"] and m["interpupil_px"] >= MIN_INTERPUPIL_PX
    hint = "" if ok else ("Open your eyes and look at the screen" if not m["eyes_open"]
                          else "Your eyes are too small in the image - move closer")
    result.append(("Eyes visible", ok, hint))
    return result


def impacts(m, items, baseline):
    """What each unmet check does to accuracy if calibration starts anyway: [(label, impact, severity)],
    severity 1 = some loss, 2 = large loss. Only the distance effect is quantified (the eyes' size in the
    image scales with 1/distance); the others are described, not invented as numbers."""
    result = []
    for label, ok, _ in items:
        if ok:
            continue
        if baseline:
            result.append((label, "Differs from your calibration position: the cursor will be offset, and the "
                                  "drift dot only fixes part of that", 2))
        elif label.startswith("Face"):
            result.append((label, "No face: calibration cannot run", 3))
        elif label.startswith("Distance"):
            ratio = m["distance_cm"] / REFERENCE_DISTANCE_CM
            if ratio > 1:
                result.append((label, f"Your eyes look smaller to the camera: expect roughly {ratio:.1f}x the error "
                                      "and more jitter", 2 if ratio > 1.4 else 1))
            else:
                result.append((label, "Too close: your head can leave the camera view during the head-movement "
                                      "part, and edge dots may be missed", 1))
        elif label.startswith("Centred"):
            result.append((label, "Your head may leave the camera view when it moves: tracking drops out near "
                                  "the image edge", 1))
        elif label.startswith("Camera aim"):
            result.append((label, "The camera sees your eyes at a steep angle: less accurate, mostly up/down", 1))
        elif label.startswith("Head"):
            result.append((label, "A turned or tilted head hides part of one eye: less accurate, mostly "
                                  "left/right", 1))
        elif label.startswith("Lighting"):
            too_bright = m["face_brightness"] > FACE_BRIGHTNESS[1]
            result.append((label, "Washed-out eyes: less accurate" if too_bright else
                           "Dark, noisy eye detail: more cursor jitter and less accuracy", 1 if too_bright else 2))
        elif label.startswith("Eyes"):
            result.append((label, "Calibration needs open eyes on each dot" if not m["eyes_open"] else
                           "Eyes too small in the image: much less accurate", 2))
    return result


def rating(issues):
    """Overall expected accuracy for the impacts() list."""
    total = sum(severity for _, _, severity in issues)
    return ("best" if total == 0 else "slightly reduced" if total == 1 else "reduced" if total <= 3
            else "much reduced")


def confirm_start(stage, issues):
    """Shown after SPACE with unmet checks: lists what each costs. Starts after a countdown (SPACE: now);
    ESC goes back to positioning. Returns True to start."""
    end = time.monotonic() + CONFIRM_SECONDS
    while (remaining := end - time.monotonic()) > 0:
        canvas = stage.blank()
        stage.text(canvas, [f"Starting with expected accuracy: {rating(issues).upper()}"], y=200, scale=1.1)
        y = 300
        for label, impact, severity in issues:
            cv2.circle(canvas, (260, y - 8), 9, RED if severity >= 2 else AMBER, -1, cv2.LINE_AA)
            cv2.putText(canvas, label, (285, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2, cv2.LINE_AA)
            cv2.putText(canvas, impact, (285, y + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, AMBER, 1, cv2.LINE_AA)
            y += 90
        stage.text(canvas, [f"Starting in {math.ceil(remaining)} ...      SPACE: start now    "
                            "ESC: back to positioning"], y=max(y + 40, 800), scale=0.8)
        key = stage.show(canvas, 30)
        if key == 32:
            return True
        if key == 27:
            return False
    return True


def draw_camera_choice(canvas, cameras, current, left, y):
    """Camera choice line: '1 HD User Facing   [2 Microsoft LifeCam]   - press a number to switch'."""
    x = left
    cv2.putText(canvas, "Camera:", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, GREY, 1, cv2.LINE_AA)
    x += 100
    for number, (index, name) in enumerate(cameras, start=1):
        label = f"{number}  {name.replace(chr(0xAE), '')}"
        selected = index == current
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2 if selected else 1)
        if selected:
            cv2.rectangle(canvas, (x - 8, y - h - 8), (x + w + 8, y + 8), GREEN, 2)
        cv2.putText(canvas, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE if selected else GREY,
                    2 if selected else 1, cv2.LINE_AA)
        x += w + 40
    if len(cameras) > 1:
        cv2.putText(canvas, "press a number to switch", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREY, 1, cv2.LINE_AA)


def draw(stage, frame, info, m, items, baseline, camera, ready_fraction, countdown, cameras=None, current=None):
    canvas = stage.blank()
    height, width = frame.shape[:2]
    scale = PREVIEW_WIDTH / width
    preview = cv2.flip(cv2.resize(frame, (PREVIEW_WIDTH, int(height * scale))), 1)
    ph, pw = preview.shape[:2]
    left, top = (stage.width - pw) // 2, 110

    # Face oval at the target position and size (mirrored, like the preview).
    target_x = baseline["centre_x"] if baseline else 0.5
    target_y = baseline["centre_y"] if baseline else sum(FACE_Y_RANGE) / 2
    target_cm = baseline["distance_cm"] if baseline else sum(DISTANCE_CM) / 2
    interpupil_px = camera[0, 0] * INTERPUPIL_MM / (target_cm * 10) * scale
    centre = (int((1 - target_x) * pw), int(target_y * ph))
    all_ok = all(ok for _, ok, _ in items)
    cv2.ellipse(preview, centre, (int(interpupil_px * 1.45), int(interpupil_px * 2.0)), 0, 0, 360,
                GREEN if all_ok else WHITE, 3, cv2.LINE_AA)
    if info["points"] is not None:
        for ring in IRISES:
            p = info["points"][list(ring)].mean(axis=0) * scale
            cv2.circle(preview, (int(pw - p[0]), int(p[1])), 5, GREEN if all_ok else AMBER, -1, cv2.LINE_AA)
    canvas[top:top + ph, left:left + pw] = preview

    title = "Match your calibration position" if baseline else "Get into position"
    stage.text(canvas, [title], y=45, scale=1.1)
    if cameras:
        draw_camera_choice(canvas, cameras, current, left, top - 18)
    y = top + ph + 45
    for label, ok, hint in items:
        cv2.circle(canvas, (left + 12, y - 8), 9, GREEN if ok else RED, -1, cv2.LINE_AA)
        cv2.putText(canvas, label, (left + 35, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, WHITE, 2, cv2.LINE_AA)
        if hint:
            cv2.putText(canvas, hint, (left + 300, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, AMBER, 2, cv2.LINE_AA)
        y += 38
    issues = impacts(m, items, baseline)
    if countdown is not None:
        message, color = f"Good position - starting in {countdown}", GREEN
    elif all_ok:
        message, color = "Good - hold still", GREEN
    elif m is None:
        message, color = "No face found - calibration needs your face in view", RED
    else:
        n = len(issues)
        message = (f"SPACE: start anyway - expected accuracy {rating(issues)} "
                   f"({n} issue{'s' if n > 1 else ''})")
        color = AMBER
    bar = int(pw * ready_fraction)
    cv2.rectangle(canvas, (left, y), (left + pw, y + 10), (60, 60, 60), -1)
    cv2.rectangle(canvas, (left, y), (left + bar, y + 10), GREEN, -1)
    cv2.putText(canvas, message, (left, y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)
    cv2.putText(canvas, "ESC cancel", (left + pw - 120, y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREY, 1, cv2.LINE_AA)
    return canvas


def position_check(cap, extract, stage, hfov_deg, baseline=None, cameras=None, current=None):
    """Runs the positioning window. Returns the measured posture (medians over the ready period),
    the current measurement if the user pressed SPACE (with "warnings" and "expected_accuracy" when
    checks were unmet and the user confirmed starting anyway), None on ESC, or {"switch_camera": index} when
    the user picked another camera from `cameras` ([(index, name)], number keys 1-9)."""
    camera = None
    good, good_since, countdown_start = [], None, None
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        if camera is None:
            camera = camera_matrix(frame.shape[1], frame.shape[0], hfov_deg)
        _, info = extract(frame)
        m = measure(frame, info, camera)
        items = checks(m, baseline)
        now = time.monotonic()
        if all(ok for _, ok, _ in items):
            if good_since is None:
                good_since, good = now, []
            good.append(m)
        else:
            good_since, countdown_start, good = None, None, []
        ready = 0.0 if good_since is None else min(1.0, (now - good_since) / READY_SECONDS)
        if ready >= 1.0 and countdown_start is None:
            countdown_start = now
        countdown = None
        if countdown_start is not None:
            remaining = COUNTDOWN_SECONDS - (now - countdown_start)
            if remaining <= 0:
                return {key: float(np.median([g[key] for g in good])) for key in good[0]}
            countdown = math.ceil(remaining)
        key = stage.show(draw(stage, frame, info, m, items, baseline, camera, ready, countdown, cameras, current))
        if key == 27:
            return None
        if cameras and ord("1") <= key <= ord("9") and key - ord("1") < len(cameras):
            index = cameras[key - ord("1")][0]
            if index != current:
                return {"switch_camera": index}
        if key == 32 and m is not None:
            issues = impacts(m, items, baseline)
            if not issues:
                return dict(m)
            # The user's choice: start anyway, after seeing what each unmet check costs.
            if confirm_start(stage, issues):
                return dict(m, warnings=[label for label, _, _ in issues], expected_accuracy=rating(issues))
            good_since, countdown_start, good = None, None, []
