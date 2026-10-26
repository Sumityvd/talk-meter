"""Webcam eye tracker that drives GazePointer's Network sensor or the Windows cursor.

Pipeline per camera frame (cnn engine, the default):
  MediaPipe FaceLandmarker (478 landmarks incl. irises)
  -> gaze_cnn.py: head pose, normalized face patch, ETH-XGaze CNN -> 3D gaze ray -> screen point
  -> personal correction to that point (ridge on eye-in-head features, head position, CNN embedding
     and eye-pixel components), fitted at calibration
  -> fixation lock + glide -> UDP "x,y" (physical screen pixels) or the cursor.

Each start: positioning window (live view, camera choice) -> quick ~2 min calibration -> streaming.

Usage:
  .venv\\Scripts\\python.exe webcam_gaze.py --output cursor                      # quick calibration, then go
  .venv\\Scripts\\python.exe webcam_gaze.py --output cursor --calibration full   # ~6 min incl. accuracy test
  .venv\\Scripts\\python.exe webcam_gaze.py --list-cameras
Keys in the status window: q quit, p position check, r recalibrate (quick), v accuracy test,
c or Ctrl+Alt+G drift fix.
"""

import argparse
import base64
import ctypes
import ctypes.wintypes
import json
import math
import os
import re
import socket
import sys
import threading
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

import positioning

HERE = Path(__file__).resolve().parent   # bundled resources (models/); inside the .exe build too
FROZEN = getattr(sys, "frozen", False)   # running as the PyInstaller-built WebcamGaze.exe
# Writable files (calibrations, benchmark results, error log): next to the scripts when run from source,
# in the user's local app data for the .exe (its own folder may be read-only, e.g. under Program Files).
DATA_DIR = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "WebcamGaze" if FROZEN else HERE
MODEL_PATH = HERE / "models" / "face_landmarker.task"
PROFILE_PATH = DATA_DIR / "profiles" / "calibration.json"
SAMPLES_PATH = DATA_DIR / "profiles" / "calibration_samples.npz"

# Corner a is the image-left corner, b the image-right one (camera frame, not mirrored).
EYES = (
    {"a": 33, "b": 133, "top": 159, "bottom": 145},   # subject's right eye
    {"a": 362, "b": 263, "top": 386, "bottom": 374},  # subject's left eye
)
IRISES = ((468, 469, 470, 471, 472), (473, 474, 475, 476, 477))
LOOK_BLENDSHAPES = (
    "eyeLookDownLeft", "eyeLookInLeft", "eyeLookOutLeft", "eyeLookUpLeft",
    "eyeLookDownRight", "eyeLookInRight", "eyeLookOutRight", "eyeLookUpRight",
)
BLINK_THRESHOLD = 0.45

CALIBRATION_GRID = (6, 4)
HEAD_MOTION_GRID = (4, 3)
# Guided head movement in calibration part 2: (instruction, kind, amount: degrees or cm). Turns and tilts
# accept either direction; the amounts are what a person comfortably does while still reading a screen.
HEAD_MOVES = (
    ("Turn your head to the LEFT", "turn", 8.0),
    ("Lean CLOSER to the screen", "closer", 5.0),
    ("Tilt your chin UP", "tilt", 6.0),
    ("Shift your head to the RIGHT (don't turn it)", "shift", 4.0),
    ("Turn your head to the RIGHT", "turn", 8.0),
    ("Lean BACK from the screen", "back", 5.0),
    ("Tilt your chin DOWN", "tilt", 6.0),
    ("Shift your head to the LEFT (don't turn it)", "shift", 4.0),
)
HEAD_MOVE_TIMEOUT_SECONDS = 8.0   # per dot; a dot whose movement never happens is skipped
FEATURE_Z_CLIP = 3.0      # live features are clipped to +-3 SD of the calibration data
# Smallest SD used to standardize each feature: natural variation the model must tolerate even if
# the calibration happened to see less (eye ratios, lid opening, yaw/pitch/roll rad, tx/ty/tz cm,
# eyeLook blendshapes).
FEATURE_SCALE_FLOOR = np.array([0.005, 0.005, 0.01] * 2 + [0.03] * 3 + [0.5] * 3 + [0.03] * 8)
EDGE_HOLD_PX = 150        # gaze this far past a screen edge is held at the edge; further = off-screen
CALIBRATION_MARGIN = 0.06
MOVE_SECONDS = 0.45
SETTLE_SECONDS = 0.55
COLLECT_SECONDS = 1.3
MIN_SAMPLES = 10
EYE_FEATURES = 6          # leading feature columns that describe the eyes (vs. head pose)
FRAME_SECONDS = 1 / 30
GLIDE_MS = 25             # cursor ease time constant (see --glide-ms)
POINTER_DIAMETER = 60     # gaze pointer size in physical pixels (about 1 cm on a 15.6" 1080p screen)
FEEDBACK_GAIN = 0.3       # share of a click-revealed error applied to the drift offset
MAX_FEEDBACK_PX = 200     # larger click-revealed errors are treated as unintended clicks
MAX_OFFSET_PX = 300
CAMERA_WARMUP_SECONDS = 4.0
# Laptop cameras built into the lid; --camera auto prefers any other (external USB) webcam.
BUILTIN_CAMERA = re.compile(r"user facing|integrated|built-?in|internal|front camera|ir camera", re.IGNORECASE)
# Horizontal field of view by device name (part, lower case); others use DEFAULT_HFOV_DEG.
KNOWN_HFOV_DEG = {"lifecam hd-6000": 59.0}   # 66 deg diagonal at 16:9
DEFAULT_HFOV_DEG = 65.0
DARK_FRAME_MEAN = 15         # mean brightness (0-255) below which a face can't be found
GAZE_TIMEOUT_SECONDS = 0.5   # longer than a blink: no usable gaze for this long releases the cursor
STATUS_LOG_SECONDS = 10
ANNOUNCE_SECONDS = 5         # instruction screens between calibration parts continue by themselves
BENCH_DIR = DATA_DIR / "bench"
BENCH_TIMING = (0.45, 0.8, 1.5)   # move, settle, collect seconds per dot (benchmark.py)
BENCH_PASSES = (("still", (4, 4, 0.1)), ("moved", (3, 3, 0.15)))   # (columns, rows, margin)
BENCH_LABEL = "ours-calibration"
PURSUIT_SECONDS = 30
# Calibration lengths. A calibration hardly carries over to a new sitting (2026-10-23 recordings: saved
# calibration + drift dot 152 px, 40 s top-up 90 px, full fresh calibration 59 px), so a quick fresh
# calibration is the default at every start.
CALIBRATION_MODES = {
    "quick": {"grid": (4, 4), "head_dots": 6, "pursuit_seconds": 20, "benchmark": False},   # ~2 min
    "full": {"grid": CALIBRATION_GRID, "head_dots": 12, "pursuit_seconds": PURSUIT_SECONDS, "benchmark": True},
}
PURSUIT_LEAD_SECONDS = 1.5   # dot rests at its start so the eyes can find it
PURSUIT_LAG_SECONDS = 0.08   # camera exposure-to-delivery latency; frames show the eyes this much earlier
PURSUIT_PERIODS = (13.0, 9.0)


# --------------------------------------------------------- stabilization ----

class FixationStabilizer:
    """Holds the cursor at the running mean of the current fixation and jumps only on saccades.

    A sample within `radius` of the fixation centre refines the centre (mean over the last
    `window` frames, ~0.4 s). A saccade is accepted once `confirm` consecutive samples agree on
    a new location.
    """

    def __init__(self, radius, confirm=2, window=12):
        self.radius, self.confirm, self.window = radius, confirm, window
        self.reset()

    def reset(self):
        self.center, self.count, self.pending = None, 0, []

    def __call__(self, point):
        if self.center is None:
            self.center, self.count = point.astype(float), 1
            return self.center
        if np.linalg.norm(point - self.center) <= self.radius:
            self.pending.clear()
            self.count = min(self.count + 1, self.window)
            self.center = self.center + (point - self.center) / self.count
            return self.center
        self.pending.append(point)
        recent = np.array(self.pending[-self.confirm:])
        if len(recent) == self.confirm and np.all(np.linalg.norm(recent - recent.mean(axis=0), axis=1) <= self.radius):
            self.center, self.count = recent.mean(axis=0), self.confirm
            self.pending.clear()
        return self.center


class CursorGlide:
    """Eases the displayed cursor toward the stabilized gaze instead of teleporting."""

    def __init__(self, tau_seconds):
        self.tau = tau_seconds
        self.position = None

    def reset(self):
        self.position = None

    def step(self, target, dt):
        if self.position is None:
            self.position = np.array(target, float)
        else:
            self.position = self.position + (target - self.position) * (1 - math.exp(-dt / self.tau))
        return self.position


class SystemCursor:
    """Moves the Windows cursor (physical pixels). Real mouse movement pauses gaze for a second."""

    PAUSE_SECONDS = 1.0

    def __init__(self):
        self._last = None
        self._paused_until = 0.0

    def __call__(self, position):
        point = ctypes.wintypes.POINT()
        ctypes.windll.user32.GetCursorPos(ctypes.byref(point))
        now = time.monotonic()
        if self._last is not None and max(abs(point.x - self._last[0]), abs(point.y - self._last[1])) > 2:
            self._paused_until = now + self.PAUSE_SECONDS
        if now < self._paused_until:
            self._last = (point.x, point.y)
            return
        self._last = (int(round(position[0])), int(round(position[1])))
        ctypes.windll.user32.SetCursorPos(*self._last)

    def close(self):
        pass


class UdpOutput:
    """GazePointer's Network sensor: "x,y" datagrams in physical screen pixels."""

    def __init__(self, address):
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._address = address

    def __call__(self, position):
        self._socket.sendto(f"{position[0]:.1f},{position[1]:.1f}".encode("ascii"), self._address)

    def close(self):
        self._socket.close()


class GazeOverlay:
    """A circular gaze pointer drawn on top of everything; the mouse cursor stays the user's.

    A small borderless, always-on-top Tk window holding a ring, moved to the gaze point. It is
    click-through and never takes focus (WS_EX_TRANSPARENT | WS_EX_NOACTIVATE), and is hidden while there
    is no gaze. Tk runs in its own thread; callers only hand over positions."""

    KEY = "#ff00ff"                       # colour keyed out as transparent
    RING, EDGE, DOT = "#ff9a1a", "#202020", "#ff9a1a"

    def __init__(self, diameter):
        self.diameter = int(diameter)
        self._position = None
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def __call__(self, position):
        self._position = (int(round(position[0])), int(round(position[1])))

    def hide(self):
        self._position = None

    def close(self):
        # Wait for the Tk thread to tear Tk down itself: if the program exits first, Tcl's interpreter is
        # deleted from the main thread and Tcl aborts the process ("deleted by the wrong thread").
        self._running = False
        self._thread.join(timeout=10)

    def _run(self):
        import tkinter as tk
        d = self.diameter
        root = tk.Tk()
        root.overrideredirect(True)
        root.attributes("-topmost", True)
        root.attributes("-transparentcolor", self.KEY)
        root.configure(bg=self.KEY)
        canvas = tk.Canvas(root, width=d, height=d, bg=self.KEY, highlightthickness=0)
        canvas.pack()
        ring = max(3, d // 15)
        canvas.create_oval(2, 2, d - 2, d - 2, outline=self.EDGE, width=ring + 4)     # dark edge: visible on light
        canvas.create_oval(2, 2, d - 2, d - 2, outline=self.RING, width=ring)         # orange ring: visible on dark
        dot = max(2, d // 20)
        canvas.create_oval(d / 2 - dot, d / 2 - dot, d / 2 + dot, d / 2 + dot, fill=self.DOT, outline=self.EDGE)
        root.update_idletasks()
        user32 = ctypes.windll.user32
        hwnd = user32.GetParent(root.winfo_id()) or root.winfo_id()
        GWL_EXSTYLE, WS_EX_LAYERED, WS_EX_TRANSPARENT = -20, 0x80000, 0x20
        WS_EX_TOOLWINDOW, WS_EX_NOACTIVATE, WS_EX_TOPMOST = 0x80, 0x8000000, 0x8
        style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW
                              | WS_EX_NOACTIVATE | WS_EX_TOPMOST)
        root.withdraw()
        state = {"shown": False}

        def tick():
            if not self._running:
                root.destroy()
                return
            position = self._position
            if position is None:
                if state["shown"]:
                    root.withdraw()
                    state["shown"] = False
            else:
                root.geometry(f"{d}x{d}+{position[0] - d // 2}+{position[1] - d // 2}")
                if not state["shown"]:
                    root.deiconify()
                    root.attributes("-topmost", True)
                    state["shown"] = True
            root.after(15, tick)

        root.after(15, tick)
        root.mainloop()
        # Tk objects must be freed in the thread that created them, or Tcl aborts at interpreter exit.
        tick = canvas = root = None
        import gc
        gc.collect()


class GazeSender(threading.Thread):
    """Sends the gliding cursor to its output at a fixed rate, independent of the camera's 30 fps."""

    def __init__(self, output, tau_seconds, rate_hz=60):
        super().__init__(daemon=True)
        self._output = output
        self._glide = CursorGlide(tau_seconds)
        self._interval = 1.0 / rate_hz
        self._lock = threading.Lock()
        self._target = None
        self._running = True
        self.position = None

    def set_target(self, target):
        with self._lock:
            self._target = None if target is None else np.array(target, float)

    def run(self):
        last = time.monotonic()
        while self._running:
            time.sleep(self._interval)
            now = time.monotonic()
            dt, last = now - last, now
            with self._lock:
                target = self._target
            if target is None:
                if self.position is not None and hasattr(self._output, "hide"):
                    self._output.hide()
                self._glide.reset()
                self.position = None
                continue
            position = self._glide.step(target, dt)
            self.position = position
            self._output(position)

    def stop(self):
        self._running = False
        self.join(timeout=1)
        self._output.close()


def stabilizer_radius(precision_px):
    return float(np.clip(1.8 * precision_px, 60, 180)) if precision_px else 120.0


def target_correction(cursor, bounds, inset=0.3):
    """Vector moving `cursor` into the inner part of an invoked target (zero if already there).

    Users fixate the middle area of a button, so a click reached from outside that area
    reveals calibration bias.
    """
    left, top, right, bottom = bounds
    dx, dy = (right - left) * inset, (bottom - top) * inset
    inner = np.array([np.clip(cursor[0], left + dx, right - dx), np.clip(cursor[1], top + dy, bottom - dy)])
    return inner - cursor


# ---------------------------------------------------------------- screen ----

def screen_info():
    """Physical pixel size of the primary screen and its size in millimetres."""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        ctypes.windll.user32.SetProcessDPIAware()
    user32, gdi32 = ctypes.windll.user32, ctypes.windll.gdi32
    width, height = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    dc = user32.GetDC(0)
    width_mm, height_mm = gdi32.GetDeviceCaps(dc, 4), gdi32.GetDeviceCaps(dc, 6)
    user32.ReleaseDC(0, dc)
    return width, height, width_mm, height_mm


# ---------------------------------------------------------------- camera ----

def open_camera(index):
    # Media Foundation delivers 30 fps at 720p on the built-in camera, where DirectShow falls back
    # to uncompressed YUY2 at ~8 fps. Some USB cameras (LifeCam HD-6000) intermittently stream pure
    # black right after opening; wait that out, then try the other backend.
    for backend in (cv2.CAP_MSMF, cv2.CAP_DSHOW):
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # newest frame only; queued frames are pure latency
        deadline = time.monotonic() + CAMERA_WARMUP_SECONDS
        while time.monotonic() < deadline:
            ok, frame = cap.read()
            if ok and frame.mean() > 1.0:
                if frame.mean() < DARK_FRAME_MEAN:
                    print(f"warning: camera {index} picture is very dark (brightness {frame.mean():.0f}/255) - "
                          "lens covered, camera facing away, or room too dark? Faces will not be found.", flush=True)
                return cap
        cap.release()
    return None


def camera_names():
    """{index: device name}, empty if cv2-enumerate-cameras is missing."""
    try:
        from cv2_enumerate_cameras import enumerate_cameras
        return {c.index: c.name for c in enumerate_cameras(cv2.CAP_MSMF)}
    except ImportError:
        return {}


def is_builtin_camera(name):
    return bool(BUILTIN_CAMERA.search(name or ""))


def resolve_camera(spec):
    """Camera index from an index string, a case-insensitive part of the device name, or "auto"
    (the first external webcam; the laptop's built-in camera only if there is no other)."""
    if spec.isdigit():
        return int(spec)
    names = camera_names()
    if not names:
        if spec == "auto":
            return 0
        sys.exit("camera names need cv2-enumerate-cameras (pip install cv2-enumerate-cameras); "
                 "or pass an index from --list-cameras")
    if spec == "auto":
        external = [i for i, name in sorted(names.items()) if not is_builtin_camera(name)]
        index = external[0] if external else min(names)
        kind = "external webcam" if external else "no external webcam found, using the built-in camera"
        print(f"camera auto: {kind} - index {index}: {names[index]}")
        return index
    matches = [i for i, name in names.items() if spec.lower() in name.lower()]
    if len(matches) != 1:
        found = ", ".join(f"{i}: {n}" for i, n in names.items())
        sys.exit(f"'{spec}' matches {len(matches)} cameras ({found})")
    print(f"camera '{spec}' is index {matches[0]}: {names[matches[0]]}")
    return matches[0]


class CameraSession:
    """The open camera with its per-camera settings and feature extractor. The positioning window can
    switch it to another camera during calibration (the calibration is saved for that camera)."""

    def __init__(self, engine, screen, hfov_override=None, above_override=None):
        self.engine, self.screen = engine, screen
        self.hfov_override, self.above_override = hfov_override, above_override
        self.cap = self.extract = self.index = None
        self.name, self.hfov, self.camera_above = "", DEFAULT_HFOV_DEG, 8.0
        self.names = camera_names()

    def open(self, index):
        """Opens camera `index` (keeping the current one if that fails); returns True on success."""
        cap = open_camera(index)
        if cap is None:
            print(f"camera {index} did not open or only sends black frames", flush=True)
            return False
        if self.cap is not None:
            self.cap.release()
        self.cap, self.index = cap, index
        self.name = self.names.get(index, f"camera {index}")
        self.hfov = self.hfov_override or next(
            (v for k, v in KNOWN_HFOV_DEG.items() if k in self.name.lower()), DEFAULT_HFOV_DEG)
        self.camera_above = self.above_override or (8.0 if is_builtin_camera(self.name) else 15.0)
        if self.engine == "cnn":
            self.extract = CnnFeatureExtractor(self.screen, self.hfov, self.camera_above)
        else:
            self.extract = FeatureExtractor()
        print(f"camera {index}: {self.name} - field of view {self.hfov:g} deg, "
              f"lens {self.camera_above:g} mm above the screen", flush=True)
        return True

    def options(self):
        """[(index, label)] of connected cameras for the positioning window."""
        return [(i, f"{n} (built-in)" if is_builtin_camera(n) else n) for i, n in sorted(self.names.items())] \
            or [(self.index, self.name)]

    def release(self):
        if self.cap is not None:
            self.cap.release()


def list_cameras():
    names = camera_names()
    for index in range(6):
        cap = open_camera(index)
        if cap is None:
            if index in names:
                print(f"camera {index}: {names[index]} - no picture (covered or in use?)")
            continue
        ok, frame = cap.read()
        if ok:
            print(f"camera {index}: {frame.shape[1]}x{frame.shape[0]} @ {cap.get(cv2.CAP_PROP_FPS):.0f} fps"
                  f"  {names.get(index, '')}")
        cap.release()


# -------------------------------------------------------------- features ----

class FeatureExtractor:
    def __init__(self):
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ms = 0

    def __call__(self, frame):
        """Returns (features, info) or (None, info). info has 'face', 'blink', 'points', 'distance_cm'."""
        height, width = frame.shape[:2]
        now_ms = max(int(time.monotonic() * 1000), self._last_ms + 1)
        self._last_ms = now_ms
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = self._landmarker.detect_for_video(image, now_ms)

        info = {"face": False, "blink": False, "points": None, "distance_cm": None}
        if not result.face_landmarks:
            return None, info
        info["face"] = True

        points = np.array([(p.x * width, p.y * height) for p in result.face_landmarks[0]])
        info["points"] = points
        scores = {c.category_name: c.score for c in result.face_blendshapes[0]}
        if max(scores.get("eyeBlinkLeft", 0), scores.get("eyeBlinkRight", 0)) > BLINK_THRESHOLD:
            info["blink"] = True
            return None, info

        iris_centers = [points[list(ring)].mean(axis=0) for ring in IRISES]
        eye_features = []
        for eye in EYES:
            a, b = points[eye["a"]], points[eye["b"]]
            middle = (a + b) / 2
            # Pair each eye with the iris nearest to it rather than trusting index order.
            iris = min(iris_centers, key=lambda c: np.linalg.norm(c - middle))
            axis = b - a
            eye_width = np.linalg.norm(axis)
            u = axis / eye_width
            v = np.array([-u[1], u[0]])
            offset = iris - middle
            opening = abs((points[eye["top"]] - points[eye["bottom"]]) @ v) / eye_width
            eye_features += [offset @ u / eye_width, offset @ v / eye_width, opening]

        matrix = np.asarray(result.facial_transformation_matrixes[0])
        rotation, translation = matrix[:3, :3], matrix[:3, 3]
        yaw = math.atan2(rotation[0, 2], rotation[2, 2])
        pitch = math.asin(max(-1.0, min(1.0, -rotation[1, 2])))
        roll = math.atan2(rotation[1, 0], rotation[1, 1])
        info["distance_cm"] = abs(float(translation[2]))

        features = eye_features + [yaw, pitch, roll, *translation] + [scores.get(n, 0.0) for n in LOOK_BLENDSHAPES]
        return np.array(features, dtype=np.float64), info


class CnnFeatureExtractor:
    """Landmark features plus the CNN engine (gaze_cnn.py).

    Feature layout: [0:20] landmark features, [20:22] geometric CNN gaze point (px),
    [22:24] CNN pitch/yaw (rad), [24:2072] CNN embedding, [2072:6392] normalized eye pixels (2 x 60x36),
    [6392:6398] gaze ray in camera space: origin (face centre, mm) and unit direction.
    """

    def __init__(self, screen, hfov_deg, camera_above_mm):
        import gaze_cnn
        self._gaze_cnn = gaze_cnn
        self._base = FeatureExtractor()
        width, height, width_mm, height_mm = screen[:4]
        self._geometry = gaze_cnn.ScreenGeometry(width, height, width_mm, height_mm, camera_above_mm)
        self._hfov = hfov_deg
        self._cnn = None

    def __call__(self, frame):
        features, info = self._base(frame)
        if features is None:
            return None, info
        if self._cnn is None:
            # Mirror averaging cut frame-to-frame noise ~30% and light landmark smoothing helped
            # calibrated accuracy in a same-frames A/B on a LifeCam HD-6000.
            self._cnn = self._gaze_cnn.CnnGaze((frame.shape[1], frame.shape[0]), self._hfov, self._geometry,
                                               smoothing=0.5, flip=True)
            print(f"CNN gaze running on {self._cnn.provider}", flush=True)
        result = self._cnn(frame, info["points"])
        if result is None or not np.all(np.isfinite(result["screen_px"])):
            info["reject"] = "head pose" if result is None else "gaze ray misses screen plane"
            return None, info
        info["cnn_px"] = result["screen_px"]
        # PnP with the configured field of view measures distance far better than MediaPipe's generic camera.
        info["distance_cm"] = float(result["face_center"][2]) / 10
        return np.concatenate([features, result["screen_px"], result["pitchyaw"], result["embedding"],
                               result["eyes"], result["face_center"], result["direction"]]), info


# ----------------------------------------------------------------- model ----

class GazeModel:
    """Standardize -> polynomial expansion -> ridge regression to screen pixels."""

    def __init__(self, degree, ridge, mean, scale, weights):
        self.degree, self.ridge = degree, ridge
        self.mean, self.scale, self.weights = mean, scale, weights

    @staticmethod
    def expand(z, degree):
        """degree 1: linear; 1.5: linear + quadratic eye terms; 2: full quadratic."""
        columns = [np.ones((z.shape[0], 1)), z]
        if degree >= 2:
            i, j = np.triu_indices(z.shape[1])
            columns.append(z[:, i] * z[:, j])
        elif degree > 1:
            i, j = np.triu_indices(EYE_FEATURES)
            columns.append(z[:, i] * z[:, j])
        return np.hstack(columns)

    @classmethod
    def fit(cls, x, y, degree, ridge):
        mean = x.mean(axis=0)
        scale = np.maximum(x.std(axis=0), FEATURE_SCALE_FLOOR)
        phi = cls.expand((x - mean) / scale, degree)
        penalty = ridge * np.eye(phi.shape[1])
        penalty[0, 0] = 0.0
        weights = np.linalg.solve(phi.T @ phi + penalty, phi.T @ y)
        return cls(degree, ridge, mean, scale, weights)

    def predict(self, x):
        # Clip to the calibrated range: a feature the calibration barely saw vary (e.g. head
        # distance, eyeLookUp) would otherwise get a huge z-score and fling the cursor off-screen.
        z = np.clip((np.atleast_2d(x) - self.mean) / self.scale, -FEATURE_Z_CLIP, FEATURE_Z_CLIP)
        return self.expand(z, self.degree) @ self.weights

    @classmethod
    def select(cls, x, y, groups):
        """Leave-one-target-out CV over degree and ridge strength; returns (model, cv_error_px)."""
        best = None
        for degree in (1, 1.5, 2):
            for ridge in np.logspace(-2, 4, 13):
                errors = []
                for g in np.unique(groups):
                    held = groups == g
                    model = cls.fit(x[~held], y[~held], degree, ridge)
                    predicted = model.predict(x[held]).mean(axis=0)
                    errors.append(np.linalg.norm(predicted - y[held][0]))
                error = float(np.mean(errors))
                if best is None or error < best[0]:
                    best = (error, degree, ridge)
        error, degree, ridge = best
        return cls.fit(x, y, degree, ridge), error

    def refit(self, x, y):
        return GazeModel.fit(x, y, self.degree, self.ridge)

    def describe(self):
        return f"degree={self.degree} ridge={self.ridge:.3g}"

    def to_json(self):
        return {"degree": self.degree, "ridge": self.ridge, "mean": self.mean.tolist(),
                "scale": self.scale.tolist(), "weights": self.weights.tolist()}

    @classmethod
    def from_json(cls, data):
        return cls(data["degree"], data["ridge"], np.array(data["mean"]),
                   np.array(data["scale"]), np.array(data["weights"]))


CNN_BASE = 24             # landmark features + CNN point + CNN pitch/yaw; the PCA blocks follow
CNN_FEATURES_VERSION = 6  # bump when the features or CnnGazeModel's format change; old calibrations are redone
# Base features the CNN model uses: eye-in-head (iris offsets, lid opening) and the CNN point and
# pitch/yaw. MediaPipe head pose and eyeLook blendshapes are left out: the CNN geometry already
# handles the head, and a 2 cm change in sitting distance (8 calibration SDs of MediaPipe's tz)
# moved predictions ~250 px when they were included.
# Plus the gaze ray from the CNN engine: the head's 3D position (PnP face centre, mm) and the gaze
# direction. Metric head position let the correction follow head movement (2026-10-23 session with
# guided head movement: 83/58 -> 63/41 px head still/moved).
MODEL_BASE_COLUMNS = np.r_[0:EYE_FEATURES, 20:24, 6392:6398]
MODEL_BASE_FLOOR = np.concatenate([FEATURE_SCALE_FLOOR[:EYE_FEATURES], [20.0, 20.0, 0.01, 0.01],
                                   [2.0] * 3, [0.005] * 3])
MODEL_POINT = slice(EYE_FEATURES, EYE_FEATURES + 2)   # CNN point within the model's base columns
# High-dimensional feature blocks, each reduced by PCA; CV picks how many components to keep.
PCA_BLOCKS = {"embedding": slice(24, 2072), "eyes": slice(2072, 6392)}
# Fixed personal-model settings (no per-calibration search). On sessions recorded 2026-10-23 (LifeCam),
# within-session CV kept picking settings that did worse on the benchmark recorded right after
# (embedding 0: 116/120 px head still/moved) than these (47/46 px as a correction to the CNN point).
MODEL_COMPONENTS = {"embedding": 32, "eyes": 16}
MODEL_RIDGE = 100.0


def encode_array(a):
    return {"shape": list(a.shape), "float32": base64.b64encode(np.ascontiguousarray(a, np.float32)).decode("ascii")}


def decode_array(d):
    return np.frombuffer(base64.b64decode(d["float32"]), np.float32).astype(np.float64).reshape(d["shape"])


class CnnGazeModel:
    """Personal calibration on top of the CNN: a frozen backbone with a ridge head.

    The prediction is the CNN's geometric screen point plus a learned personal correction
    (`residual`). The geometric point follows head movement physically; the correction removes this
    person's and this camera's systematic error. Learning the screen position from scratch instead
    made head movement throw the cursor off (2026-10-23: 55/61 px head still/moved, against 47/46 px
    as a correction).

    Correction inputs: eye-in-head landmark features, the CNN point and pitch/yaw, and principal
    components of the CNN embedding and of the normalized eye pixels. The CNN point also gets quadratic
    terms, so the correction can bend the geometry error away.
    """

    def __init__(self, components, ridge, pcas, mean, scale, weights, residual=True):
        self.components, self.ridge = components, ridge   # components: {block: k}
        self.pcas = pcas                                   # {block: (mean, basis[:, k])}
        self.mean, self.scale, self.weights = mean, scale, weights
        self.residual = residual                           # False: profiles saved before 2026-10-23 evening

    @staticmethod
    def fit_pcas(x, components):
        pcas = {}
        for name, columns in PCA_BLOCKS.items():
            block = x[:, columns]
            center = block.mean(axis=0)
            k = min(components[name], *block.shape)
            if k == 0:
                pcas[name] = (center, np.zeros((block.shape[1], 0)))
                continue
            _, _, vt = np.linalg.svd(block - center, full_matrices=False)
            pcas[name] = (center, vt[:k].T)
        return pcas

    @staticmethod
    def reduce(x, pcas):
        x = np.atleast_2d(x)
        parts = [x[:, MODEL_BASE_COLUMNS]]
        for name, columns in PCA_BLOCKS.items():
            center, basis = pcas[name]
            parts.append((x[:, columns] - center) @ basis)
        return np.hstack(parts)

    @staticmethod
    def expand(z):
        point = z[:, MODEL_POINT]
        quadratic = np.column_stack([point[:, 0] ** 2, point[:, 0] * point[:, 1], point[:, 1] ** 2])
        return np.hstack([np.ones((z.shape[0], 1)), z, quadratic])

    @classmethod
    def fit_reduced(cls, r, y, ridge, pcas, components):
        mean = r.mean(axis=0)
        floor = np.concatenate([MODEL_BASE_FLOOR, np.full(r.shape[1] - len(MODEL_BASE_COLUMNS), 1e-3)])
        scale = np.maximum(r.std(axis=0), floor)
        phi = cls.expand((r - mean) / scale)
        penalty = ridge * np.eye(phi.shape[1])
        penalty[0, 0] = 0.0
        weights = np.linalg.solve(phi.T @ phi + penalty, phi.T @ (y - r[:, MODEL_POINT]))
        return cls(components, ridge, pcas, mean, scale, weights)

    @classmethod
    def fit(cls, x, y, components=None, ridge=None):
        components = components or MODEL_COMPONENTS
        ridge = MODEL_RIDGE if ridge is None else ridge
        pcas = cls.fit_pcas(x, components)
        return cls.fit_reduced(cls.reduce(x, pcas), y, ridge, pcas, components)

    def predict_reduced(self, r):
        z = np.clip((r - self.mean) / self.scale, -FEATURE_Z_CLIP, FEATURE_Z_CLIP)
        correction = self.expand(z) @ self.weights
        return correction + r[:, MODEL_POINT] if self.residual else correction

    def predict(self, x):
        return self.predict_reduced(self.reduce(x, self.pcas))

    @classmethod
    def select(cls, x, y, groups):
        """Fits the fixed configuration (MODEL_COMPONENTS, MODEL_RIDGE) and reports its
        leave-one-target-out error. No search: within-session CV kept choosing settings that did worse
        on the benchmark that follows (see MODEL_COMPONENTS). Smooth-pursuit samples (groups < 0) always
        train; PCA is computed once on all samples (unsupervised, so no label leakage)."""
        pcas = cls.fit_pcas(x, MODEL_COMPONENTS)
        r = cls.reduce(x, pcas)
        errors = []
        for g in np.unique(groups[groups >= 0]):
            held = groups == g
            model = cls.fit_reduced(r[~held], y[~held], MODEL_RIDGE, pcas, MODEL_COMPONENTS)
            errors.append(np.linalg.norm(model.predict_reduced(r[held]).mean(axis=0) - y[held][0]))
        return cls.fit_reduced(r, y, MODEL_RIDGE, pcas, MODEL_COMPONENTS), float(np.mean(errors))

    def refit(self, x, y):
        return CnnGazeModel.fit(x, y, self.components, self.ridge)

    def describe(self):
        return f"{'correction to the CNN point, ' if self.residual else ''}embedding components=" \
               f"{self.components['embedding']} eye components={self.components['eyes']} ridge={self.ridge:.3g}"

    def to_json(self):
        return {"kind": "cnn", "features": CNN_FEATURES_VERSION, "residual": self.residual,
                "components": self.components, "ridge": self.ridge,
                "pcas": {name: {"mean": encode_array(center), "basis": encode_array(basis)}
                         for name, (center, basis) in self.pcas.items()},
                "mean": self.mean.tolist(), "scale": self.scale.tolist(), "weights": self.weights.tolist()}

    @classmethod
    def from_json(cls, data):
        pcas = {name: (decode_array(p["mean"]), decode_array(p["basis"])) for name, p in data["pcas"].items()}
        return cls(data["components"], data["ridge"], pcas, np.array(data["mean"]), np.array(data["scale"]),
                   np.array(data["weights"]), residual=data.get("residual", False))


def model_from_json(data):
    return CnnGazeModel.from_json(data) if data.get("kind") == "cnn" else GazeModel.from_json(data)


# --------------------------------------------------------------- display ----

class Stage:
    """Full-screen black canvas for calibration and validation targets."""

    NAME = "webcam-gaze calibration"

    def __init__(self, width, height):
        self.width, self.height = width, height
        cv2.namedWindow(self.NAME, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setWindowProperty(self.NAME, cv2.WND_PROP_TOPMOST, 1)

    def blank(self):
        return np.zeros((self.height, self.width, 3), np.uint8)

    def text(self, canvas, lines, y=None, scale=1.0):
        y = y or self.height // 3
        for line in lines:
            (w, h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
            cv2.putText(canvas, line, ((self.width - w) // 2, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                        (230, 230, 230), 2, cv2.LINE_AA)
            y += int(h * 2.2)

    def show(self, canvas, wait_ms=1):
        cv2.imshow(self.NAME, canvas)
        return cv2.waitKey(wait_ms) & 0xFF

    def close(self):
        cv2.destroyWindow(self.NAME)


def draw_target(canvas, center, progress):
    x, y = int(center[0]), int(center[1])
    radius = int(28 - 18 * progress)
    cv2.circle(canvas, (x, y), radius, (60, 200, 255), -1, cv2.LINE_AA)
    cv2.circle(canvas, (x, y), 4, (0, 0, 0), -1, cv2.LINE_AA)


def wait_for_space(stage, lines):
    canvas = stage.blank()
    stage.text(canvas, lines)
    while True:
        key = stage.show(canvas, 30)
        if key == 32:
            return True
        if key == 27:
            return False


def announce(stage, lines, seconds=ANNOUNCE_SECONDS):
    """Instruction screen that continues by itself after a countdown (SPACE: now, ESC: skip/cancel)."""
    end = time.monotonic() + seconds
    while (remaining := end - time.monotonic()) > 0:
        canvas = stage.blank()
        stage.text(canvas, lines + ["", f"Starting in {math.ceil(remaining)} ...      SPACE: start now    ESC: skip"])
        key = stage.show(canvas, 30)
        if key == 32:
            return True
        if key == 27:
            return False
    return True


def position_user(session, screen, baseline=None):
    """Positioning window (positioning.py) on its own full-screen stage; returns the posture or None.
    The camera is fixed here (the saved calibration belongs to it); recalibrate to change camera."""
    stage = Stage(*screen[:2])
    try:
        return positioning.position_check(session.cap, session.extract, stage, session.hfov, baseline,
                                          cameras=[(session.index, session.name)], current=session.index)
    finally:
        stage.close()


def collect_targets(cap, extract, stage, targets, timing=(MOVE_SECONDS, SETTLE_SECONDS, COLLECT_SECONDS),
                    trim=True):
    """Show each target and collect features while the user fixates it.

    timing: (move, settle, collect) seconds. trim: drop outlying samples (for training; an accuracy
    test keeps every sample so it is scored like benchmark.py scores any tracker).
    Returns list of (target_px, features[N, F], distances_cm) or None on ESC.
    """
    move_seconds, settle_seconds, collect_seconds = timing
    results = []
    previous = (stage.width / 2, stage.height / 2)
    queue = list(targets)
    retries = {t: 0 for t in targets}
    while queue:
        target = queue.pop(0)
        samples, distances = [], []
        start = time.monotonic()
        total = move_seconds + settle_seconds + collect_seconds
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            elapsed = time.monotonic() - start
            if elapsed >= total:
                break
            features, info = extract(frame)
            if elapsed < move_seconds:
                k = elapsed / move_seconds
                k = k * k * (3 - 2 * k)
                center = (previous[0] + (target[0] - previous[0]) * k, previous[1] + (target[1] - previous[1]) * k)
                progress = 0.0
            else:
                center = target
                progress = (elapsed - move_seconds) / (settle_seconds + collect_seconds)
                if elapsed >= move_seconds + settle_seconds and features is not None:
                    samples.append(features)
                    distances.append(info["distance_cm"])
            canvas = stage.blank()
            draw_target(canvas, center, progress)
            if not info["face"]:
                stage.text(canvas, ["Face not found - look at the screen"], y=stage.height - 60, scale=0.8)
            if stage.show(canvas) == 27:
                return None
        previous = target
        if len(samples) < MIN_SAMPLES and retries[target] < 2:
            retries[target] += 1
            queue.append(target)
            continue
        if len(samples) >= MIN_SAMPLES:
            samples = np.array(samples)
            results.append((np.array(target, float), trim_outliers(samples) if trim else samples, distances))
    return results


def head_move_progress(move, m, baseline, camera):
    """How far (0..1+) the head has moved in the asked way, from positioning.measure() values."""
    kind, amount = move[1], move[2]
    if m is None:
        return 0.0
    if kind == "turn":
        change = abs((m["yaw"] - baseline["yaw"] + 180) % 360 - 180)
    elif kind == "tilt":
        change = abs((m["pitch"] - baseline["pitch"] + 180) % 360 - 180)
    elif kind == "closer":
        change = baseline["distance_cm"] - m["distance_cm"]
    elif kind == "back":
        change = m["distance_cm"] - baseline["distance_cm"]
    else:   # sideways shift, cm at the face: image offset * distance / focal length
        change = abs(m["centre_x"] - baseline["centre_x"]) * 2 * camera[0, 2] * m["distance_cm"] / camera[0, 0]
    return max(0.0, change / amount)


def collect_head_moves(cap, extract, stage, targets, hfov_deg, baseline):
    """Part 2 of calibration: the eyes stay on each dot while the head moves as asked (turn, tilt,
    closer, back, sideways). Samples are only taken once the camera sees the head actually displaced,
    so the model learns real head variety (without this guidance, a recorded session moved < 1 cm).
    Returns [(target, features[N, F], distances)] like collect_targets()."""
    results, previous, camera = [], (stage.width / 2, stage.height / 2), None
    for number, target in enumerate(targets):
        move = HEAD_MOVES[number % len(HEAD_MOVES)]
        samples, distances = [], []
        start, held_since = time.monotonic(), None
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            if camera is None:
                camera = positioning.camera_matrix(frame.shape[1], frame.shape[0], hfov_deg)
            elapsed = time.monotonic() - start
            features, info = extract(frame)
            progress = head_move_progress(move, positioning.measure(frame, info, camera), baseline, camera)
            if elapsed < MOVE_SECONDS:
                k = elapsed / MOVE_SECONDS
                k = k * k * (3 - 2 * k)
                center = (previous[0] + (target[0] - previous[0]) * k, previous[1] + (target[1] - previous[1]) * k)
            else:
                center = target
                if progress >= 1.0 and features is not None:
                    held_since = held_since or time.monotonic()
                    samples.append(features)
                    distances.append(info["distance_cm"])
            collected = len(samples) / max(1.0, COLLECT_SECONDS / FRAME_SECONDS)
            canvas = stage.blank()
            draw_target(canvas, center, min(1.0, collected))
            x, y = int(center[0]), int(center[1])
            color = (90, 220, 90) if progress >= 1.0 else (40, 190, 250)
            cv2.ellipse(canvas, (x, y), (40, 40), -90, 0, 360 * min(1.0, progress), color, 4, cv2.LINE_AA)
            label = move[0] if progress < 1.0 else "Hold it - keep looking at the dot"
            (w, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            tx = int(np.clip(x - w / 2, 20, stage.width - w - 20))
            ty = y + 80 if y < stage.height - 120 else y - 70
            cv2.putText(canvas, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
            if not info["face"]:
                stage.text(canvas, ["Face not found - look at the screen"], y=stage.height - 60, scale=0.8)
            if stage.show(canvas) == 27:
                return None
            if collected >= 1.0 or elapsed > MOVE_SECONDS + HEAD_MOVE_TIMEOUT_SECONDS:
                break
        previous = target
        if len(samples) >= MIN_SAMPLES:
            results.append((np.array(target, float), trim_outliers(np.array(samples)), distances))
    print(f"head movement: {len(results)} of {len(targets)} dots with the head moved as asked", flush=True)
    return results


def pursuit_position(t, width, height):
    """Lissajous path covering the screen; about 6 deg/s at 60 cm, slow enough for smooth pursuit."""
    x = 0.5 + 0.42 * math.sin(2 * math.pi * t / PURSUIT_PERIODS[0])
    y = 0.5 + 0.40 * math.sin(2 * math.pi * t / PURSUIT_PERIODS[1] + math.pi / 2)
    return np.array([x * width, y * height])


def collect_pursuit(cap, extract, stage, seconds):
    """Moving dot the user follows; every frame is labelled with where the dot was when the frame
    was exposed. Returns (features[N, F], targets[N, 2], segment[N]) or None on ESC."""
    samples, targets, segments = [], [], []
    start = time.monotonic()
    while (elapsed := time.monotonic() - start) < PURSUIT_LEAD_SECONDS + seconds:
        ok, frame = cap.read()
        captured = time.monotonic() - start - PURSUIT_LAG_SECONDS
        if not ok:
            continue
        features, info = extract(frame)
        t = max(0.0, elapsed - PURSUIT_LEAD_SECONDS)
        canvas = stage.blank()
        draw_target(canvas, pursuit_position(t, stage.width, stage.height), 0.5)
        if not info["face"]:
            stage.text(canvas, ["Face not found - look at the screen"], y=stage.height - 60, scale=0.8)
        if stage.show(canvas) == 27:
            return None
        if features is not None and captured > PURSUIT_LEAD_SECONDS + 0.3:
            t_captured = captured - PURSUIT_LEAD_SECONDS
            samples.append(features)
            targets.append(pursuit_position(t_captured, stage.width, stage.height))
            segments.append(int(t_captured // 2))
    if len(samples) < 100:
        return None
    return np.array(samples), np.array(targets), np.array(segments)


def trim_outliers(samples, keep=0.8):
    """Drop the samples furthest from the per-target median (late saccades, partial blinks)."""
    median = np.median(samples, axis=0)
    spread = np.maximum(np.median(np.abs(samples - median), axis=0), 1e-6)
    distance = np.linalg.norm((samples - median) / spread, axis=1)
    return samples[distance <= np.quantile(distance, keep)]


def stack(collected):
    x = np.vstack([s for _, s, _ in collected])
    y = np.vstack([np.repeat(t[None, :], len(s), axis=0) for t, s, _ in collected])
    groups = np.concatenate([np.full(len(s), i) for i, (_, s, _) in enumerate(collected)])
    return x, y, groups


def grid_targets(width, height, grid=CALIBRATION_GRID, margin=CALIBRATION_MARGIN, seed=7):
    columns, rows = grid
    xs = np.linspace(margin, 1 - margin, columns) * width
    ys = np.linspace(margin, 1 - margin, rows) * height
    targets = [(float(x), float(y)) for y in ys for x in xs]
    np.random.default_rng(seed).shuffle(targets)
    return targets


# ------------------------------------------------------------ procedures ----

# ------------------------------------------------------------- benchmark ----
# The same test benchmark.py runs on any cursor-moving tracker (e.g. GazeFlow), so every calibration
# ends with a number that compares directly.

def benchmark_grid(width, height, columns, rows, margin):
    xs = np.linspace(margin, 1 - margin, columns) * width
    ys = np.linspace(margin, 1 - margin, rows) * height
    targets = [(float(x), float(y)) for y in ys for x in xs]
    np.random.default_rng(columns * 10 + rows).shuffle(targets)
    return targets


def score_targets(results, mm_per_px, distance_cm):
    """results: [(target, cursor samples[N, 2])] -> accuracy/precision report (benchmark.py format)."""
    def to_degrees(px):
        return math.degrees(math.atan(px * mm_per_px / (distance_cm * 10)))

    accuracy, precision, per_target = [], [], []
    for target, samples in results:
        mean = samples.mean(axis=0)
        a = float(np.linalg.norm(mean - target))
        p = float(np.sqrt(np.mean(np.sum((samples - mean) ** 2, axis=1))))
        accuracy.append(a)
        precision.append(p)
        per_target.append({"target": [round(float(v)) for v in target], "mean_cursor": [round(float(v)) for v in mean],
                           "accuracy_px": round(a), "precision_px": round(p)})
    a, p = float(np.mean(accuracy)), float(np.mean(precision))
    return {"accuracy_px": a, "accuracy_deg": to_degrees(a), "accuracy_cm": a * mm_per_px / 10,
            "precision_px": p, "precision_deg": to_degrees(p), "per_target": per_target}


def save_bench(report):
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    path = BENCH_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}_{report['label']}.json"
    path.write_text(json.dumps(report, indent=1))
    return path


def reference_accuracy(label="gazeflow"):
    """Median head-still / head-moved accuracy (px) of saved benchmark runs with this label, or None."""
    runs = []
    for path in BENCH_DIR.glob(f"*_{label}.json"):
        try:
            runs.append(json.loads(path.read_text()))
        except (OSError, ValueError):
            continue
    if not runs:
        return None
    pick = lambda key: float(np.median([r[key]["accuracy_px"] for r in runs if r.get(key)])) \
        if any(r.get(key) for r in runs) else None
    return {"still": pick("still"), "moved": pick("moved"), "runs": len(runs)}


def with_pursuit(x, y, groups, pursuit):
    """Appends smooth-pursuit samples; their groups are negative so CV never holds them out."""
    if pursuit is None:
        return x, y, groups
    px, py, segments = pursuit
    return np.vstack([x, px]), np.vstack([y, py]), np.concatenate([groups, -1 - segments])


def calibrate(session, stage, engine, mode="quick"):
    """mode: a CALIBRATION_MODES key (dot grid, guided head-move dots, moving-dot seconds)."""
    plan = CALIBRATION_MODES[mode]
    # Positioning, with the camera choice: switching reopens the session and shows the new camera.
    while True:
        posture = positioning.position_check(session.cap, session.extract, stage, session.hfov,
                                             cameras=session.options(), current=session.index)
        if posture is None:
            return None
        if "switch_camera" not in posture:
            break
        session.open(posture["switch_camera"])
    if posture.get("warnings"):
        print(f"calibration started anyway with: {', '.join(posture['warnings'])} "
              f"(expected accuracy {posture['expected_accuracy']})", flush=True)
    cap, extract = session.cap, session.extract
    if not announce(stage, [
        "Calibration",
        "Follow the dot with your EYES and look at its black centre until it shrinks.",
        "Keep your head natural and fairly still.",
    ], seconds=3):
        return None
    collected = collect_targets(cap, extract, stage, grid_targets(stage.width, stage.height, plan["grid"]))
    if not collected or len(collected) < 8:
        return None
    still_targets = len(collected)
    # Without head-pose variety the model only works in the exact posture used for calibration.
    if announce(stage, [
        "Part 2: head movement",
        "Keep your EYES on each dot and move your HEAD the way the text under the dot asks.",
        "The ring fills as you move and turns green when you have moved enough - then hold still a moment.",
    ], seconds=7):
        targets = grid_targets(stage.width, stage.height, HEAD_MOTION_GRID, 0.12, seed=11)[:plan["head_dots"]]
        moving = collect_head_moves(cap, extract, stage, targets, session.hfov, posture)
        collected += moving or []
    pursuit = None
    if engine == "cnn" and announce(stage, [
        "Part 3: follow the moving dot",
        f"Follow the dot smoothly with your eyes for {plan['pursuit_seconds']} seconds.",
    ]):
        pursuit = collect_pursuit(cap, extract, stage, plan["pursuit_seconds"])
    x, y, groups = with_pursuit(*stack(collected), pursuit)
    canvas = stage.blank()
    stage.text(canvas, ["Computing your gaze model..."])
    stage.show(canvas, 30)
    model, cv_error = (CnnGazeModel if engine == "cnn" else GazeModel).select(x, y, groups)
    print(f"calibration: {len(collected)} targets, {len(x)} samples "
          f"({0 if pursuit is None else len(pursuit[0])} pursuit), model {model.describe()}, "
          f"cross-validated error {cv_error:.0f} px")
    return model, collected, still_targets, pursuit, posture


def simulate_cursor(model, samples, precision_px):
    """Cursor positions the tracker would produce for one fixation: model -> fixation lock -> glide."""
    stabilizer = FixationStabilizer(stabilizer_radius(precision_px))
    glide = CursorGlide(GLIDE_MS / 1000)
    return np.array([glide.step(stabilizer(p), FRAME_SECONDS) for p in model.predict(samples)])


def validate(cap, extract, stage, model, screen):
    """Accuracy test with benchmark.py's protocol: 16 dots with the head still, then 9 dots after
    shifting the head ~5 cm. Every sample is kept (no outlier trimming). Scored on the cursor path the
    tracker would produce, so the result compares directly with benchmark.py runs of other trackers.
    Returns (report, collected samples for training) or (None, None)."""
    width, height, width_mm, height_mm, distance_cm = screen
    passes = {}
    for name, grid in BENCH_PASSES:
        lines = (["Accuracy test 1/2 - the same test GazeFlow got", "Look at the black centre of each dot.",
                  "Sit normally."] if name == "still" else
                 ["Accuracy test 2/2 - head moved",
                  "Shift your head about 5 cm (a hand's width) to one side and stay there.",
                  "Keep looking at each dot's centre."])
        if not announce(stage, lines):
            if name == "still":
                return None, None
            break
        collected = collect_targets(cap, extract, stage, benchmark_grid(width, height, *grid), BENCH_TIMING,
                                    trim=False)
        if not collected:
            if name == "still":
                return None, None
            break
        passes[name] = collected

    mm_per_px = (width_mm / width + height_mm / height) / 2 if width_mm and height_mm else 0.0
    # The cnn engine measures distance by PnP with the configured FOV and is used for degrees unless
    # --distance-cm is given; MediaPipe's depth (landmarks engine) assumes a generic camera, so it is
    # only reported.
    estimates = [d for _, _, ds in passes["still"] for d in ds if d]
    estimated_cm = float(np.median(estimates)) if estimates else float("nan")
    distance_cm = distance_cm or (estimated_cm if estimates else 60.0)

    def to_degrees(px):
        return math.degrees(math.atan(px * mm_per_px / (distance_cm * 10))) if mm_per_px else float("nan")

    # Raw model output (before smoothing), head still.
    accuracy_px, precision_px = [], []
    for target, samples, _ in passes["still"]:
        predicted = model.predict(samples)
        mean = predicted.mean(axis=0)
        accuracy_px.append(float(np.linalg.norm(mean - target)))
        precision_px.append(float(np.sqrt(np.mean(np.sum((predicted - mean) ** 2, axis=1)))))
    accuracy, precision = float(np.mean(accuracy_px)), float(np.mean(precision_px))

    # Cursor path per pass, scored exactly like benchmark.py scores the real cursor.
    bench = {"label": BENCH_LABEL, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "screen": [width, height],
             "distance_cm": distance_cm, "source": "simulated cursor during calibration"}
    cursors = {}
    for name, collected in passes.items():
        cursors[name] = [(target, simulate_cursor(model, samples, precision)) for target, samples, _ in collected]
        bench[name] = score_targets(cursors[name], mm_per_px, distance_cm)
    bench_path = save_bench(bench)

    still = bench["still"]
    report = {
        "accuracy_px": accuracy, "precision_px": precision,
        "cursor_jitter_px": still["precision_px"], "cursor_jitter_mm": still["precision_px"] * mm_per_px,
        "accuracy_mm": accuracy * mm_per_px, "precision_mm": precision * mm_per_px,
        "accuracy_deg": to_degrees(accuracy), "precision_deg": to_degrees(precision),
        "distance_cm": distance_cm, "measured_distance_cm": estimated_cm, "targets": len(passes["still"]),
        "benchmark": {name: {k: v for k, v in bench[name].items() if k != "per_target"} for name in cursors},
        "per_target": still["per_target"],
    }

    canvas = stage.blank()
    for name, color in (("still", (80, 255, 80)), ("moved", (255, 170, 80))):
        for target, cursor in cursors.get(name, []):
            mean = cursor.mean(axis=0)
            cv2.circle(canvas, tuple(int(v) for v in target), 10, (60, 200, 255), 2, cv2.LINE_AA)
            cv2.circle(canvas, (int(mean[0]), int(mean[1])), 7, color, -1, cv2.LINE_AA)
            cv2.line(canvas, tuple(int(v) for v in target), (int(mean[0]), int(mean[1])), color, 1, cv2.LINE_AA)
    reference = reference_accuracy()
    moved = bench.get("moved")
    lines = [
        f"Head still: {still['accuracy_px']:.0f} px = {still['accuracy_cm']:.1f} cm = {still['accuracy_deg']:.2f} deg"
        f"      cursor jitter {still['precision_px']:.0f} px",
        (f"Head moved: {moved['accuracy_px']:.0f} px = {moved['accuracy_cm']:.1f} cm = {moved['accuracy_deg']:.2f} deg"
         if moved else "Head moved: skipped"),
    ]
    if reference and reference["still"] is not None:
        moved_ref = f", head moved {reference['moved']:.0f} px" if reference["moved"] is not None else ""
        lines.append(f"GazeFlow on the same test: head still {reference['still']:.0f} px{moved_ref}"
                     f"  ({reference['runs']} run{'s' if reference['runs'] > 1 else ''}, benchmark.py)")
    lines += [f"Degrees assume {distance_cm:.0f} cm eye-screen distance.  Raw model accuracy {accuracy:.0f} px.",
              "Yellow = dot, green = cursor (head still), orange = cursor (head moved).  Any key to continue."]
    stage.text(canvas, lines, y=60, scale=0.75)
    stage.show(canvas, 0)
    print("validation:", json.dumps({k: round(v, 2) if isinstance(v, float) else v
                                     for k, v in report.items() if k not in ("per_target", "benchmark")}))
    print(f"benchmark (same test as benchmark.py): head still {still['accuracy_px']:.0f} px / "
          f"{still['accuracy_deg']:.2f} deg" + (f", head moved {moved['accuracy_px']:.0f} px / "
                                               f"{moved['accuracy_deg']:.2f} deg" if moved else "")
          + (f"; GazeFlow {reference['still']:.0f} px" if reference and reference["still"] is not None else "")
          + f"  -> {bench_path.name}", flush=True)
    return report, [c for collected in passes.values() for c in collected]


def save_profile(model, screen, camera, report, posture=None):
    """posture: the user's position at calibration (positioning.py); None keeps the saved one."""
    if posture is None:
        try:
            posture = json.loads(PROFILE_PATH.read_text()).get("posture")
        except (OSError, ValueError):
            posture = None
    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROFILE_PATH.write_text(json.dumps({
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "screen": list(screen[:2]), "camera": camera, "camera_name": camera_names().get(camera),
        "posture": posture, "model": model.to_json(), "validation": report,
    }, indent=1))


def load_profile(screen, camera):
    if not PROFILE_PATH.exists():
        return None
    data = json.loads(PROFILE_PATH.read_text())
    # Camera numbers shift when a camera is plugged in or disabled, so match by device name when known.
    saved_name, name = data.get("camera_name"), camera_names().get(camera)
    same_camera = saved_name == name if saved_name and name else data.get("camera") == camera
    if data.get("screen") != list(screen[:2]) or not same_camera:
        return None
    if data["model"].get("kind") == "cnn" and data["model"].get("features") != CNN_FEATURES_VERSION:
        print("CNN engine changed since the saved calibration - recalibrating")
        return None
    return model_from_json(data["model"]), data.get("validation"), data.get("posture")


def saved_precision():
    try:
        return json.loads(PROFILE_PATH.read_text())["validation"]["precision_px"]
    except (OSError, KeyError, TypeError, ValueError):
        return None


def limit(vector, length):
    norm = np.linalg.norm(vector)
    return vector * (length / norm) if norm > length else vector


def calibration_report(model, collected, posture):
    """Minimal report when no accuracy test ran: raw jitter (the cursor smoothing is sized from it) and
    the measured eye distance."""
    spreads = []
    for _, samples, _ in collected:
        predicted = model.predict(samples)
        spreads.append(float(np.sqrt(np.mean(np.sum((predicted - predicted.mean(axis=0)) ** 2, axis=1)))))
    return {"precision_px": float(np.mean(spreads)) if spreads else None,
            "measured_distance_cm": posture.get("distance_cm") if posture else None, "accuracy_test": False}


def calibrate_and_validate(session, screen, engine, mode="quick", benchmark=None):
    """Returns (model, posture) or None if cancelled. The calibration is saved for the camera chosen
    in the positioning window (session.index). benchmark: run the accuracy test (benchmark.py protocol)
    afterwards; None = the mode's default."""
    benchmark = CALIBRATION_MODES[mode]["benchmark"] if benchmark is None else benchmark
    stage = Stage(*screen[:2])
    try:
        calibration = calibrate(session, stage, engine, mode)
        if calibration is None:
            return None
        model, collected, still_targets, pursuit, posture = calibration
        cap, extract, camera = session.cap, session.extract, session.index

        def save_samples(validation):
            # Raw samples allow refitting after model changes without a new calibration.
            # group_pass (per target group >= 0): 0 = head still, 1 = head moving, 2 = accuracy test;
            # negative groups are smooth-pursuit segments.
            x, y, groups = with_pursuit(*stack(collected + validation), pursuit)
            group_pass = [0] * still_targets + [1] * (len(collected) - still_targets) + [2] * len(validation)
            SAMPLES_PATH.parent.mkdir(parents=True, exist_ok=True)   # first run of the .exe: no folder yet
            np.savez_compressed(SAMPLES_PATH, x=x.astype(np.float32), y=y, groups=groups,
                                group_pass=np.array(group_pass))
            return x, y

        save_samples([])   # kept even if the accuracy test is interrupted
        report, validation = validate(cap, extract, stage, model, screen) if benchmark else (None, None)
        if validation:
            # The accuracy test points are extra calibration data for the streaming model.
            x, y = save_samples(validation)
            model = model.refit(x, y)
        if report is None:
            report = calibration_report(model, collected[:still_targets], posture)
        save_profile(model, screen, camera, report, posture)
        return model, posture
    finally:
        stage.close()


def refit_profile(screen, camera, engine):
    """Re-runs model selection on the recorded calibration samples; keeps the old accuracy report."""
    if not SAMPLES_PATH.exists():
        sys.exit(f"no recorded samples at {SAMPLES_PATH}; calibrate instead")
    data = np.load(SAMPLES_PATH)
    x, y, groups = data["x"].astype(np.float64), data["y"], data["groups"]
    model, cv_error = (CnnGazeModel if engine == "cnn" else GazeModel).select(x, y, groups)
    try:
        report = json.loads(PROFILE_PATH.read_text()).get("validation")
    except (OSError, ValueError):
        report = None
    if report:
        report["measured_before_refit"] = True
    save_profile(model, screen, camera, report)
    print(f"refit from {len(x)} samples: model {model.describe()}, cross-validated error {cv_error:.0f} px")


def run_validation_only(cap, extract, screen, camera, model):
    stage = Stage(*screen[:2])
    try:
        report, _ = validate(cap, extract, stage, model, screen)
        if report:
            save_profile(model, screen, camera, report)
    finally:
        stage.close()


def drift_offset(cap, extract, screen, model, prompt=False):
    """One-point drift correction: fixate the screen centre; returns target - mean raw prediction."""
    stage = Stage(*screen[:2])
    try:
        if prompt and not announce(stage, ["Drift check", "Look at the centre of the dot until it shrinks."],
                                   seconds=3):
            return None
        collected = collect_targets(cap, extract, stage, [(stage.width / 2, stage.height / 2)])
    finally:
        stage.close()
    if not collected:
        return None
    target, samples, _ = collected[0]
    offset = target - model.predict(samples).mean(axis=0)
    print(f"drift correction: offset {offset[0]:+.0f},{offset[1]:+.0f} px")
    return offset


def drift_hotkey_pressed():
    """Ctrl+Alt+G, polled so it works while GazePointer has focus."""
    state = ctypes.windll.user32.GetAsyncKeyState
    return all(state(vk) & 0x8000 for vk in (0x11, 0x12, ord("G")))


def stream(session, model, screen, port, glide_ms, learn_from_clicks, output_mode, engine, offset, posture,
           pointer_size=POINTER_DIAMETER):
    address = ("127.0.0.1", port)
    # GazePointer reports invoked targets to port + 1; they reveal calibration bias.
    feedback = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    feedback.bind(("127.0.0.1", port + 1))
    feedback.setblocking(False)
    if output_mode == "overlay":
        output = GazeOverlay(pointer_size)
    elif output_mode == "cursor":
        output = SystemCursor()
    else:
        output = UdpOutput(address)
    sender = GazeSender(output, glide_ms / 1000)
    sender.start()
    stabilizer = FixationStabilizer(stabilizer_radius(saved_precision()))
    hotkey_down = False
    status = "webcam-gaze"
    cv2.namedWindow(status, cv2.WINDOW_AUTOSIZE)
    frames, fps, fps_start = 0, 0.0, time.monotonic()
    destination = {"cursor": "the system cursor", "overlay": "the on-screen gaze pointer"}.get(
        output_mode, f"udp://{address[0]}:{port}")
    print(f"streaming gaze to {destination}  "
          "(q quit, p position check, r recalibrate, v accuracy test, c or Ctrl+Alt+G drift fix)", flush=True)

    def restart_filters():
        nonlocal stabilizer
        sender.set_target(None)
        stabilizer = FixationStabilizer(stabilizer_radius(saved_precision()))

    stats, stats_start, last_gaze = dict.fromkeys(("frames", "face", "blink", "gaze"), 0), time.monotonic(), 0.0
    reasons = {}
    while True:
        ok, frame = session.cap.read()
        if not ok:
            continue
        features, info = session.extract(frame)
        now = time.monotonic()
        gaze = None
        stats["frames"] += 1
        stats["face"] += info["face"]
        stats["blink"] += info["blink"]
        if info.get("reject"):
            reasons[info["reject"]] = reasons.get(info["reject"], 0) + 1
        if features is not None:
            stats["gaze"] += 1
            last_gaze = now
            gaze = stabilizer(model.predict(features)[0] + offset)
            # Small overshoots are held at the edge so the cursor stays visible; gaze clearly
            # off-screen (e.g. at the keyboard) is passed on so GazePointer sees "eyes off"
            # instead of snapping onto edge buttons.
            low, high = -EDGE_HOLD_PX, np.array(screen[:2]) + EDGE_HOLD_PX
            if np.all(gaze >= low) and np.all(gaze <= high):
                gaze = np.clip(gaze, 0, np.array(screen[:2]) - 1)
            sender.set_target(gaze)
        elif not info["face"] or now - last_gaze > GAZE_TIMEOUT_SECONDS:
            # No usable gaze (face gone, or a blink / rejected frames lasting longer than a blink):
            # release the cursor rather than holding it at a stale position.
            stabilizer.reset()
            sender.set_target(None)
        if now - stats_start >= STATUS_LOG_SECONDS:
            print(f"status: {stats['frames'] / (now - stats_start):.0f} fps, face {stats['face']}, "
                  f"blink {stats['blink']}, rejected {stats['face'] - stats['blink'] - stats['gaze']}, "
                  f"gaze {stats['gaze']} of {stats['frames']} frames"
                  + (f"  (rejected: {reasons})" if reasons else "")
                  + (f"  CAMERA PICTURE TOO DARK ({frame.mean():.0f}/255)" if frame.mean() < DARK_FRAME_MEAN else ""),
                  flush=True)
            stats, stats_start, reasons = dict.fromkeys(stats, 0), now, {}
        cursor = sender.position

        while True:
            try:
                message = feedback.recv(256).decode("ascii", "replace").split(",")
            except (BlockingIOError, ConnectionResetError):
                break
            if learn_from_clicks and message[0] == "target" and len(message) == 5 and cursor is not None:
                correction = target_correction(cursor, [float(v) for v in message[1:]])
                if np.any(correction) and np.linalg.norm(correction) <= MAX_FEEDBACK_PX:
                    offset = limit(offset + FEEDBACK_GAIN * correction, MAX_OFFSET_PX)
                    print(f"click feedback: drift offset now {offset[0]:+.0f},{offset[1]:+.0f} px", flush=True)

        frames += 1
        if now - fps_start >= 1.0:
            fps, frames, fps_start = frames / (now - fps_start), 0, now

        preview = cv2.resize(frame, (480, int(480 * frame.shape[0] / frame.shape[1])))
        if info["points"] is not None:
            k = 480 / frame.shape[1]
            for index in (*[e[c] for e in EYES for c in ("a", "b", "top", "bottom")], *IRISES[0], *IRISES[1]):
                p = info["points"][index] * k
                cv2.circle(preview, (int(p[0]), int(p[1])), 1, (0, 255, 0), -1)
        state = "no face" if not info["face"] else "blink" if info["blink"] else (
            f"gaze {gaze[0]:.0f},{gaze[1]:.0f}" if gaze is not None else "")
        cv2.putText(preview, f"{fps:4.1f} fps  {state}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(preview, "q quit  p position  r recalibrate  v accuracy  c / Ctrl+Alt+G drift",
                    (8, preview.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.imshow(status, preview)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or cv2.getWindowProperty(status, cv2.WND_PROP_VISIBLE) < 1:
            break
        pressed = drift_hotkey_pressed()
        drift_requested = key == ord("c") or (pressed and not hotkey_down)
        hotkey_down = pressed
        if key in (ord("r"), ord("v"), ord("p")) or drift_requested:
            sender.set_target(None)  # GazePointer must not see a frozen cursor meanwhile
        if key == ord("r"):
            calibration = calibrate_and_validate(session, screen, engine)
            if calibration is not None:
                (model, posture), offset = calibration, np.zeros(2)
            restart_filters()
        if key == ord("p"):
            # Back into the calibration posture, then a drift check for what remains.
            if position_user(session, screen, posture) is not None:
                new_offset = drift_offset(session.cap, session.extract, screen, model, prompt=True)
                if new_offset is not None:
                    offset = new_offset
            restart_filters()
        if key == ord("v"):
            run_validation_only(session.cap, session.extract, screen, session.index, model)
            restart_filters()
        if drift_requested:
            new_offset = drift_offset(session.cap, session.extract, screen, model)
            if new_offset is not None:
                offset = new_offset
            restart_filters()
    sender.stop()
    feedback.close()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", default="auto",
                        help="auto (external webcam, else built-in), a camera index, or part of its name, e.g. "
                             "LifeCam (see --list-cameras); names survive Windows renumbering cameras")
    parser.add_argument("--list-cameras", action="store_true")
    parser.add_argument("--calibration", choices=("quick", "full", "saved"), default="quick",
                        help="quick: ~2 min fresh calibration (default; a calibration hardly carries over to a "
                             "new sitting); full: ~6 min incl. the accuracy test; saved: reuse the saved one "
                             "(position check + drift dot; fast but much less accurate)")
    parser.add_argument("--benchmark", action="store_true",
                        help="run the accuracy test (benchmark.py protocol) after a quick calibration too")
    parser.add_argument("--recalibrate", action="store_true", help="same as --calibration full")
    parser.add_argument("--port", type=int, default=4242, help="GazePointer NetworkPort")
    parser.add_argument("--glide-ms", type=float, default=GLIDE_MS,
                        help="cursor ease time constant; lower = snappier, higher = silkier")
    parser.add_argument("--learn-from-clicks", action="store_true",
                        help="correct drift from GazePointer's invoked targets; only for layouts where "
                             "buttons are not right next to text you read")
    parser.add_argument("--distance-cm", type=float, default=None,
                        help="your eye-to-screen distance, used to express accuracy in degrees "
                             "(default: measured by the cnn engine, 60 for landmarks)")
    parser.add_argument("--output", choices=("overlay", "cursor", "udp"), default="overlay",
                        help="overlay: a circular gaze pointer on top of the screen, the mouse stays yours "
                             "(default); cursor: move the Windows cursor; udp: GazePointer's Network sensor")
    parser.add_argument("--pointer-size", type=int, default=POINTER_DIAMETER,
                        help=f"gaze pointer diameter in pixels (overlay output, default {POINTER_DIAMETER})")
    parser.add_argument("--engine", choices=("cnn", "landmarks"), default="cnn",
                        help="cnn: ETH-XGaze appearance model (gaze_cnn.py); landmarks: iris/eye-corner curve fit")
    parser.add_argument("--hfov-deg", type=float, default=None,
                        help="webcam horizontal field of view; default from the camera name (LifeCam HD-6000: "
                             f"59), else {DEFAULT_HFOV_DEG:g}. Calibration absorbs small errors")
    parser.add_argument("--camera-above-mm", type=float, default=None,
                        help="distance from the webcam lens to the top edge of the screen (cnn engine); "
                             "default 15 for an external webcam clipped on the screen, 8 for a built-in one")
    parser.add_argument("--no-position-check", action="store_true",
                        help="skip the positioning window before the drift check (saved calibration)")
    parser.add_argument("--no-start-drift-fix", action="store_true",
                        help="skip the centre-dot drift fix when starting with a saved calibration")
    parser.add_argument("--self-test", action="store_true",
                        help="check that the face and gaze models load and run (no camera needed), then exit")
    parser.add_argument("--refit", action="store_true",
                        help="rebuild the saved calibration from its recorded samples (after a model change), then exit")
    args = parser.parse_args()

    global PROFILE_PATH, SAMPLES_PATH
    if args.engine == "cnn":
        # Separate files so each engine keeps its own calibration.
        PROFILE_PATH = PROFILE_PATH.with_name("calibration_cnn.json")
        SAMPLES_PATH = SAMPLES_PATH.with_name("calibration_cnn_samples.npz")

    distance_cm = args.distance_cm or (None if args.engine == "cnn" else 60.0)
    screen = (*screen_info(), distance_cm)
    if args.list_cameras:
        list_cameras()
        return
    if args.self_test:
        self_test(screen)
        return
    args.camera = resolve_camera(args.camera)
    if args.refit:
        refit_profile(screen, args.camera, args.engine)
        return
    if not MODEL_PATH.exists():
        sys.exit(f"missing {MODEL_PATH} - download face_landmarker.task from "
                 "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task")
    if args.engine == "cnn" and not (HERE / "models" / "xgaze_resnet50.onnx").exists():
        sys.exit("missing models/xgaze_resnet50.onnx - see README (export_xgaze_onnx.py), or use --engine landmarks")

    session = CameraSession(args.engine, screen, args.hfov_deg, args.camera_above_mm)
    if not session.open(args.camera):
        sys.exit(f"camera {args.camera} did not open or only sends black frames (lens covered? in use by "
                 "another app?); try --list-cameras")
    try:
        mode = "full" if args.recalibrate else args.calibration
        loaded = load_profile(screen, session.index) if mode == "saved" else None
        if mode == "saved" and not loaded:
            print("no usable saved calibration for this camera - running a quick calibration")
            mode = "quick"
        offset = np.zeros(2)
        if loaded:
            model, report, posture = loaded
            if report and "accuracy_px" in report:
                note = " (measured before a refit)" if report.get("measured_before_refit") else ""
                print(f"using saved calibration: accuracy {report['accuracy_px']:.0f} px / "
                      f"{report['accuracy_deg']:.2f} deg{note}")
            # Sit as at calibration (the positioning window guides back to the saved posture), then one
            # dot removes the shared offset that remains.
            if not args.no_position_check:
                position_user(session, screen, posture)
            if not args.no_start_drift_fix:
                offset = drift_offset(session.cap, session.extract, screen, model, prompt=True)
                offset = np.zeros(2) if offset is None else offset
        else:
            calibration = calibrate_and_validate(session, screen, args.engine, mode,
                                                 benchmark=True if args.benchmark else None)
            if calibration is None:
                sys.exit("calibration cancelled")
            model, posture = calibration
        stream(session, model, screen, args.port, args.glide_ms, args.learn_from_clicks, args.output, args.engine,
               offset, posture, args.pointer_size)
    finally:
        session.release()


def self_test(screen):
    """Loads both models and runs them once on a synthetic frame; prints what it found."""
    import gaze_cnn
    frame = np.full((720, 1280, 3), 128, np.uint8)
    features, info = FeatureExtractor()(frame)
    print(f"face landmarker: OK ({MODEL_PATH.name}; no face in a blank frame, as expected: {not info['face']})")
    session = gaze_cnn.session()
    gaze, embedding = gaze_cnn.CnnGaze.infer(type("S", (), {"session": session})(), [np.zeros((224, 224, 3), np.uint8)])
    print(f"gaze model: OK ({gaze_cnn.ONNX_PATH.name} on {session.get_providers()[0]}; "
          f"output {gaze.shape} + embedding {embedding.shape})")
    overlay = GazeOverlay(POINTER_DIAMETER)
    overlay((screen[0] / 2, screen[1] / 2))
    time.sleep(1.0)
    overlay.close()
    print("gaze pointer: OK (shown at the screen centre for 1 s)")
    names = camera_names()
    print("cameras:", ", ".join(f"{i}: {n}" for i, n in sorted(names.items())) or "none found")
    print(f"screen {screen[0]}x{screen[1]} px, {screen[2]}x{screen[3]} mm; calibrations are saved in {DATA_DIR}")


def pause_if_double_clicked():
    """The .exe opens its own console when double-clicked; keep it open so messages can be read."""
    if FROZEN:
        try:
            input("\nPress Enter to close this window.")
        except EOFError:
            pass


if __name__ == "__main__":
    try:
        main()
    except SystemExit as stop:
        if stop.code not in (None, 0):
            print(stop.code if isinstance(stop.code, str) else f"exit code {stop.code}")
            pause_if_double_clicked()
        raise
    except Exception:
        import traceback
        PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(PROFILE_PATH.parent / "webcam_gaze.log", "a") as log:
            log.write(f"--- {time.strftime('%Y-%m-%d %H:%M:%S')}\n{traceback.format_exc()}")
        traceback.print_exc()
        print(f"\nThis error was also written to {PROFILE_PATH.parent / 'webcam_gaze.log'}")
        pause_if_double_clicked()
        raise
