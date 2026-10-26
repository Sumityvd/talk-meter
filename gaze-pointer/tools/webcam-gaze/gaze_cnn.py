"""Appearance-based gaze: ETH-XGaze ResNet-50 on a normalized face patch, intersected with the screen.

Per frame (MediaPipe landmarks in, screen point out):
  1. Head pose: solvePnP of 6 landmarks (eye corners, nose wings) against ETH-XGaze's generic face model.
  2. Data normalization (Zhang et al., "Revisiting data normalization for appearance-based gaze
     estimation", ETRA 2018): warp the image to a virtual camera 600 mm from the face centre,
     looking straight at it with the head's roll removed -> 224x224 face patch.
  3. CNN -> gaze pitch/yaw in the normalized camera, plus its 2048-d embedding.
  4. Rotate the gaze back to the real camera, intersect the ray from the face centre with the
     screen plane. The webcam is in the laptop lid, so the screen lies in the camera's z = 0 plane.

Camera intrinsics are estimated from the horizontal field of view, so the geometric point carries
a few degrees of systematic error; the personal calibration in webcam_gaze.py removes it.

Normalization constants and face model follow https://github.com/xucong-zhang/ETH-XGaze (CC BY-NC-SA 4.0).
"""

import math
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ONNX_PATH = Path(__file__).resolve().parent / "models" / "xgaze_resnet50.onnx"

# ETH-XGaze face_model.txt rows 20, 23, 26, 29 (eye corners) and 15, 19 (nose wings), mm, camera axes
# (x right in the image, y down, z away from the camera).
FACE_MODEL = np.array([
    [-46.50949860, -38.32709503, 36.41600418],
    [-17.76072121, -32.58519745, 29.07615662],
    [18.04391098, -30.95682335, 29.06296730],
    [44.73758698, -34.10787201, 36.73243713],
    [-10.61166477, 12.95834923, 21.62276459],
    [11.28962994, 13.86424446, 21.83790016],
])
# The same points in MediaPipe's 478-point mesh (dlib 36, 39, 42, 45, 31, 35).
MEDIAPIPE_POINTS = [33, 133, 362, 263, 98, 327]

FOCAL_NORM = 960.0
DISTANCE_NORM = 600.0
PATCH = 224
# Per-eye normalization (MPIIGaze convention): each eye seen by its own virtual camera, 60x36 grey.
# The face patch shows an eye only ~45 px wide; these come from the full-resolution frame.
EYE_SIZE = (60, 36)
EYE_PIXELS = 2 * EYE_SIZE[0] * EYE_SIZE[1]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def camera_matrix(width, height, hfov_deg):
    focal = (width / 2) / math.tan(math.radians(hfov_deg) / 2)
    return np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1.0]])


def pitchyaw_to_vector(pitch, yaw):
    """Gaze direction from ETH-XGaze/MPIIGaze angles (looking at the camera = pitch 0, yaw 0 = -z)."""
    return np.array([-math.cos(pitch) * math.sin(yaw), -math.sin(pitch), -math.cos(pitch) * math.cos(yaw)])


def session(prefer_gpu=True):
    providers = ["CPUExecutionProvider"]
    if prefer_gpu and "DmlExecutionProvider" in ort.get_available_providers():
        providers.insert(0, "DmlExecutionProvider")
    return ort.InferenceSession(str(ONNX_PATH), providers=providers)


class ScreenGeometry:
    """Converts camera-space millimetres on the screen plane to screen pixels.

    camera_above_mm: distance from the camera to the top edge of the visible screen area.
    """

    def __init__(self, width_px, height_px, width_mm, height_mm, camera_above_mm=8.0):
        self.width_px, self.height_px = width_px, height_px
        self.width_mm, self.height_mm = width_mm, height_mm
        self.camera_above_mm = camera_above_mm

    def to_pixels(self, point_mm):
        # Camera +x points to the user's left as they face the screen, camera +y points down.
        sx_mm = self.width_mm / 2 - point_mm[0]
        sy_mm = point_mm[1] - self.camera_above_mm
        return np.array([sx_mm / self.width_mm * self.width_px, sy_mm / self.height_mm * self.height_px])


class CnnGaze:
    """smoothing: weight of the newest frame in an exponential average of the 6 pose landmarks
    (1 = none). Landmark jitter shakes the face crop, and the CNN is sensitive to the crop.
    flip: also run the mirrored patch and average (yaw negated) - halves noise for 2x GPU time.
    """

    def __init__(self, frame_size, hfov_deg, geometry, prefer_gpu=True, smoothing=1.0, flip=False):
        self.camera = camera_matrix(*frame_size, hfov_deg)
        self.geometry = geometry
        self.session = session(prefer_gpu)
        self.provider = self.session.get_providers()[0]
        self.smoothing, self.flip = smoothing, flip
        self._pose = None   # previous rvec, tvec as the solvePnP starting point
        self._points = None

    def reset(self):
        self._pose = self._points = None

    def head_pose(self, points):
        current = points[MEDIAPIPE_POINTS].astype(np.float64)
        # Restart the average after a big jump (new face position) so the crop never lags behind.
        if self._points is None or np.abs(current - self._points).max() > 40:
            self._points = current
        else:
            self._points = self._points + self.smoothing * (current - self._points)
        image_points = self._points.reshape(-1, 1, 2)
        model_points = FACE_MODEL.reshape(-1, 1, 3)
        if self._pose is None:
            ok, rvec, tvec = cv2.solvePnP(model_points, image_points, self.camera, None, flags=cv2.SOLVEPNP_EPNP)
            if not ok:
                return None
        else:
            rvec, tvec = (v.copy() for v in self._pose)
        ok, rvec, tvec = cv2.solvePnP(model_points, image_points, self.camera, None, rvec, tvec, True)
        # A face behind the camera or absurdly far away means PnP diverged; restart next frame.
        if not ok or not 150 < tvec[2, 0] < 1500:
            self._pose = None
            return None
        self._pose = (rvec, tvec)
        return rvec, tvec

    def _warp(self, image, center, head_rotation, size):
        """Virtual camera DISTANCE_NORM from `center` (mm), looking at it, head roll removed."""
        distance = np.linalg.norm(center)
        forward = center / distance
        down = np.cross(forward, head_rotation[:, 0])
        down /= np.linalg.norm(down)
        right = np.cross(down, forward)
        right /= np.linalg.norm(right)
        rotation = np.array([right, down, forward])
        camera_norm = np.array([[FOCAL_NORM, 0, size[0] / 2], [0, FOCAL_NORM, size[1] / 2], [0, 0, 1.0]])
        scale = np.diag([1.0, 1.0, DISTANCE_NORM / distance])
        warp = camera_norm @ scale @ rotation @ np.linalg.inv(self.camera)
        return cv2.warpPerspective(image, warp, size), rotation

    def normalize(self, frame, rvec, tvec):
        """Returns (face patch BGR 224x224, normalization rotation R, face centre in mm, eye pixels)."""
        head_rotation = cv2.Rodrigues(rvec)[0]
        face = head_rotation @ FACE_MODEL.T + tvec.reshape(3, 1)
        face_center = np.mean([face[:, 0:4].mean(axis=1), face[:, 4:6].mean(axis=1)], axis=0)
        patch, rotation = self._warp(frame, face_center, head_rotation, (PATCH, PATCH))
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eyes = []
        for corners in ((0, 1), (2, 3)):   # subject's right eye, left eye
            eye, _ = self._warp(grey, face[:, corners].mean(axis=1), head_rotation, EYE_SIZE)
            eyes.append(cv2.equalizeHist(eye).ravel())
        return patch, rotation, face_center, np.concatenate(eyes).astype(np.float32) / 255.0

    def infer(self, patches):
        """patches: list of BGR 224x224 -> (gaze [N, 2], embedding [N, 2048])."""
        batch = np.stack([cv2.cvtColor(p, cv2.COLOR_BGR2RGB) for p in patches]).astype(np.float32) / 255.0
        batch = ((batch - IMAGENET_MEAN) / IMAGENET_STD).transpose(0, 3, 1, 2)
        gaze, embedding = self.session.run(None, {"face": np.ascontiguousarray(batch)})
        return gaze, embedding

    def __call__(self, frame, points):
        """Returns dict(pitchyaw, embedding, screen_px, face_center, patch) or None."""
        pose = self.head_pose(points)
        if pose is None:
            return None
        patch, rotation, face_center, eyes = self.normalize(frame, *pose)
        if self.flip:
            gaze, embedding = self.infer([patch, cv2.flip(patch, 1)])
            pitch = float(gaze[0, 0] + gaze[1, 0]) / 2
            yaw = float(gaze[0, 1] - gaze[1, 1]) / 2
        else:
            gaze, embedding = self.infer([patch])
            pitch, yaw = (float(v) for v in gaze[0])
        direction = rotation.T @ pitchyaw_to_vector(pitch, yaw)
        screen_px = np.array([np.nan, np.nan])
        if direction[2] < -1e-6:   # pointing back toward the camera / screen plane
            hit = face_center + direction * (-face_center[2] / direction[2])
            screen_px = self.geometry.to_pixels(hit)
        return {"pitchyaw": np.array([pitch, yaw]), "embedding": embedding[0], "screen_px": screen_px,
                "direction": direction,
                "face_center": face_center, "patch": patch, "eyes": eyes}
