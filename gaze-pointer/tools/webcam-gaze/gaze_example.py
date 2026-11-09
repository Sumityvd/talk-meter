"""Smallest way to use the gaze model from your own Python code: camera frame in, gaze out.

    from gaze_example import GazeEstimator
    gaze = GazeEstimator(frame_size=(1280, 720), screen_px=(1920, 1080), screen_mm=(344, 193))
    result = gaze(frame)            # frame: BGR image from OpenCV
    if result:
        x, y = result["screen_px"]  # where the person looks, in screen pixels (uncalibrated)

This uses the model as it comes, without the 2-minute personal calibration that webcam_gaze.py does, so the
point is off by several degrees (often 200+ px) in a fairly constant direction. See INTEGRATION.md.

Demo (prints the gaze point; --show draws it on the camera picture):
    python gaze_example.py --camera 0 --show
    python gaze_example.py --video recording.mp4 --frames 50

Needs models/xgaze_resnet50.onnx and models/face_landmarker.task (see INTEGRATION.md for the download).
ETH-XGaze model: CC BY-NC-SA 4.0, non-commercial use only.
"""

import argparse
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

import gaze_cnn

LANDMARKER_PATH = Path(__file__).resolve().parent / "models" / "face_landmarker.task"
BLINK_THRESHOLD = 0.45   # the tracker's own value: frames above it are skipped


class GazeEstimator:
    """frame_size: camera picture (width, height) in pixels.
    screen_px / screen_mm: the screen's resolution and its visible size in millimetres.
    camera_above_mm: from the camera lens down to the top edge of the picture area (about 8 for a laptop's
    built-in camera, 15 for a webcam clipped on top). The camera must sit at the top centre of the screen.
    hfov_deg: the camera's horizontal field of view; 65 is a reasonable guess for an unknown webcam.
    """

    def __init__(self, frame_size, screen_px, screen_mm, camera_above_mm=8.0, hfov_deg=65.0, prefer_gpu=True):
        geometry = gaze_cnn.ScreenGeometry(*screen_px, *screen_mm, camera_above_mm=camera_above_mm)
        # smoothing and flip as the tracker uses them: steadier crops, mirrored patch averaged in
        self.model = gaze_cnn.CnnGaze(frame_size, hfov_deg, geometry, prefer_gpu=prefer_gpu, smoothing=0.5, flip=True)
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(LANDMARKER_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ms = 0

    def __call__(self, frame):
        """frame: BGR image of frame_size. Returns None (no face, or a blink) or a dict:
        screen_px: (x, y) on the screen, NaN if the gaze misses the screen plane
        pitchyaw: gaze angles in radians, in the model's normalized camera
        direction: unit 3D gaze vector in camera coordinates; face_center: face position in mm from the camera
        """
        height, width = frame.shape[:2]
        now_ms = max(int(time.monotonic() * 1000), self._last_ms + 1)   # VIDEO mode needs rising timestamps
        self._last_ms = now_ms
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        found = self.landmarker.detect_for_video(image, now_ms)
        if not found.face_landmarks:
            return None
        scores = {c.category_name: c.score for c in found.face_blendshapes[0]}
        if max(scores.get("eyeBlinkLeft", 0), scores.get("eyeBlinkRight", 0)) > BLINK_THRESHOLD:
            return None
        points = np.array([(p.x * width, p.y * height) for p in found.face_landmarks[0]])
        result = self.model(frame, points)
        if result is None:
            return None
        return {key: result[key] for key in ("screen_px", "pitchyaw", "direction", "face_center")}

    def close(self):
        self.landmarker.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    source.add_argument("--video", help="a video file instead of a camera")
    parser.add_argument("--screen-px", type=int, nargs=2, default=(1920, 1080), metavar=("W", "H"))
    parser.add_argument("--screen-mm", type=float, nargs=2, default=(344, 193), metavar=("W", "H"),
                        help="visible screen size in mm (default: a 15.6-inch laptop)")
    parser.add_argument("--camera-above-mm", type=float, default=8.0)
    parser.add_argument("--hfov-deg", type=float, default=65.0)
    parser.add_argument("--frames", type=int, default=0, help="stop after this many frames (0 = until q or the end)")
    parser.add_argument("--show", action="store_true", help="show the camera picture with the gaze point")
    parser.add_argument("--cpu", action="store_true", help="don't use the GPU")
    args = parser.parse_args()

    capture = cv2.VideoCapture(args.video) if args.video else cv2.VideoCapture(args.camera, cv2.CAP_MSMF)
    if not args.video:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    ok, frame = capture.read()
    if not ok:
        raise SystemExit("Could not read from the camera or video.")
    size = (frame.shape[1], frame.shape[0])
    gaze = GazeEstimator(size, args.screen_px, args.screen_mm, args.camera_above_mm, args.hfov_deg, prefer_gpu=not args.cpu)
    print(f"{size[0]}x{size[1]} frames, model on {gaze.model.provider}")
    count = 0
    try:
        while ok:
            result = gaze(frame)
            count += 1
            if result is None:
                print("no face (or a blink)")
            else:
                x, y = result["screen_px"]
                pitch, yaw = np.degrees(result["pitchyaw"])
                print(f"screen {x:7.0f} {y:7.0f} px | pitch {pitch:6.1f} yaw {yaw:6.1f} deg | "
                      f"face {result['face_center'][2] / 10:.0f} cm from camera")
            if args.show:
                picture = frame.copy()
                if result is not None and np.isfinite(result["screen_px"]).all():
                    # The screen drawn small in the top-left corner, with the gaze point on it.
                    w, h = 320, int(320 * args.screen_px[1] / args.screen_px[0])
                    cv2.rectangle(picture, (10, 10), (10 + w, 10 + h), (255, 255, 255), 1)
                    gx = int(10 + np.clip(result["screen_px"][0] / args.screen_px[0], 0, 1) * w)
                    gy = int(10 + np.clip(result["screen_px"][1] / args.screen_px[1], 0, 1) * h)
                    cv2.circle(picture, (gx, gy), 8, (0, 200, 255), 2)
                cv2.imshow("gaze (q to quit)", picture)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if args.frames and count >= args.frames:
                break
            ok, frame = capture.read()
    finally:
        capture.release()
        gaze.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
