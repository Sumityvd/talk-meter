"""Prepares MPIIFaceGaze (laptop-webcam gaze dataset, CC BY-NC-SA 4.0, non-commercial research) for
train_gaze.py, using exactly the tracker's own face pipeline so training images match what it sees:
MediaPipe landmarks -> solvePnP head pose (gaze_cnn.FACE_MODEL) -> 224x224 normalized face patch.

Label: the true gaze vector from our face centre to the dataset's 3D gaze target (camera mm), rotated into
the normalized camera of the patch, as pitch/yaw in the ETH-XGaze convention (gaze_cnn.pitchyaw_to_vector).

Input:  D:\\gaze-data\\MPIIFaceGaze\\pXX\\ (images, pXX.txt, Calibration\\camera.json)
Output: D:\\gaze-data\\prepared\\pXX\\NNNNN.jpg + pXX.npz (pitchyaw, face_center, rotation, gaze_px, source)

Usage: python prepare_mpiifacegaze.py [--people p00 p01 ...]
"""

import argparse
import ctypes
import json
import math
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

import gaze_cnn
from webcam_gaze import MODEL_PATH

SOURCE = Path(r"D:\gaze-data\MPIIFaceGaze")
OUTPUT = Path(r"D:\gaze-data\prepared")


def vector_to_pitchyaw(v):
    """Inverse of gaze_cnn.pitchyaw_to_vector: v = [-cos p sin y, -sin p, -cos p cos y]."""
    v = v / np.linalg.norm(v)
    return math.asin(-v[1]), math.atan2(-v[0], -v[2])


def prepare_person(person, landmarker, engine):
    folder = SOURCE / person
    calibration = json.loads((folder / "Calibration" / "camera.json").read_text())
    camera = np.array(calibration["cameraMatrix"])
    distortion = np.array(calibration["distCoeffs"])
    out = OUTPUT / person
    out.mkdir(parents=True, exist_ok=True)
    engine.camera = camera
    rows = {"pitchyaw": [], "face_center": [], "rotation": [], "gaze_px": [], "source": [], "check_deg": []}
    lines = (folder / f"{person}.txt").read_text().split("\n")
    skipped = 0
    for line in lines:
        parts = line.split()
        if len(parts) < 28:
            continue
        image = cv2.imread(str(folder / parts[0]))
        if image is None:
            skipped += 1
            continue
        image = cv2.undistort(image, camera, distortion)
        result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                            data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
        if not result.face_landmarks:
            skipped += 1
            continue
        h, w = image.shape[:2]
        points = np.array([(p.x * w, p.y * h) for p in result.face_landmarks[0]])
        engine.reset()   # independent photos: no pose carried over from the previous one
        pose = engine.head_pose(points)
        if pose is None:
            skipped += 1
            continue
        patch, rotation, face_center, _ = engine.normalize(image, *pose)
        values = np.array(parts[1:27], float)   # 26 numbers; parts[27] is the eye label ("left"/"right")
        dataset_fc, target = values[20:23], values[23:26]
        gaze = target - face_center
        pitch, yaw = vector_to_pitchyaw(rotation @ gaze)
        # Sanity: our gaze vector vs the dataset's own (origins differ slightly, so a few degrees is normal).
        own = target - dataset_fc
        check = math.degrees(math.acos(np.clip(gaze @ own / np.linalg.norm(gaze) / np.linalg.norm(own), -1, 1)))
        index = len(rows["source"])
        cv2.imwrite(str(out / f"{index:05d}.jpg"), patch, [cv2.IMWRITE_JPEG_QUALITY, 92])
        rows["pitchyaw"].append((pitch, yaw))
        rows["face_center"].append(face_center)
        rows["rotation"].append(rotation)
        rows["gaze_px"].append(values[0:2])
        rows["source"].append(parts[0])
        rows["check_deg"].append(check)
    arrays = {k: np.array(v, np.float32) for k, v in rows.items() if k != "source"}
    np.savez_compressed(OUTPUT / f"{person}.npz", source=np.array(rows["source"]), **arrays)
    return len(rows["source"]), skipped, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--people", nargs="*", default=[p.name for p in sorted(SOURCE.glob("p??"))])
    args = parser.parse_args()
    # Background job: lower priority and few threads, so a calibration running at the same time stays smooth.
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    cv2.setNumThreads(2)
    landmarker = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)), num_faces=1))
    engine = gaze_cnn.CnnGaze((1280, 720), 60.0, geometry=None, prefer_gpu=False)   # crops only, no inference
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for person in args.people:
        start = time.monotonic()
        kept, skipped, arrays = prepare_person(person, landmarker, engine)
        py = np.degrees(arrays["pitchyaw"])
        print(f"{person}: {kept} patches, {skipped} skipped, {time.monotonic() - start:.0f} s | "
              f"pitch {py[:, 0].min():.0f}..{py[:, 0].max():.0f} deg, yaw {py[:, 1].min():.0f}..{py[:, 1].max():.0f} deg"
              f" | label vs dataset gaze: median {np.median(arrays['check_deg']):.1f} deg", flush=True)


if __name__ == "__main__":
    main()
