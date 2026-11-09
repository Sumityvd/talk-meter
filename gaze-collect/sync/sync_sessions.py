"""Moves finished gaze-collect sessions from Supabase to this PC, in the format train_gaze.session_items() reads.

Each frame's face crop is pasted back at its place in an empty camera frame, then goes through the tracker's own
gaze_cnn code (head_pose, normalize), so patches come from exactly the same pipeline as a native recording.
A session is deleted from Supabase only after its download and conversion succeed.

Environment: GAZE_SUPABASE_URL, GAZE_SUPABASE_SERVICE_KEY (never commit it), and optionally
GAZE_TRACKER_DIR (default D:\\gaze-v2\\webcam-gaze) and GAZE_FRIENDS_DIR (default D:\\gaze-train\\friends).

Run with the tracker's venv:
  python sync_sessions.py                          download and convert every finished session
  python sync_sessions.py --add-participant Rahul  create an invite code (the name stays on this PC only)
  python sync_sessions.py --delete-participant Rahul   remove their data from Supabase and this PC (a name, or an ID like VK7Q3XA)
  python sync_sessions.py --list                   participants and session counts
  python sync_sessions.py --convert session.bin OUT    convert a local file (no Supabase)
"""

import argparse
import hashlib
import json
import os
import re
import secrets
import shutil
import struct
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np

TRACKER_DIR = Path(os.environ.get("GAZE_TRACKER_DIR", r"D:\gaze-v2\webcam-gaze"))
FRIENDS_DIR = Path(os.environ.get("GAZE_FRIENDS_DIR", r"D:\gaze-train\friends"))
sys.path.insert(0, str(TRACKER_DIR))
import gaze_cnn  # noqa: E402

BUCKET = "gaze-sessions"
LANDMARKS = 478
SMOOTHING = 0.5            # as the tracker's CnnFeatureExtractor
BLINK_THRESHOLD = 0.45
FEATURE_COLUMNS = 6399     # session.npz layout: face centre in 6392:6395, patch id in 6398
MIN_FREE_BYTES = 2 * 1024 ** 3
STALE_UPLOAD_DAYS = 7
CODE_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"


class Normalizer(gaze_cnn.CnnGaze):
    """CnnGaze's head_pose and normalize, without loading the gaze model."""

    def __init__(self, frame_size, hfov_deg):
        self.camera = gaze_cnn.camera_matrix(*frame_size, hfov_deg)
        self.smoothing, self.flip = SMOOTHING, False
        self.reset()


def read_session_file(path):
    data = Path(path).read_bytes()
    if data[:4] != b"GZC1":
        raise ValueError(f"{path} is not a gaze-collect session file")
    size = struct.unpack("<I", data[4:8])[0]
    manifest = json.loads(data[8:8 + size])
    count = len(manifest["frames"])
    start = 8 + size
    landmarks = np.frombuffer(data, "<f4", count * LANDMARKS * 2, start).reshape(count, LANDMARKS, 2)
    return manifest, landmarks, data, start + landmarks.nbytes


def convert(session_file, output, participant=None):
    """session file -> output/patches/NNNNN.jpg, session.npz, meta.json. Returns a small report."""
    manifest, landmarks, data, offset = read_session_file(session_file)
    width, height = manifest["frame_size"]
    normalizer = Normalizer((width, height), manifest["hfov_deg"])
    frame = np.zeros((height, width, 3), np.uint8)
    output = Path(output)
    (output / "patches").mkdir(parents=True, exist_ok=True)

    rotations, rows, targets, groups = [], [], [], []
    dropped = {"blink": 0, "pose": 0}
    for index, record in enumerate(manifest["frames"]):
        jpeg = np.frombuffer(data, np.uint8, record["jpeg_bytes"], offset)
        offset += record["jpeg_bytes"]
        if record["blink"] > BLINK_THRESHOLD:
            dropped["blink"] += 1
            continue
        pose = normalizer.head_pose(landmarks[index].astype(np.float64))
        if pose is None:
            dropped["pose"] += 1
            continue
        x, y, w, h = record["crop"]
        frame[y:y + h, x:x + w] = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)
        patch, rotation, face_center = normalizer.normalize(frame, *pose)[:3]
        frame[y:y + h, x:x + w] = 0
        patch_id = len(rotations)
        cv2.imwrite(str(output / "patches" / f"{patch_id:05d}.jpg"), patch, [cv2.IMWRITE_JPEG_QUALITY, 92])
        rotations.append(np.asarray(rotation, np.float32))
        row = np.zeros(FEATURE_COLUMNS, np.float32)
        row[6392:6395] = face_center
        row[6398] = patch_id
        rows.append(row)
        targets.append((record["x"], record["y"]))
        groups.append(record["group"])
    if not rows:
        raise ValueError(f"{session_file}: no usable frames")

    np.savez_compressed(output / "session.npz", x=np.stack(rows), y=np.array(targets, np.float64),
                        groups=np.array(groups, np.int64), group_pass=np.array(manifest["group_pass"], np.int64),
                        rotations=np.stack(rotations))
    keys = ("lighting", "camera_name", "camera_position", "hfov_deg", "camera_above_mm", "screen_px", "screen_mm",
            "device_pixel_ratio", "frame_size", "posture", "warnings", "expected_accuracy", "consent_version",
            "duration_s", "created")
    meta = {key: manifest[key] for key in keys}
    meta.update(mode="web", source="web", participant=participant, gaze_model=None,
                screen_source=manifest.get("screen_source"))
    (output / "meta.json").write_text(json.dumps(meta, indent=1))
    return {"frames": len(rows), "dropped": dropped, "lighting": manifest["lighting"]}


# ------------------------------------------------------------------ Supabase ----

def request(path, method="GET", body=None, headers=None, raw=False):
    base = os.environ.get("GAZE_SUPABASE_URL", "").rstrip("/")
    key = os.environ.get("GAZE_SUPABASE_SERVICE_KEY", "")
    if not base or not key:
        sys.exit("Set GAZE_SUPABASE_URL and GAZE_SUPABASE_SERVICE_KEY first.")
    payload = None if body is None else json.dumps(body).encode()
    all_headers = {"apikey": key, "Authorization": f"Bearer {key}", "Content-Type": "application/json", **(headers or {})}
    with urllib.request.urlopen(urllib.request.Request(base + path, payload, all_headers, method=method), timeout=300) as reply:
        content = reply.read()
    return content if raw else (json.loads(content) if content else None)


def participants_file():
    return FRIENDS_DIR / "participants.json"


def load_participants():
    return json.loads(participants_file().read_text()) if participants_file().exists() else {}


def save_participants(data):
    FRIENDS_DIR.mkdir(parents=True, exist_ok=True)
    participants_file().write_text(json.dumps(data, indent=1))


def add_participant(name):
    people = load_participants()
    if name in people:
        sys.exit(f"{name} already has a code (alias {people[name]['alias']}).")
    existing = request("/rest/v1/gaze_participants?select=alias")
    # Your own invites are P01, P02, ...; people who joined through the site are V + 6 letters.
    number = 1 + max([int(p["alias"][1:]) for p in existing if re.fullmatch(r"P\d+", p["alias"])] or [0])
    alias = f"P{number:02d}"
    code = "".join(secrets.choice(CODE_ALPHABET) for _ in range(10))
    code_hash = hashlib.sha256(code.encode()).hexdigest()
    row = request("/rest/v1/gaze_participants", "POST", {"alias": alias, "code_hash": code_hash},
                  {"Prefer": "return=representation"})[0]
    people[name] = {"alias": alias, "id": row["id"]}
    save_participants(people)
    print(f"{name}: alias {alias}, invite code {code}\nThe code is shown only now; send it to {name} privately.")


def delete_objects(folder):
    files = request(f"/storage/v1/object/list/{BUCKET}", "POST", {"prefix": folder, "limit": 1000}) or []
    names = [f"{folder}/{f['name']}" for f in files]
    if names:
        request(f"/storage/v1/object/{BUCKET}", "DELETE", {"prefixes": names})


def delete_participant(who):
    """`who` is a name you invited, or the participant ID someone gave you (for example VK7Q3XA or P02)."""
    people = load_participants()
    name = who if who in people else next((k for k, v in people.items() if v["alias"] == who.upper()), None)
    if name:
        person = people[name]
    else:
        alias = who.strip().upper()
        if not re.fullmatch(r"[A-Z0-9]{2,12}", alias):
            sys.exit(f"Not a participant ID or known name: {who}")
        rows = request(f"/rest/v1/gaze_participants?alias=eq.{alias}&select=id,alias")
        if not rows:
            sys.exit(f"Unknown participant {who}. Known names: {', '.join(people) or 'none'}")
        person = rows[0]
    for row in request(f"/rest/v1/gaze_sessions?participant_id=eq.{person['id']}&select=id"):
        delete_objects(f"{person['id']}/{row['id']}")
    request(f"/rest/v1/gaze_participants?id=eq.{person['id']}", "DELETE")   # sessions go with it (cascade)
    shutil.rmtree(FRIENDS_DIR / person["alias"], ignore_errors=True)
    if name:
        del people[name]
        save_participants(people)
    print(f"Deleted {name or 'participant'} ({person['alias']}) from Supabase and this PC.")


def list_participants():
    names = {v["id"]: k for k, v in load_participants().items()}
    participants = request("/rest/v1/gaze_participants?select=id,alias,consent_version,created_at&order=created_at")
    sessions = request("/rest/v1/gaze_sessions?select=participant_id,status,size_bytes")
    held = sum(s["size_bytes"] for s in sessions if s["status"] != "synced")
    for person in participants:
        mine = [s for s in sessions if s["participant_id"] == person["id"]]
        name = names.get(person["id"], "(joined through the site)")
        print(f"{person['alias']} {name}: {sum(s['status'] == 'synced' for s in mine)} synced, "
              f"{sum(s['status'] == 'uploaded' for s in mine)} waiting, {sum(s['status'] == 'uploading' for s in mine)} unfinished"
              f" | consent {person['consent_version'] or 'none'}, joined {person['created_at'][:10]}")
    print(f"{len(participants)} participants. Held in Supabase: {held / 1e6:.0f} MB")


def sync():
    aliases = {row["id"]: row["alias"] for row in request("/rest/v1/gaze_participants?select=id,alias")}
    rows = request("/rest/v1/gaze_sessions?status=eq.uploaded&select=*&order=created_at")
    print(f"{len(rows)} finished session(s) to download")
    FRIENDS_DIR.mkdir(parents=True, exist_ok=True)
    for row in rows:
        free = shutil.disk_usage(FRIENDS_DIR).free
        if free < MIN_FREE_BYTES + 2 * row["size_bytes"]:
            sys.exit(f"Stopping: only {free / 1e9:.1f} GB free on {FRIENDS_DIR.drive}. Nothing else was downloaded or deleted.")
        alias = aliases[row["participant_id"]]
        folder = f"{row['participant_id']}/{row['id']}"
        stamp = row["created_at"][:19].replace("-", "").replace(":", "").replace("T", "-")
        output = FRIENDS_DIR / alias / f"{stamp}-{row['id'][:8]}"
        (output / "raw").mkdir(parents=True, exist_ok=True)
        session_file = output / "raw" / "session.bin"
        digest = hashlib.sha256()
        with open(session_file, "wb") as out:
            for index in range(row["part_count"]):
                part = request(f"/storage/v1/object/{BUCKET}/{folder}/part-{index:03d}", raw=True)
                out.write(part)
                digest.update(part)
        if digest.hexdigest() != row["sha256"]:
            shutil.rmtree(output)
            print(f"{alias} {row['id']}: checksum mismatch, left in Supabase")
            continue
        report = convert(session_file, output, alias)
        request(f"/rest/v1/gaze_sessions?id=eq.{row['id']}", "PATCH",
                {"status": "synced", "synced_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        delete_objects(folder)
        print(f"{alias} {output.name}: {report['frames']} frames, {report['lighting']}, dropped {report['dropped']}, "
              f"{row['size_bytes'] / 1e6:.0f} MB")
    cutoff = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - STALE_UPLOAD_DAYS * 86400))
    for row in request(f"/rest/v1/gaze_sessions?status=eq.uploading&created_at=lt.{cutoff}&select=id,participant_id"):
        delete_objects(f"{row['participant_id']}/{row['id']}")
        request(f"/rest/v1/gaze_sessions?id=eq.{row['id']}", "DELETE")
        print(f"Removed an upload unfinished for {STALE_UPLOAD_DAYS} days: {row['id']}")
    print(f"Free on {FRIENDS_DIR.drive}: {shutil.disk_usage(FRIENDS_DIR).free / 1e9:.1f} GB")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--add-participant", metavar="NAME")
    parser.add_argument("--delete-participant", metavar="NAME_OR_ID")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--convert", nargs=2, metavar=("FILE", "OUTPUT"))
    args = parser.parse_args()
    try:
        if args.convert:
            print(convert(*args.convert))
        elif args.add_participant:
            add_participant(args.add_participant)
        elif args.delete_participant:
            delete_participant(args.delete_participant)
        elif args.list:
            list_participants()
        else:
            sync()
    except urllib.error.HTTPError as error:
        sys.exit(f"Supabase error {error.code}: {error.read().decode(errors='replace')[:300]}")


if __name__ == "__main__":
    main()
