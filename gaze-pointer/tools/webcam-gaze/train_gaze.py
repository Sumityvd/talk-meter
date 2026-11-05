"""Fine-tunes ETH-XGaze for webcams (stage A) on MPIIFaceGaze patches made by prepare_mpiifacegaze.py,
and scores it against the unmodified ETH-XGaze.

Scores (mean angular error, degrees; lower is better):
  held-out people  MPIIFaceGaze people never used for training (--val), so this is accuracy on new faces
  your sessions    face patches recorded by webcam_gaze.py calibrations (sessions/*). The label goes through
                   the assumed camera/screen geometry, which adds a fixed error, so only compare models here
  flip             the tracker averages each patch with its mirror image, so 'flip' is what it would see

Run with the CUDA training venv:
  D:\\gaze-train\\.venv\\Scripts\\python.exe train_gaze.py --baseline-only        # just score ETH-XGaze
  D:\\gaze-train\\.venv\\Scripts\\python.exe train_gaze.py --limit 200 --epochs 1  # smoke test
  D:\\gaze-train\\.venv\\Scripts\\python.exe train_gaze.py                         # stage A, p03 p08 p14 held out
  D:\\gaze-train\\.venv\\Scripts\\python.exe train_gaze.py --val none              # final: all 15 people
Output: D:\\gaze-train\\runs\\<time>\\ (train.log, best.pth, metrics.json)
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from xgaze_model import load_model

HERE = Path(__file__).resolve().parent
DATA = Path(r"D:\gaze-data\prepared")
RUNS = Path(r"D:\gaze-train\runs")
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def vector_to_pitchyaw(v):
    """Inverse of gaze_cnn.pitchyaw_to_vector, for arrays [N, 3]."""
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    return np.stack([np.arcsin(-v[:, 1]), np.arctan2(-v[:, 0], -v[:, 2])], axis=1)


def angular_error_deg(a, b):
    """a, b: pitch/yaw [N, 2] -> angle between the gaze vectors, degrees [N]."""
    def vec(py):
        p, y = py[:, 0], py[:, 1]
        return np.stack([-np.cos(p) * np.sin(y), -np.sin(p), -np.cos(p) * np.cos(y)], axis=1)
    cos = np.clip(np.sum(vec(a) * vec(b), axis=1), -1, 1)
    return np.degrees(np.arccos(cos))


# ------------------------------------------------------------------ data ----

def mpii_items(people, limit=None):
    items = []
    for person in people:
        labels = np.load(DATA / f"{person}.npz")["pitchyaw"]
        items += [(DATA / person / f"{i:05d}.jpg", labels[i]) for i in range(len(labels))]
    if limit:
        random.Random(0).shuffle(items)
        items = items[:limit]
    return items


def session_items(session_dir, fixations_only=True):
    """Frames of one recorded calibration, labelled through its saved screen geometry.
    fixations_only: leave out the moving-dot frames, whose labels are less exact (for scoring)."""
    data = np.load(session_dir / "session.npz")
    meta = json.loads((session_dir / "meta.json").read_text())
    x, y, groups = data["x"], data["y"], data["groups"]
    keep = x[:, 6398] >= 0
    if fixations_only:
        keep &= groups >= 0
    (width_px, height_px), (width_mm, height_mm) = meta["screen_px"], meta["screen_mm"]
    # Inverse of gaze_cnn.ScreenGeometry.to_pixels (camera in the screen plane, screen z = 0).
    target = np.stack([width_mm / 2 - y[keep, 0] / width_px * width_mm,
                       y[keep, 1] / height_px * height_mm + meta["camera_above_mm"],
                       np.zeros(keep.sum())], axis=1)
    gaze = target - x[keep, 6392:6395]
    ids = x[keep, 6398].astype(int)
    rotations = data["rotations"][ids]
    labels = vector_to_pitchyaw(np.einsum("nij,nj->ni", rotations, gaze))
    return [(session_dir / "patches" / f"{i:05d}.jpg", label) for i, label in zip(ids, labels)]


def augment(image, rng):
    """Webcam-like degradations so the model stops relying on studio-quality detail."""
    h, w = image.shape[:2]
    flip = rng.random() < 0.5
    if flip:
        image = image[:, ::-1]
    # Crop jitter: landmark noise moves the face crop a few pixels at runtime.
    scale = rng.uniform(0.95, 1.05)
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), 0, scale)
    matrix[:, 2] += rng.uniform(-8, 8, 2)
    image = cv2.warpAffine(np.ascontiguousarray(image), matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if rng.random() < 0.5:   # low-resolution webcam
        factor = rng.uniform(0.3, 1.0)
        small = cv2.resize(image, (max(8, int(w * factor)), max(8, int(h * factor))), interpolation=cv2.INTER_AREA)
        image = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.3:
        image = cv2.GaussianBlur(image, (5, 5) if rng.random() < 0.4 else (3, 3), 0)
    if rng.random() < 0.5:   # compression artefacts
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(rng.uniform(30, 95))])
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    image = image.astype(np.float32) / 255.0
    if rng.random() < 0.8:   # exposure, contrast, gamma
        image = np.clip(image * rng.uniform(0.6, 1.4) + rng.uniform(-0.1, 0.1), 0, 1)
        image = image ** rng.uniform(0.7, 1.4)
    if rng.random() < 0.5:   # white balance
        image = np.clip(image * rng.uniform(0.92, 1.08, 3).astype(np.float32), 0, 1)
    return image, flip


class PatchDataset(torch.utils.data.Dataset):
    def __init__(self, items, train):
        self.items, self.train = items, train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, label = self.items[index]
        image = cv2.imread(str(path))
        label = np.array(label, np.float32)
        if self.train:
            rng = np.random.default_rng()
            image, flip = augment(image, rng)
            if flip:
                label[1] = -label[1]
        else:
            image = image.astype(np.float32) / 255.0
        image = (cv2.cvtColor(image, cv2.COLOR_BGR2RGB) - MEAN) / STD
        return torch.from_numpy(image.transpose(2, 0, 1).copy()), torch.from_numpy(label)


WORKERS = 0   # set by --workers
# fp16 is off by default: on the GTX 1650 (no tensor cores) cuDNN fp16 returned NaN at batch 64 every time.
AMP = False   # set by --amp


def loader(items, train, batch):
    # Each worker process loads its own copy of torch's CUDA DLLs (>1 GB of commit), which exhausted the paging
    # file on this 8 GB laptop. The GTX 1650 is the bottleneck anyway, so decoding in-process costs little.
    workers = WORKERS if train else 0
    return torch.utils.data.DataLoader(PatchDataset(items, train), batch_size=batch, shuffle=train,
                                       num_workers=workers, pin_memory=True, persistent_workers=workers > 0,
                                       drop_last=train)


# ------------------------------------------------------------- evaluation ----

@torch.no_grad()
def evaluate(model, data_loader, device):
    """Returns (plain error, flip-averaged error) in degrees, means over all samples."""
    model.eval()
    plain, flipped, labels = [], [], []
    for images, label in data_loader:
        images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.float16, enabled=AMP):
            gaze, _ = model(images)
            mirror, _ = model(torch.flip(images, dims=[3]))
        gaze, mirror = gaze.float().cpu().numpy(), mirror.float().cpu().numpy()
        plain.append(gaze)
        flipped.append(np.stack([(gaze[:, 0] + mirror[:, 0]) / 2, (gaze[:, 1] - mirror[:, 1]) / 2], axis=1))
        labels.append(label.numpy())
    labels = np.concatenate(labels)
    return (float(angular_error_deg(np.concatenate(plain), labels).mean()),
            float(angular_error_deg(np.concatenate(flipped), labels).mean()))


def score(model, sets, device, log):
    result = {}
    for name, data_loader in sets.items():
        plain, flip = evaluate(model, data_loader, device)
        result[name] = {"plain": plain, "flip": flip}
    log("  " + "   ".join(f"{name}: {r['flip']:.2f} deg (plain {r['plain']:.2f})" for name, r in result.items()))
    return result


# --------------------------------------------------------------- training ----

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--val", nargs="+", default=["p03", "p08", "p14"],
                        help="held-out MPIIFaceGaze people, or 'none' to train on all 15")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--freeze", default="layer2", choices=("none", "layer1", "layer2", "layer3"),
                        help="keep the backbone frozen up to and including this block (generic edge/texture "
                             "features); layer2 = 1.5 -> 0.83 s/step on a GTX 1650")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4, help="backbone learning rate (head: 10x)")
    parser.add_argument("--limit", type=int, help="use only this many training images (smoke test)")
    parser.add_argument("--baseline-only", action="store_true", help="score the unmodified ETH-XGaze and stop")
    parser.add_argument("--workers", type=int, default=0,
                        help="data-loading processes (each needs >1 GB of memory; 0 = load in this process)")
    parser.add_argument("--amp", action="store_true", help="fp16 mixed precision (gave NaN on a GTX 1650)")
    args = parser.parse_args()
    global WORKERS, AMP
    WORKERS, AMP = args.workers, args.amp

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    run = RUNS / time.strftime("%Y%m%d-%H%M%S")
    run.mkdir(parents=True, exist_ok=True)
    log_file = open(run / "train.log", "a", encoding="utf-8")

    def log(message):
        print(message, flush=True)
        log_file.write(message + "\n")
        log_file.flush()

    everyone = sorted(p.stem for p in DATA.glob("p??.npz"))
    val_people = [] if args.val == ["none"] else args.val
    train_people = [p for p in everyone if p not in val_people]
    sets = {}
    if val_people:
        sets["held-out people"] = loader(mpii_items(val_people, args.limit and args.limit * 2), False, 32)
    sessions = sorted(d for d in (HERE / "sessions").glob("*") if (d / "session.npz").exists())
    if sessions:
        sets["your sessions"] = loader([i for d in sessions for i in session_items(d)], False, 32)
    log(f"run {run} | train on {train_people} | held out {val_people or 'none'} | "
        f"{len(sessions)} recorded session(s)")

    model = load_model().to(device).to(memory_format=torch.channels_last)
    log("ETH-XGaze (unmodified):")
    metrics = {"args": vars(args), "baseline": score(model, sets, device, log), "epochs": []}
    (run / "metrics.json").write_text(json.dumps(metrics, indent=1))
    if args.baseline_only:
        return

    blocks = ["conv1", "bn1", "layer1", "layer2", "layer3"]
    frozen = [] if args.freeze == "none" else blocks[:blocks.index(args.freeze) + 1]
    for name, parameter in model.backbone.named_parameters():
        parameter.requires_grad = name.split(".")[0] not in frozen
    train_items = mpii_items(train_people, args.limit)
    train_loader = loader(train_items, True, args.batch)
    head = list(model.gaze_fc.parameters())
    backbone = [p for n, p in model.named_parameters() if not n.startswith("gaze_fc.") and p.requires_grad]
    optimizer = torch.optim.AdamW([{"params": backbone, "lr": args.lr}, {"params": head, "lr": args.lr * 10}],
                                  weight_decay=1e-4)
    total = args.epochs * len(train_loader)
    warmup = min(200, total // 10)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: (s + 1) / warmup if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / (total - warmup))))
    scaler = torch.amp.GradScaler("cuda", enabled=AMP)
    key = "held-out people" if val_people else None
    best = metrics["baseline"][key]["flip"] if key else None
    log(f"training on {len(train_items)} images, {len(train_loader)} steps/epoch, {args.epochs} epochs, "
        f"frozen: {frozen or 'nothing'}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        for name in frozen:   # frozen blocks keep ETH-XGaze's batch-norm statistics
            getattr(model.backbone, name).eval()
        start, running = time.monotonic(), []
        for step, (images, labels) in enumerate(train_loader, 1):
            images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=AMP):
                gaze, _ = model(images)
                loss = torch.nn.functional.l1_loss(gaze.float(), labels)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            schedule.step()
            running.append(loss.item())
            if step % 200 == 0:
                log(f"  epoch {epoch} step {step}/{len(train_loader)}  loss {np.mean(running[-200:]):.4f}  "
                    f"{(time.monotonic() - start) / step:.2f} s/step  "
                    f"gpu {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
        log(f"epoch {epoch}: loss {np.mean(running):.4f}, {time.monotonic() - start:.0f} s")
        result = score(model, sets, device, log)
        metrics["epochs"].append(result)
        torch.save(model.state_dict(), run / "last.pth")
        if key is None or result[key]["flip"] < best:
            best = result[key]["flip"] if key else None
            torch.save(model.state_dict(), run / "best.pth")
            log("  saved best.pth")
        (run / "metrics.json").write_text(json.dumps(metrics, indent=1))
    if key:
        base = metrics["baseline"][key]["flip"]
        log(f"held-out people: ETH-XGaze {base:.2f} deg -> best fine-tuned {best:.2f} deg "
            f"({(base - best) / base * 100:+.0f}% {'better' if best < base else 'no improvement'})")


if __name__ == "__main__":
    main()
