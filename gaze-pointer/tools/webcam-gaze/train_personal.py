"""Stage B: fine-tunes the stage A model on one person's recorded calibrations (sessions/*).

Leave-one-session-out, so every recorded sitting gets a model that has never seen it (fold_<session>.pth);
compare_models.py scores those on the unseen sitting. Then final.pth is trained on all sessions for real use.
Only the last ResNet block and the gaze head train, with a small learning rate, so the model adjusts to this
person and camera without forgetting what it learned from everyone else.

Run with the CUDA training venv:
  D:\\gaze-train\\.venv\\Scripts\\python.exe train_personal.py --base D:\\gaze-train\\runs\\<stage A run>\\last.pth
Output: D:\\gaze-train\\personal\\<time>\\ (train.log, fold_*.pth, final.pth, metrics.json)
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from train_gaze import HERE, evaluate, loader, session_items
from xgaze_model import load_model

OUTPUT = Path(r"D:\gaze-train\personal")


def fine_tune(base, sessions, epochs, lr, batch, device, log, fixations_only=False):
    model = load_model(base).to(device).to(memory_format=torch.channels_last)
    for name, parameter in model.backbone.named_parameters():
        parameter.requires_grad = name.startswith("layer4")
    items = [item for d in sessions for item in session_items(d, fixations_only=fixations_only)]
    data = loader(items, True, batch)
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": lr},
        {"params": model.gaze_fc.parameters(), "lr": lr * 10}], weight_decay=1e-4)
    total = epochs * len(data)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 0.5 * (1 + math.cos(math.pi * s / total)))
    for epoch in range(1, epochs + 1):
        model.train()
        for block in ("conv1", "bn1", "layer1", "layer2", "layer3"):   # keep the stage A batch-norm statistics
            getattr(model.backbone, block).eval()
        start, running = time.monotonic(), []
        for images, labels in data:
            images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            gaze, _ = model(images)
            loss = torch.nn.functional.l1_loss(gaze, labels.to(device, non_blocking=True))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            schedule.step()
            running.append(loss.item())
        log(f"  epoch {epoch}: loss {np.mean(running):.4f}, {time.monotonic() - start:.0f} s")
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True, help="stage A weights (train_gaze.py output)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5, help="last block learning rate (head: 10x)")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--fixations-only", action="store_true",
                        help="train only on the still dots, not the moving-dot frames (their labels look less exact)")
    parser.add_argument("--final-only", action="store_true", help="skip the held-out folds, just train final.pth")
    parser.add_argument("--live-folds-only", action="store_true",
                        help="only hold out sittings that have an accuracy test, and skip final.pth (faster experiments)")
    parser.add_argument("--hold-out", nargs="+", help="only hold out these sittings, and skip final.pth")
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    run = OUTPUT / time.strftime("%Y%m%d-%H%M%S")
    run.mkdir(parents=True, exist_ok=True)
    log_file = open(run / "train.log", "a", encoding="utf-8")

    def log(message):
        print(message, flush=True)
        log_file.write(message + "\n")
        log_file.flush()

    sessions = sorted(d for d in (HERE / "sessions").glob("*") if (d / "session.npz").exists())
    log(f"run {run} | base {args.base} | sessions {[d.name for d in sessions]}")
    base_model = load_model(args.base).to(device).to(memory_format=torch.channels_last)
    metrics = {"args": vars(args), "folds": {}}
    held_outs = [] if args.final_only else sessions
    if args.live_folds_only:
        held_outs = [d for d in sessions if np.any(np.load(d / "session.npz")["group_pass"] == 2)]
    if args.hold_out:
        held_outs = [d for d in sessions if d.name in args.hold_out]
        assert len(held_outs) == len(args.hold_out), "unknown sitting in --hold-out"
    for held_out in held_outs:
        test = loader(session_items(held_out), False, 32)
        before = evaluate(base_model, test, device)
        log(f"held out {held_out.name}: stage A {before[1]:.2f} deg (plain {before[0]:.2f})")
        model = fine_tune(args.base, [d for d in sessions if d != held_out], args.epochs, args.lr, args.batch,
                          device, log, args.fixations_only)
        after = evaluate(model, test, device)
        log(f"  personal {after[1]:.2f} deg (plain {after[0]:.2f})")
        torch.save(model.state_dict(), run / f"fold_{held_out.name}.pth")
        metrics["folds"][held_out.name] = {"stage_a": before, "personal": after}
        (run / "metrics.json").write_text(json.dumps(metrics, indent=1))
        del model
        torch.cuda.empty_cache()
    if held_outs:
        mean = lambda key: float(np.mean([f[key][1] for f in metrics["folds"].values()]))
        log(f"unseen sitting, mean of folds: stage A {mean('stage_a'):.2f} deg -> personal {mean('personal'):.2f} deg")
    if args.live_folds_only or args.hold_out:
        return
    log("final model on all sessions:")
    torch.save(fine_tune(args.base, sessions, args.epochs, args.lr, args.batch, device, log,
                         args.fixations_only).state_dict(),
               run / "final.pth")
    log("saved final.pth")


if __name__ == "__main__":
    main()
