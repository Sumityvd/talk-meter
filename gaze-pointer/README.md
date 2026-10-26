# gaze-pointer

A webcam eye tracker for Windows. It shows a circular pointer on the screen where you are looking, without
taking over the mouse, and runs on an ordinary webcam with no Tobii hardware.

> This folder is a separate project from talk-meter and is **not part of the talk-meter site**: talk-meter
> builds only from `index.html` and `src/`, and `.vercelignore` keeps this folder out of uploads.

## What's here

| Path | What it is |
|---|---|
| [`tools/webcam-gaze/`](tools/webcam-gaze/) | **The main app.** A Python webcam gaze tracker with positioning window, a 2-minute calibration, an on-screen gaze pointer, a built-in accuracy benchmark, and a Windows `.exe` build script. [Read its README.](tools/webcam-gaze/README.md) |
| `lib/`, `apps/` | Microsoft's [GazePointer SDK](https://github.com/MSREnable/GazePointer) (C#/WPF) with our changes: a fix for the Tobii assembly crash, a `Network` (UDP) sensor that takes gaze from the webcam tracker, and SnapRadius (snap-to-button). |
| `external/` | Tobii SDK binaries, as shipped in Microsoft's repo. |
| `notes/` | A static analysis of the GazePointer DLLs. |
| `README.upstream.md` | Microsoft's original GazePointer README. |

## How the tracker works (short)

1. MediaPipe finds the face and eyes.
2. An appearance-based CNN (ETH-XGaze) estimates the gaze direction and intersects it with the screen.
3. A personal correction is fitted during the 2-minute calibration.

See [`tools/webcam-gaze/README.md`](tools/webcam-gaze/README.md) for setup, model downloads, measured
accuracy and options.

## Licences

- Microsoft's GazePointer SDK and our changes to it: MIT, see [`LICENSE`](LICENSE).
- **The gaze model is not included.** ETH-XGaze (Zhang et al., ECCV 2020) is CC BY-NC-SA 4.0, so it may be
  used for non-commercial purposes only; download it as described in the tool's README. The MediaPipe face
  landmarker model (Apache 2.0) is also downloaded separately.
- The Tobii SDK files in `external/` are as distributed in Microsoft's repository.
