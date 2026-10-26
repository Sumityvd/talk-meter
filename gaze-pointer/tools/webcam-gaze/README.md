# webcam-gaze

Webcam eye tracker. By default it shows **a circular gaze pointer** (`--output overlay`): a ring about
60 px across, drawn on top of everything where you look. Your mouse stays entirely yours: the pointer is
click-through, never takes focus, and hides when no gaze is detected. `--pointer-size` sets its diameter.

Other outputs:

- `--output cursor`: moves the Windows cursor.
- `--output udp`: sends gaze to GazePointer's `Network` sensor (UDP `x,y` in physical screen pixels to
  `127.0.0.1:4242`) for dwell clicks.

No Tobii hardware needed.

Two engines:

| `--engine` | How it works |
|---|---|
| `cnn` (default) | ETH-XGaze ResNet-50 on a normalized face patch → 3D gaze ray → screen point, then a personal ridge head on the CNN's output and embedding (`gaze_cnn.py`) |
| `landmarks` | MediaPipe iris/eye-corner features → per-user polynomial fit. Best measured: 98 px / 1.67° |

Measured with `benchmark.py` on 2026-10-23. Setup: Microsoft LifeCam HD-6000 on top of a 15.6" laptop
screen, about 55 cm from the eyes. Error is the mean distance between where the cursor goes and the dot.

| Tracker | Head still | Head moved ~5 cm | Cursor jitter |
|---|---|---|---|
| GazeFlow GazePointer (closed source, for reference) | 36 px / 0.67° | 34 px | 32 px |
| `cnn` engine, calibrated and tested right away | 149 px / 2.8° | 79 px / 1.5° | 38 px |
| `cnn` engine, calibration a day old | 82 px / 1.5° | 92 px / 1.7° | 15 px |

Across two recorded sessions (calibrate in one, test in the other, plus the startup drift check), the
`cnn` engine levels off around 90 px (about 1.7°). GazeFlow is about 2.5× more accurate. Our engine gives a
steadier cursor. GazePointer's `SnapRadius` setting makes up some of the gap for clicking.

## Setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`models/face_landmarker.task` is MediaPipe's face landmarker
(<https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task>).

`cnn` engine model, once:

1. Download the ETH-XGaze pre-trained model (<https://github.com/xucong-zhang/ETH-XGaze>, "pre-trained model")
   to `models/xgaze_epoch_24_ckpt.pth.tar` (`py -3.12 -m gdown 1Ma6zJrECNTjo_mToZ5GKk7EF-0FS4nEC -O models/xgaze_epoch_24_ckpt.pth.tar`).
2. `py -3.12 export_xgaze_onnx.py` (needs CPU PyTorch + onnxruntime) → `models/xgaze_resnet50.onnx`.

ETH-XGaze (Zhang et al., ECCV 2020) is licensed CC BY-NC-SA 4.0: personal / non-commercial use only.
It runs on the GPU through DirectML (about 9 ms per frame on a GTX 1650) and falls back to the CPU (about 35 ms).

For GazePointerTest, `%APPDATA%\Microsoft\MSREnable\GazePointerTest.config`:

```xml
<AppSettings><Mouse SnapRadius="100"><Sensor Sensor="Network" NetworkPort="4242" /><Filter ActiveFilter="NullFilter" /></Mouse></AppSettings>
```

`NullFilter` because the tracker already smooths (fixation lock + glide); filtering twice only adds lag.

## Run

```powershell
.\.venv\Scripts\python.exe webcam_gaze.py                      # quick calibration, then the gaze pointer
.\.venv\Scripts\python.exe webcam_gaze.py --pointer-size 80     # bigger pointer
.\.venv\Scripts\python.exe webcam_gaze.py --output cursor       # move the mouse cursor instead
.\.venv\Scripts\python.exe webcam_gaze.py --list-cameras
```

`--camera` takes an index or part of the device name. Use a name: Windows renumbers cameras when one is
plugged in, unplugged or disabled. Starting with a saved calibration shows a one-dot drift check first
(`--no-start-drift-fix` skips it). `--refit` rebuilds the saved calibration from its recorded samples after a
model change, without recalibrating. Every 10 s the console prints a status line (fps, face found, blinks,
rejected frames). It warns when the camera picture is too dark to find a face.

`--camera auto` (the default) picks an external USB webcam over the laptop's built-in camera. It also sets
the field of view from the camera name (LifeCam HD-6000: 59°) and the lens height above the screen
(15 mm for a clip-on webcam, 8 mm for a built-in one).

**Positioning window.** Calibration starts with a mirrored live view of you, an oval to put your face in,
and a live checklist. Each failing item shows what to do:

- distance (45–75 cm, from the pupil gap in the image)
- centred left/right
- camera aim (tilt up/down)
- head straight
- lighting (too dark, too bright, or a bright window behind you)
- eyes visible

The top line lists every connected camera (built-in ones are marked). Press its number to switch:
the live view changes straight away, and the calibration is saved for the camera you chose.
Calibration starts by itself once everything has been green for 2 s.

**Starting anyway.** You can start at any time with SPACE, even while some checks are red; only a visible
face is required. The bottom line always shows the accuracy to expect if you start now ("slightly
reduced" to "much reduced"). After SPACE, a 5-second screen lists what each unmet check costs, for
example *"Distance 84 cm: your eyes look smaller to the camera: expect roughly 1.5x the error and more
jitter"*. Calibration then continues by itself; ESC goes back to positioning. Only the distance effect is
given as a number, because it follows from how large the eyes appear in the image. The others are described
in words. Unmet checks are saved with the calibration (`posture.warnings` in the profile) and printed. Your position is saved with the
calibration. On later starts, the same window guides you back to that position before the drift check,
because sitting differently from the calibration was the largest error measured. Press `p` while
streaming to run it again.

**Guided head movement (calibration part 2).** Your eyes stay on each dot while you move your head as
the text under it asks: turn left/right, chin up/down, lean closer/back, or shift sideways. A ring fills
as the camera sees you move, and samples are taken only once it turns green. This way the model learns
real head movement. Without guidance, a recorded calibration moved less than 1 cm, and that calibration
then failed when the head moved (192 px against 49 px with the head still).

**Every start calibrates (about 2 minutes).** A calibration hardly carries over to a new sitting. On
recordings from 2026-10-23, average error in a new sitting was:

| How the new sitting starts | Error |
|---|---|
| Saved calibration + drift dot | 152 px |
| 40 s top-up | 90 px |
| Full fresh calibration | 59 px |

So `--calibration quick` (16 dots, 6 guided head moves, 20 s moving dot) is the default.

- `--calibration full` takes about 6 minutes (24 dots, 12 head moves, 30 s moving dot) and ends with the accuracy test.
- `--benchmark` adds the accuracy test to a quick calibration; `v` runs it any time.
- `--calibration saved` reuses the saved calibration with a position check and drift dot: fast, but much less accurate.

The first run calibrates. Part 1 is 24 dots, part 2 is 12 dots while you move your head, and part 3
(cnn engine only) is a 30 s moving dot you follow with your eyes. A 9-point accuracy test follows.
Each engine keeps its own calibration in `profiles/` (`calibration.json`, `calibration_cnn.json`); `--recalibrate` forces a new one.

Camera geometry for the `cnn` engine:

- `--hfov-deg` is the camera's horizontal field of view. The LifeCam HD-6000 has a 66° diagonal, which is about 59° horizontal. Built-in laptop cameras are about 65°.
- `--camera-above-mm` is the distance from the lens to the top edge of the visible screen. Mount an
  external camera centred on top of the screen, facing you, in the plane of the screen.
- Calibration corrects small errors in both.

Status-window keys: `q` quit, `r` recalibrate, `v` re-run the accuracy test, `c` drift fix.
**Ctrl+Alt+G** runs the drift fix from anywhere (look at the centre dot).
With `--output cursor`, moving the real mouse pauses gaze for 1 s.

`--learn-from-clicks` makes GazePointer's invoked targets (reported on UDP port + 1) correct drift
automatically. Leave it off when buttons sit right next to text you read.

## Benchmark against other trackers

Every calibration ends with this same test, run by the tracker itself. Pass 1 is 16 dots with your head
still; pass 2 is 9 dots after moving your head about 5 cm. It uses the same timing, keeps every sample, and
is scored on the cursor path the tracker produces. The result screen shows it next to GazeFlow's
`benchmark.py` result. Each run is saved in `bench/` as `ours-calibration`, so `benchmark.py --summary`
tracks it over time.

`benchmark.py` measures wherever the **system cursor** goes while you look at targets. That way any engine that
moves the cursor, including GazeFlow's GazePointer, is scored the same way. Pass 1 is 16 targets with your
normal posture. Pass 2 is 9 targets after shifting your head about 5 cm.

```powershell
# 1. GazeFlow GazePointer running and calibrated (close webcam_gaze first - one camera user at a time)
.\.venv\Scripts\python.exe benchmark.py --label gazeflow
# 2. our engine moving the cursor (close GazeFlow first)
.\.venv\Scripts\python.exe webcam_gaze.py --camera LifeCam --hfov-deg 59 --camera-above-mm 15 --output cursor
.\.venv\Scripts\python.exe benchmark.py --label ours-cnn          # in a second terminal
# 3. median of all runs per label
.\.venv\Scripts\python.exe benchmark.py --summary
```

Do 3 runs per engine with the same camera, lighting and seat. Results are saved in `bench/`.

## Getting the best accuracy

- Light your face from the front; avoid a bright window behind you.
- Put the camera centred above the screen, 50–70 cm from your eyes.
- Keep the posture you calibrated in; recalibrate (`r`) after moving the laptop, camera or chair.
- Set `--distance-cm` to your real eye-to-screen distance so the reported degrees are right.
