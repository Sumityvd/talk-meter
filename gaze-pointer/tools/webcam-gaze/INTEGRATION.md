# Using the gaze model in your own app

This folder tracks where someone is looking on the screen using an ordinary webcam. The gaze model is
**ETH-XGaze** (a ResNet-50 by Zhang et al., ETH Zurich) converted to ONNX. Google's MediaPipe face
landmarker finds the face first. Everything runs locally from Python, with no internet or server.

**Licence first:** the ETH-XGaze model is licensed [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
- **Non-commercial only:** no paid app, no ads, no company product.
- **Credit:** credit ETH-XGaze wherever your app lists credits.
- **Share-alike:** if you pass the model on, or a model trained from it, it keeps this licence.

The MediaPipe model is Apache 2.0.

## 1. Setup

You need Python 3.12 on Windows. Other systems need one change, given below the commands.

```powershell
git clone --depth 1 https://github.com/Sumityvd/talk-meter.git
cd talk-meter\gaze-pointer\tools\webcam-gaze
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
mkdir models
curl.exe -L -o models\xgaze_resnet50.onnx https://github.com/Sumityvd/talk-meter/releases/download/gaze-model-v1/xgaze_resnet50.onnx
curl.exe -L -o models\face_landmarker.task https://github.com/Sumityvd/talk-meter/releases/download/gaze-model-v1/face_landmarker.task
```

The files are also listed at <https://github.com/Sumityvd/talk-meter/releases/tag/gaze-model-v1>, with checksums.

`requirements.txt` installs `onnxruntime-directml`, which runs the model on any DirectX 12 GPU on Windows,
taking about 9 ms per frame on a GTX 1650 and about 35 ms on the CPU. On Linux or macOS, install `onnxruntime`
instead; the code then uses the CPU automatically.

Check it works:

```powershell
.\.venv\Scripts\python.exe gaze_example.py --camera 0 --show
```

It prints a screen point for every frame, and shows the camera picture with the gaze point on a small screen.

## 2. Two ways to connect it to your app

### A. Run the tracker next to your app, and receive the gaze point (more accurate)

`webcam_gaze.py` is the complete tracker. It runs a 2-minute personal calibration at start-up (dots on the
screen), which removes most of the model's per-person error, and then smooths the output. Start it with UDP
output:

```powershell
.\.venv\Scripts\python.exe webcam_gaze.py --output udp --port 4242
```

It sends `"x,y"` text datagrams in physical screen pixels to `127.0.0.1:4242`, many times a second. Your app
only needs to listen:

```python
import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("127.0.0.1", 4242))
sock.setblocking(False)

def latest_gaze():
    """The newest gaze point, or None if nothing arrived since the last call."""
    point = None
    while True:
        try:
            data, _ = sock.recvfrom(64)
        except BlockingIOError:
            return point
        x, y = map(float, data.decode().split(","))
        point = (x, y)
```

Call `latest_gaze()` from your app's loop or a timer. The tracker can be in any language as far as your app is
concerned, and nothing else in your app has to change. `--camera` picks the camera (an index or part of its
name). `README.md` lists the other options.

### B. Call the model inside your app (simpler, less accurate)

```python
import cv2
from gaze_example import GazeEstimator

camera = cv2.VideoCapture(0)
ok, frame = camera.read()
gaze = GazeEstimator(
    frame_size=(frame.shape[1], frame.shape[0]),
    screen_px=(1920, 1080),        # screen resolution
    screen_mm=(344, 193),          # visible screen area in millimetres (this is a 15.6-inch laptop)
    camera_above_mm=8,             # lens to the top edge of the picture area: ~8 built-in, ~15 clip-on webcam
    hfov_deg=65,                   # camera's horizontal field of view; 65 if unknown
)
while ok:
    result = gaze(frame)           # None when there's no face or the person blinks
    if result is not None:
        x, y = result["screen_px"]
        # ... use x, y ...
    ok, frame = camera.read()
```

`gaze_example.py` needs `gaze_cnn.py` from the same folder. Copy both into your project, along with the
`models` folder.

- The camera must be at the **top centre** of the screen, and the screen size must be correct.
- The result also has `pitchyaw` (gaze angles), `direction` (a 3D gaze ray) and `face_center` (the face's
  position in mm from the camera).
- Run it in its own thread if your app's main loop can't wait about 10 to 40 ms per frame.

## 3. How accurate it is

Measured on a 15.6-inch laptop at about 55 cm, with a Microsoft LifeCam. At that distance, 1 degree is about
1 cm on the screen, or about 54 px.

| Setup | Typical error |
|---|---|
| B: the model alone, no calibration | about 5 to 11 degrees, often 300+ px, and mostly a constant offset in one direction that changes from sitting to sitting |
| A: the tracker with its 2-minute calibration | about 70 to 90 px (1.3 to 1.7 degrees) |
| For comparison, GazeFlow (commercial webcam tracker) | about 36 px |

So, use **A** to point at buttons or regions. **B** is good enough to tell left from right, or screen from away.

To improve B yourself, add a short calibration:
1. Show 5 to 9 dots.
2. For each dot, record the average `screen_px` while the person looks at it.
3. Fit a correction from predicted to true positions. Even a per-axis `scale * x + offset` removes most of the
   constant error, and `numpy.polyfit` per axis is enough.
4. Redo it whenever the person or camera moves a lot.

## 4. Credit

> Gaze estimation: ETH-XGaze, X. Zhang, S. Park, T. Beeler, D. Bradley, S. Tang, O. Hilliges.
> "ETH-XGaze: A Large Scale Dataset for Gaze Estimation under Extreme Head Pose and Gaze Variation", ECCV 2020.
> Model licensed CC BY-NC-SA 4.0. Face landmarks: Google MediaPipe (Apache 2.0).
