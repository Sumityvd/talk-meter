WebcamGaze - test build
=======================

Shows a circular pointer on your screen where you are looking, using an ordinary webcam.
Your mouse is not affected: the pointer is only a display, and clicks go straight through it.


What you need
-------------
- Windows 10 or 11
- A webcam. An external USB webcam clipped on top of the screen works best; the laptop's built-in
  camera also works.
- About 450 MB of disk space for this folder.
- A graphics card helps (any DirectX 12 GPU) but is not required.


How to run
----------
1. Unzip the whole folder somewhere (for example to your Desktop). Keep all files together.
2. Double-click WebcamGaze.exe. A black console window opens; it shows status messages.
   Windows may warn that the app is from an unknown publisher: click "More info" -> "Run anyway".
3. Get into position (full-screen window with a live view of you):
   - Put your face in the oval and follow the hints until everything is green.
   - Several cameras? Press 1, 2, ... to pick one (the external webcam is chosen by default).
   - You can press SPACE to start anyway; the screen then tells you how much accuracy you lose.
4. Calibration takes about 2 minutes and runs by itself:
   - 16 dots: look at the black centre of each dot until it shrinks.
   - 6 dots with head movement: keep your eyes on the dot and move your head as the text asks,
     until the ring turns green.
   - A moving dot: follow it smoothly with your eyes for 20 seconds.
5. The orange ring now follows your gaze. A small "webcam-gaze" window shows the camera view.

Keys (in the small "webcam-gaze" window):
  q  quit
  r  calibrate again (do this whenever you sit differently, or the light changes)
  c  drift fix: look at the centre dot when it appears (Ctrl+Alt+G does the same from any window)
  p  position check
  v  accuracy test (about 1 minute; prints a result in the console)

Tips for good accuracy:
- Sit 50-70 cm from the screen, with your face lit from the front (no bright window behind you).
- Calibrate in the position you will use; recalibrate (r) after moving the laptop, camera or chair.
- Expect the ring to be within roughly 1-2 cm of where you look - it is a webcam, not an
  infrared eye tracker.


Options (run from a command prompt in this folder)
---------------------------------------------------
  WebcamGaze.exe --pointer-size 80     bigger pointer (default 60 pixels)
  WebcamGaze.exe --camera 0            use camera 0 (see --list-cameras)
  WebcamGaze.exe --list-cameras
  WebcamGaze.exe --benchmark           add the 1-minute accuracy test after calibration
  WebcamGaze.exe --self-test           check the installation without a camera
  WebcamGaze.exe --help                all options


If something goes wrong
-----------------------
- Run "WebcamGaze.exe --self-test" and send the output.
- Errors are written to %LOCALAPPDATA%\WebcamGaze\profiles\webcam_gaze.log - please send that file.
- "camera did not open or only sends black frames": close other apps using the camera (Teams, Zoom,
  the Camera app), open any privacy shutter, or pick another camera with --camera.
- Calibrations and test results are stored in %LOCALAPPDATA%\WebcamGaze (delete it to reset).


Licences
--------
For testing and non-commercial use only.
- Gaze model: ETH-XGaze (Zhang et al., ECCV 2020), CC BY-NC-SA 4.0 -
  https://github.com/xucong-zhang/ETH-XGaze . Non-commercial use only.
- Face landmarks: Google MediaPipe Face Landmarker, Apache 2.0.
- Built with Python, OpenCV, NumPy, ONNX Runtime (DirectML) and PyInstaller.
