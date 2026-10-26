# Builds WebcamGaze.exe (PyInstaller, folder build) and zips it for sending to testers.
# Usage: powershell -ExecutionPolicy Bypass -File build_exe.ps1 [-Out D:\WebcamGazeBuild]
# Needs: .venv with requirements.txt + pyinstaller, models\face_landmarker.task, models\xgaze_resnet50.onnx.
# Output: <Out>\dist\WebcamGaze\WebcamGaze.exe and <Out>\WebcamGaze.zip (about 450 MB unzipped, 225 MB zipped).
param([string]$Out = "D:\WebcamGazeBuild")

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $here ".venv\Scripts\python.exe"
$models = Join-Path $here "models"

& $python -m PyInstaller --noconfirm --clean --name WebcamGaze --console `
    --distpath "$Out\dist" --workpath "$Out\build" --specpath $Out `
    --collect-all mediapipe --collect-binaries onnxruntime --hidden-import cv2_enumerate_cameras `
    --add-data "$models\face_landmarker.task;models" --add-data "$models\xgaze_resnet50.onnx;models" `
    (Join-Path $here "webcam_gaze.py")
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }

Copy-Item (Join-Path $here "TESTER_README.txt") "$Out\dist\WebcamGaze\README.txt"
& "$Out\dist\WebcamGaze\WebcamGaze.exe" --self-test
if ($LASTEXITCODE -ne 0) { throw "self-test failed" }

Remove-Item "$Out\WebcamGaze.zip" -ErrorAction SilentlyContinue
Compress-Archive -Path "$Out\dist\WebcamGaze" -DestinationPath "$Out\WebcamGaze.zip" -CompressionLevel Optimal
Get-Item "$Out\WebcamGaze.zip"
