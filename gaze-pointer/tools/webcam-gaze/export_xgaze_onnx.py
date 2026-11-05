"""Converts gaze model weights to the ONNX file the tracker runs (gaze_cnn.py).

Input:  the ETH-XGaze checkpoint models/xgaze_epoch_24_ckpt.pth.tar (https://github.com/xucong-zhang/ETH-XGaze,
        "pre-trained model"; Zhang et al., ECCV 2020; CC BY-NC-SA 4.0 - personal / non-commercial use only),
        or with --weights a fine-tuned state_dict from train_gaze.py / train_personal.py (same licence)
Output: an ONNX file (default models/xgaze_resnet50.onnx)
        input  "face"      float32 [N, 3, 224, 224], normalized face patch, RGB, ImageNet mean/std
        output "gaze"      float32 [N, 2], pitch, yaw (radians) in the normalized camera
        output "embedding" float32 [N, 2048], penultimate ResNet-50 features (for personal calibration)

Usage: D:\\gaze-train\\.venv\\Scripts\\python.exe export_xgaze_onnx.py [--weights best.pth --output models\\x.onnx]
"""

import argparse

import numpy as np
import torch

from xgaze_model import MODELS, load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", help="state_dict from train_gaze.py / train_personal.py (default: ETH-XGaze)")
    parser.add_argument("--output", default=str(MODELS / "xgaze_resnet50.onnx"))
    args = parser.parse_args()

    model = load_model(args.weights).eval()
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, dummy, args.output, input_names=["face"], output_names=["gaze", "embedding"],
                      dynamic_axes={"face": {0: "n"}, "gaze": {0: "n"}, "embedding": {0: "n"}},
                      opset_version=17, dynamo=False)

    import onnxruntime as ort
    session = ort.InferenceSession(args.output, providers=["CPUExecutionProvider"])
    gaze, embedding = session.run(None, {"face": dummy.numpy()})
    with torch.no_grad():
        expected, _ = model(dummy)
    print(f"wrote {args.output}; max |onnx - torch| = {np.abs(gaze - expected.numpy()).max():.2e}, "
          f"embedding {embedding.shape}")


if __name__ == "__main__":
    main()
