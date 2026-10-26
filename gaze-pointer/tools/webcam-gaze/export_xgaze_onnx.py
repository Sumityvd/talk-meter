"""Converts the ETH-XGaze pretrained checkpoint to ONNX (run once).

Input:  models/xgaze_epoch_24_ckpt.pth.tar  (https://github.com/xucong-zhang/ETH-XGaze, "pre-trained model";
        Zhang et al., ECCV 2020; CC BY-NC-SA 4.0 - personal / non-commercial use only)
Output: models/xgaze_resnet50.onnx
        input  "face"      float32 [N, 3, 224, 224], normalized face patch, RGB, ImageNet mean/std
        output "gaze"      float32 [N, 2], pitch, yaw (radians) in the normalized camera
        output "embedding" float32 [N, 2048], penultimate ResNet-50 features (for personal calibration)

Usage: py -3.12 export_xgaze_onnx.py
"""

from pathlib import Path

import numpy as np
import torch
import torchvision

MODELS = Path(__file__).resolve().parent / "models"
CHECKPOINT = MODELS / "xgaze_epoch_24_ckpt.pth.tar"
OUTPUT = MODELS / "xgaze_resnet50.onnx"


class XGaze(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torchvision.models.resnet50()
        self.backbone.fc = torch.nn.Identity()
        self.gaze_fc = torch.nn.Linear(2048, 2)

    def forward(self, face):
        embedding = self.backbone(face)
        return self.gaze_fc(embedding), embedding


def main():
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)["model_state"]
    model = XGaze()
    # ETH-XGaze's ResNet-50 uses torchvision's layer names; its own fc is defined but unused.
    backbone = {k[len("gaze_network."):]: v for k, v in state.items()
                if k.startswith("gaze_network.") and not k.startswith("gaze_network.fc.")}
    model.backbone.load_state_dict(backbone, strict=True)
    model.gaze_fc.load_state_dict({"weight": state["gaze_fc.0.weight"], "bias": state["gaze_fc.0.bias"]})
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, dummy, str(OUTPUT), input_names=["face"], output_names=["gaze", "embedding"],
                      dynamic_axes={"face": {0: "n"}, "gaze": {0: "n"}, "embedding": {0: "n"}},
                      opset_version=17, dynamo=False)

    import onnxruntime as ort
    session = ort.InferenceSession(str(OUTPUT), providers=["CPUExecutionProvider"])
    gaze, embedding = session.run(None, {"face": dummy.numpy()})
    with torch.no_grad():
        expected, _ = model(dummy)
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size / 1e6:.0f} MB); "
          f"max |onnx - torch| = {np.abs(gaze - expected.numpy()).max():.2e}, embedding {embedding.shape}")


if __name__ == "__main__":
    main()
