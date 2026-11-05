"""The ETH-XGaze gaze network (ResNet-50 -> pitch/yaw), shared by train_gaze.py and the ONNX export.

ETH-XGaze: Zhang et al., ECCV 2020, https://github.com/xucong-zhang/ETH-XGaze (CC BY-NC-SA 4.0,
non-commercial). Fine-tuned versions of it carry the same licence.
"""

from pathlib import Path

import torch
import torchvision

MODELS = Path(__file__).resolve().parent / "models"
XGAZE_CHECKPOINT = MODELS / "xgaze_epoch_24_ckpt.pth.tar"


class XGaze(torch.nn.Module):
    """Input: normalized 224x224 face, RGB, ImageNet mean/std. Output: (pitch/yaw [N, 2], embedding [N, 2048])."""

    def __init__(self):
        super().__init__()
        self.backbone = torchvision.models.resnet50()
        self.backbone.fc = torch.nn.Identity()
        self.gaze_fc = torch.nn.Linear(2048, 2)

    def forward(self, face):
        embedding = self.backbone(face)
        return self.gaze_fc(embedding), embedding


def load_xgaze(checkpoint=XGAZE_CHECKPOINT):
    """The original ETH-XGaze weights. Its ResNet-50 uses torchvision's layer names; its own fc is unused."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)["model_state"]
    model = XGaze()
    backbone = {k[len("gaze_network."):]: v for k, v in state.items()
                if k.startswith("gaze_network.") and not k.startswith("gaze_network.fc.")}
    model.backbone.load_state_dict(backbone, strict=True)
    model.gaze_fc.load_state_dict({"weight": state["gaze_fc.0.weight"], "bias": state["gaze_fc.0.bias"]})
    return model


def load_model(path=None):
    """ETH-XGaze when `path` is None or the original checkpoint, else a state_dict saved by train_gaze.py."""
    if path is None or Path(path) == XGAZE_CHECKPOINT:
        return load_xgaze()
    model = XGaze()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model
