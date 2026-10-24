"""Model builders for the nine reference architectures.

Each model is created from its torchvision definition and its final
classification layer is replaced to output ``num_classes`` logits. New models
can be benchmarked by adding an entry here or by loading any ``nn.Module`` whose
output matches the uTHCD class count.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm

SUPPORTED_MODELS = [
    "vgg16_bn", "googlenet", "efficientnet_b0", "regnet_x_400mf",
    "mnasnet0_5", "shufflenet_v2_x0_5", "squeezenet1_0",
    "convnext_tiny", "swin_t",
]


def build_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Build one reference architecture with its head resized to ``num_classes``."""
    name = name.lower()
    weights = "DEFAULT" if pretrained else None

    if name == "vgg16_bn":
        m = tvm.vgg16_bn(weights=weights)
        m.classifier[6] = nn.Linear(m.classifier[6].in_features, num_classes)
    elif name == "googlenet":
        m = tvm.googlenet(weights=weights)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        for aux_name in ("aux1", "aux2"):
            aux = getattr(m, aux_name, None)
            if aux is not None and hasattr(aux, "fc2"):
                aux.fc2 = nn.Linear(aux.fc2.in_features, num_classes)
    elif name == "efficientnet_b0":
        m = tvm.efficientnet_b0(weights=weights)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif name == "regnet_x_400mf":
        m = tvm.regnet_x_400mf(weights=weights)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "mnasnet0_5":
        m = tvm.mnasnet0_5(weights=weights)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif name == "shufflenet_v2_x0_5":
        m = tvm.shufflenet_v2_x0_5(weights=weights)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "squeezenet1_0":
        m = tvm.squeezenet1_0(weights=weights)
        m.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
    elif name == "convnext_tiny":
        m = tvm.convnext_tiny(weights=weights)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
    elif name == "swin_t":
        m = tvm.swin_t(weights=weights)
        m.head = nn.Linear(m.head.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model '{name}'. Choose from {SUPPORTED_MODELS}.")
    return m


def load_checkpoint(model: nn.Module, path: str, device="cpu") -> nn.Module:
    """Load weights from a checkpoint that is either a raw state_dict or a dict
    containing a ``state_dict`` / ``model_state`` entry."""
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model_state", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
    model.load_state_dict(ckpt)
    return model
