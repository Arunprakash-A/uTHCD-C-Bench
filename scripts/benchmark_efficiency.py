#!/usr/bin/env python3
"""Reproduce the efficiency characteristics reported in Table 6 of the paper:
parameter count, computational complexity (GMACs), and inference throughput.

All three quantities depend only on the architecture (not on trained weights),
so this script needs neither the dataset nor checkpoints; it builds each model
with a randomly initialized 156-class head.

Protocol (matches the paper, Section 3.3):
  * NVIDIA A100, FP32, 224x224 input.
  * Parameter count: sum of elements over all parameter tensors.
  * GMACs: multiply-accumulate operations for one forward pass, via an
    operation-counting profiler (thop) at batch size 1.
  * Throughput: evaluation mode under torch.no_grad(); batch size 128;
    20 warm-up batches then 60 timed batches; torch.cuda.synchronize()
    immediately before and after the timed region; time.perf_counter timing;
    throughput = (batch_size * timed_batches) / elapsed_seconds.

Usage:
    python3 scripts/benchmark_efficiency.py

Requires: torch, torchvision, thop.
"""
import argparse
import time
import warnings

import torch
import torch.nn as nn
import torchvision.models as M

warnings.filterwarnings("ignore")

NUM_CLASSES = 156
INPUT = (3, 224, 224)

# exact torchvision variants evaluated in the paper
BUILDERS = {
    "ShuffleNet":   M.shufflenet_v2_x0_5,
    "SqueezeNet":   M.squeezenet1_0,
    "MnasNet":      M.mnasnet0_5,
    "EfficientNet": M.efficientnet_b0,
    "RegNet":       M.regnet_x_400mf,
    "GoogLeNet":    M.googlenet,
    "Swin":         M.swin_t,
    "ConvNeXt":     M.convnext_tiny,
    "VGG16-BN":     M.vgg16_bn,
}


def build(name):
    """Build the architecture with its classification head resized to 156
    Tamil character classes (mirrors uthcd_c.models)."""
    if name == "GoogLeNet":
        m = BUILDERS[name](weights=None, aux_logits=False, init_weights=True)
    else:
        m = BUILDERS[name](weights=None)
    if name == "VGG16-BN":
        m.classifier[6] = nn.Linear(m.classifier[6].in_features, NUM_CLASSES)
    elif name in ("GoogLeNet", "RegNet", "ShuffleNet"):
        m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    elif name in ("EfficientNet", "MnasNet"):
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, NUM_CLASSES)
    elif name == "SqueezeNet":
        m.classifier[1] = nn.Conv2d(512, NUM_CLASSES, kernel_size=1)
    elif name == "ConvNeXt":
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, NUM_CLASSES)
    elif name == "Swin":
        m.head = nn.Linear(m.head.in_features, NUM_CLASSES)
    return m


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def count_gmacs(model):
    """Multiply-accumulate operations for one forward pass, at batch size 1.

    Prefers fvcore's FlopCountAnalysis, whose MAC-equivalent counts match the
    torchvision reference figures reported in the paper. Falls back to thop if
    fvcore is unavailable; note that thop under-reports attention operations and
    therefore undercounts transformer models such as Swin."""
    x = torch.randn(1, *INPUT)
    try:
        from fvcore.nn import FlopCountAnalysis
        fca = FlopCountAnalysis(model.eval(), x)
        fca.unsupported_ops_warnings(False)
        fca.uncalled_modules_warnings(False)
        return fca.total() / 1e9
    except Exception:
        from thop import profile
        macs, _ = profile(model, inputs=(x,), verbose=False)
        return macs / 1e9


@torch.no_grad()
def measure_throughput(model, device, batch_size, warmup, timed):
    model = model.to(device).eval()
    x = torch.randn(batch_size, *INPUT, device=device)
    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(timed):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return batch_size * timed / elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--timed", type=int, default=60)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no_throughput", action="store_true",
                    help="skip timing (e.g. on a machine without a GPU)")
    args = ap.parse_args()
    device = torch.device(args.device)

    print(f"# device={device} precision=FP32 input={INPUT[1]}x{INPUT[2]} "
          f"batch_size={args.batch_size} warmup={args.warmup} timed={args.timed}")
    print(f"{'Model':14s} {'Params(M)':>10s} {'GMACs':>8s} {'Throughput(img/s)':>18s}")
    print("-" * 54)
    for name in BUILDERS:
        model = build(name)
        params = count_params(model) / 1e6
        gmacs = count_gmacs(model)
        if args.no_throughput or device.type == "cpu":
            thr = float("nan")
        else:
            thr = measure_throughput(model, device, args.batch_size, args.warmup, args.timed)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"{name:14s} {params:10.2f} {gmacs:8.2f} {thr:18.0f}")


if __name__ == "__main__":
    main()
