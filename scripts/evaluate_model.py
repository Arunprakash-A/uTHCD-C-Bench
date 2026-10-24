#!/usr/bin/env python3
"""Benchmark a single model on uTHCD-C.

Runs the model on the clean test set and on every corruption x severity, and
writes a ``corruption_results.json`` file holding the clean accuracy and the
top-1 error of each corruption/severity. If a baseline result file is given, the
mean Corruption Error (mCE) is also printed.

Example
-------
    python scripts/evaluate_model.py \
        --data-dir /path/to/uTHCD/test \
        --arch convnext_tiny \
        --checkpoint /path/to/model.pt \
        --output results/my_model/corruption_results.json \
        --baseline results/raw/kfold/vgg16_bn/fold_0/corruption_results.json
"""
import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from uthcd_c.corruptions import CORRUPTIONS, SEVERITIES  # noqa: E402
from uthcd_c.data import CorruptedImageFolder, build_eval_transform  # noqa: E402
from uthcd_c.models import build_model, load_checkpoint  # noqa: E402
from uthcd_c.metrics import mce  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a model on the uTHCD-C corruption suite.")
    p.add_argument("--data-dir", required=True,
                   help="Test set as an ImageFolder (one sub-directory per class).")
    p.add_argument("--output", required=True, help="Path to write corruption_results.json.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--arch", help="torchvision architecture name (see uthcd_c.models.SUPPORTED_MODELS).")
    g.add_argument("--model-script",
                   help="Python file exposing build() -> nn.Module, for a custom model.")
    p.add_argument("--checkpoint", help="Checkpoint with trained weights (state_dict).")
    p.add_argument("--baseline", help="Baseline result file for mCE normalization (e.g. VGG16-BN).")
    p.add_argument("--num-classes", type=int, default=None,
                   help="Override class count (default: inferred from --data-dir).")
    p.add_argument("--no-pretrained", action="store_true",
                   help="Do not load ImageNet weights before the checkpoint.")
    p.add_argument("--corruptions", nargs="+", default=CORRUPTIONS, choices=CORRUPTIONS)
    p.add_argument("--severities", nargs="+", type=int, default=SEVERITIES)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0, help="Seed for the stochastic corruptions.")
    return p.parse_args()


@torch.no_grad()
def top1_error(model, loader, device):
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return 100.0 * (1.0 - correct / total)


def main():
    args = parse_args()
    import numpy as np
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    transform = build_eval_transform()
    dataset = CorruptedImageFolder(args.data_dir, transform=transform)
    num_classes = args.num_classes or len(dataset.classes)

    if args.arch:
        model = build_model(args.arch, num_classes, pretrained=not args.no_pretrained)
        model_name = args.arch
    else:
        import importlib.util
        spec = importlib.util.spec_from_file_location("user_model", args.model_script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        model = mod.build()
        model_name = os.path.splitext(os.path.basename(args.model_script))[0]

    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, device=args.device)
    model = model.to(args.device).eval()

    def run():
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=(args.device == "cuda"))

    dataset.set_corruption("clean")
    clean_err = top1_error(model, run(), args.device)
    print(f"[clean] accuracy = {100 - clean_err:.2f}%")

    corrupted = {}
    for c in args.corruptions:
        corrupted[c] = {}
        for s in args.severities:
            dataset.set_corruption(c, s)
            err = top1_error(model, run(), args.device)
            corrupted[c][str(s)] = {"top1": err}
            print(f"[{c} s{s}] error = {err:.2f}%")

    results = {
        "model": model_name,
        "clean": {"top1": 100 - clean_err},
        "corrupted": corrupted,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nWrote {args.output}")

    if args.baseline:
        with open(args.baseline) as f:
            baseline = json.load(f)
        print(f"mCE (vs baseline) = {mce(results, baseline, args.corruptions):.3f}")


if __name__ == "__main__":
    main()
