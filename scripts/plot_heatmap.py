#!/usr/bin/env python3
"""Render the per-corruption mCE heatmap from ``mce_per_corruption.json``.

    python scripts/aggregate.py          # produces results/tables/mce_per_corruption.json
    python scripts/plot_heatmap.py --in results/tables/mce_per_corruption.json \
        --out results/tables/mce_heatmap.png
"""
import argparse
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

DISPLAY = {
    "vgg16_bn": "VGG16-BN", "swin_t": "Swin", "convnext_tiny": "ConvNeXt",
    "googlenet": "GoogLeNet", "efficientnet_b0": "EfficientNet", "squeezenet1_0": "SqueezeNet",
    "regnet_x_400mf": "RegNet", "shufflenet_v2_x0_5": "ShuffleNet", "mnasnet0_5": "MnasNet",
}
CORR_LABELS = {
    "gaussian_noise": "Gauss.\nnoise", "shot_noise": "Shot\nnoise", "impulse_noise": "Impulse\nnoise",
    "gaussian_blur": "Gauss.\nblur", "defocus_blur": "Defocus\nblur", "stroke_thinning": "Stroke\nthinning",
    "elastic": "Elastic", "pixelate": "Pixelate", "contrast": "Contrast", "scale": "Scale",
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="inp", default="results/tables/mce_per_corruption.json")
    p.add_argument("--out", default="results/tables/mce_heatmap.png")
    args = p.parse_args()

    data = json.load(open(args.inp))
    heatmap, agg = data["heatmap"], data["aggregate"]
    corr = list(CORR_LABELS.keys())
    models = sorted(heatmap.keys(), key=lambda m: agg[m][0])

    mat = np.array([[heatmap[m][c] for c in corr] for m in models])
    row_mean = np.array([agg[m][0] for m in models])
    full = np.hstack([mat, row_mean[:, None]])
    col_labels = [CORR_LABELS[c] for c in corr] + ["mCE\n(aggregate)"]

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    norm = TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=3.0)
    ax.imshow(np.clip(full, 0, 3.0), aspect="auto", cmap="RdBu_r", norm=norm)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=8.5)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([DISPLAY.get(m, m) for m in models], fontsize=9.5)
    ax.axvline(len(corr) - 0.5, color="black", lw=1.4)
    for i in range(len(models)):
        for j in range(len(col_labels)):
            v = full[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.6,
                    color="white" if (v < 0.45 or v > 1.9) else "black",
                    fontweight="bold" if j == len(corr) else "normal")
    im = ax.images[0]
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, extend="max")
    cbar.set_label("Corruption Error relative to VGG16-BN (CE; 1.0 = reference)", fontsize=8.5)
    ax.set_title("Per-corruption mCE by architecture (ratio-of-sums; lower is better)",
                 fontsize=10.5, pad=10)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
