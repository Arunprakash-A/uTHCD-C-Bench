#!/usr/bin/env python3
"""Aggregate per-model result files into the paper's robustness tables.

Reproduces, from ``results/raw/``:
  * the aggregate mCE table (5-fold mean +/- std),
  * the per-corruption CE matrix used for the heatmap,
  * the writer-independent vs writer-dependent mCE table.

All values use the ratio-of-sums mCE (see ``uthcd_c.metrics``) with VGG16-BN as
the reference model.

Example
-------
    python scripts/aggregate.py --raw-dir results/raw --out-dir results/tables
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from uthcd_c.corruptions import CORRUPTIONS  # noqa: E402
from uthcd_c.metrics import load_results, per_corruption_ce, mce, summarize  # noqa: E402

BASELINE = "vgg16_bn"


def _models(path):
    return sorted(d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)))


def aggregate_kfold(kfold_dir):
    """Return (mce_summary, heatmap) over folds; baseline is per-fold VGG16-BN."""
    models = _models(kfold_dir)
    folds = sorted(os.listdir(os.path.join(kfold_dir, BASELINE)))
    baseline = {f: load_results(os.path.join(kfold_dir, BASELINE, f, "corruption_results.json"))
                for f in folds}
    mce_summary, heatmap = {}, {}
    for m in models:
        per_fold_mce, per_fold_ce = [], {c: [] for c in CORRUPTIONS}
        for f in folds:
            rp = os.path.join(kfold_dir, m, f, "corruption_results.json")
            if not os.path.exists(rp):
                continue
            res = load_results(rp)
            ce = per_corruption_ce(res, baseline[f])
            per_fold_mce.append(sum(ce.values()) / len(ce))
            for c in CORRUPTIONS:
                per_fold_ce[c].append(ce[c])
        mu, sd = summarize(per_fold_mce)
        mce_summary[m] = (mu, sd)
        heatmap[m] = {c: summarize(per_fold_ce[c])[0] for c in CORRUPTIONS}
    return mce_summary, heatmap


def aggregate_writer(wi_dir, wd_dir):
    """Writer-independent mCE (mean+/-std over seeds) and writer-dependent mCE."""
    seeds = sorted(os.listdir(wi_dir))
    models = _models(os.path.join(wi_dir, seeds[0]))
    wd_baseline = load_results(os.path.join(wd_dir, BASELINE, "corruption_results.json"))
    out = {}
    for m in models:
        wd = load_results(os.path.join(wd_dir, m, "corruption_results.json"))
        wd_mce = mce(wd, wd_baseline)
        wi_vals = []
        for s in seeds:
            base = load_results(os.path.join(wi_dir, s, BASELINE, "corruption_results.json"))
            res = load_results(os.path.join(wi_dir, s, m, "corruption_results.json"))
            wi_vals.append(mce(res, base))
        out[m] = (wd_mce, *summarize(wi_vals))
    return out


def write_csv(path, header, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print("wrote", path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", default="results/raw")
    p.add_argument("--out-dir", default="results/tables")
    args = p.parse_args()

    mce_summary, heatmap = aggregate_kfold(os.path.join(args.raw_dir, "kfold"))
    rows = sorted(([m, round(mu, 4), round(sd, 4)] for m, (mu, sd) in mce_summary.items()),
                  key=lambda r: r[1])
    write_csv(os.path.join(args.out_dir, "mce_aggregate.csv"),
              ["model", "mce_mean", "mce_std"], rows)

    hm_rows = [[m] + [round(heatmap[m][c], 4) for c in CORRUPTIONS] for m in heatmap]
    hm_rows.sort(key=lambda r: mce_summary[r[0]][0])
    write_csv(os.path.join(args.out_dir, "mce_per_corruption.csv"),
              ["model"] + CORRUPTIONS, hm_rows)
    with open(os.path.join(args.out_dir, "mce_per_corruption.json"), "w") as f:
        json.dump({"heatmap": heatmap,
                   "aggregate": {m: list(v) for m, v in mce_summary.items()}}, f, indent=1)

    writer = aggregate_writer(os.path.join(args.raw_dir, "writer_independent"),
                              os.path.join(args.raw_dir, "writer_dependent_70"))
    wrows = sorted(([m, round(wd, 4), round(mu, 4), round(sd, 4)]
                    for m, (wd, mu, sd) in writer.items()), key=lambda r: r[2])
    write_csv(os.path.join(args.out_dir, "mce_writer_independent.csv"),
              ["model", "writer_dependent_mce", "writer_independent_mce_mean", "writer_independent_mce_std"],
              wrows)

    print("\nAggregate mCE (ratio-of-sums, 5-fold):")
    for m, mu, sd in rows:
        print(f"  {m:20s} {mu:.3f} +/- {sd:.3f}")


if __name__ == "__main__":
    main()
