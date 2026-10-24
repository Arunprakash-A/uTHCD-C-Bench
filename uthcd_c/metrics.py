"""Robustness metric for uTHCD-C.

The mean Corruption Error (mCE) follows the ImageNet-C convention with the
ratio-of-sums normalization: for each corruption the error is summed over the
five severity levels before dividing by the baseline, and the per-corruption
errors are then averaged.

Result files use the schema produced by ``scripts/evaluate_model.py``::

    {
      "model": "<name>",
      "clean":     {"top1": <accuracy %>},
      "corrupted": {"<corruption>": {"<severity>": {"top1": <error %>}}, ...}
    }

``clean.top1`` is an accuracy; every ``corrupted[...].top1`` is a top-1 error.
"""
from __future__ import annotations

import json
from statistics import mean, pstdev

from .corruptions import CORRUPTIONS, SEVERITIES


def load_results(path):
    with open(path) as f:
        return json.load(f)


def _errors(results, corruption):
    sev = results["corrupted"][corruption]
    return [sev[str(s)]["top1"] for s in SEVERITIES]


def corruption_error(model_results, baseline_results, corruption):
    """Ratio-of-sums CE for one corruption: sum_s E_model / sum_s E_baseline."""
    num = sum(_errors(model_results, corruption))
    den = sum(_errors(baseline_results, corruption))
    return num / den if den > 0 else 1.0


def per_corruption_ce(model_results, baseline_results, corruptions=None):
    corruptions = corruptions or CORRUPTIONS
    return {c: corruption_error(model_results, baseline_results, c) for c in corruptions}


def mce(model_results, baseline_results, corruptions=None):
    """Aggregate mCE: mean of the per-corruption ratio-of-sums CE."""
    ce = per_corruption_ce(model_results, baseline_results, corruptions)
    return sum(ce.values()) / len(ce)


def summarize(values):
    """Return (mean, population-std) of a list; std is 0 for a single value."""
    if not values:
        return float("nan"), float("nan")
    return mean(values), (pstdev(values) if len(values) > 1 else 0.0)
