"""uTHCD-C: a corruption-robustness benchmark for Tamil handwritten character recognition."""

from .corruptions import CORRUPTIONS, SEVERITIES, apply_corruption
from .metrics import corruption_error, per_corruption_ce, mce, load_results, summarize

__all__ = [
    "CORRUPTIONS", "SEVERITIES", "apply_corruption",
    "corruption_error", "per_corruption_ce", "mce", "load_results", "summarize",
]
__version__ = "1.0.0"
