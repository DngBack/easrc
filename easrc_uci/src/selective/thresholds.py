from __future__ import annotations

import numpy as np


def generate_threshold_grid(scores: np.ndarray, n_thresholds: int) -> np.ndarray:
    """
    Candidate thresholds for accept rule: accept iff score >= tau.

    Uses quantiles of scores on the calibration set plus endpoints so that
    nearly-empty and nearly-full acceptance regions are reachable.
    """
    scores = np.asarray(scores, dtype=np.float64)
    n_thresholds = max(int(n_thresholds), 3)
    smin, smax = float(scores.min()), float(scores.max())

    if smin == smax:
        span = max(abs(smin) * 1e-6, 1e-6)
        return np.array([smin - span, smin, smax + span], dtype=np.float64)

    qs = np.linspace(0.0, 1.0, n_thresholds)
    thr = np.quantile(scores, qs)
    thr = np.unique(np.concatenate([[smin - 1e-12], thr, [smax + 1e-12]]))
    return np.sort(thr.astype(np.float64))
