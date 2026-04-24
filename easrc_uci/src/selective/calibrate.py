from __future__ import annotations

from typing import Any

import numpy as np

from src.selective.risk_coverage import (
    metrics_at_threshold,
    ucb_bounds,
    ucb_epsilon,
)
from src.selective.thresholds import generate_threshold_grid


def sweep_and_pick_threshold(
    scores_cal: np.ndarray,
    cls_cal: np.ndarray,
    xai_cal: np.ndarray,
    audited_cal: np.ndarray,
    *,
    alpha: float,
    beta: float,
    delta: float,
    n_thresholds: int,
    use_ucb: bool,
    min_empirical_coverage: float,
) -> tuple[float | None, list[dict[str, Any]]]:
    """
    Sweep thresholds on calibration; pick feasible tau with largest empirical coverage.

    Returns (best_tau or None if infeasible, list of per-threshold rows for logging).
    """
    scores_cal = np.asarray(scores_cal, dtype=np.float64)
    n_cal = int(len(scores_cal))

    thresholds = generate_threshold_grid(scores_cal, n_thresholds)
    n_t = len(thresholds)
    eps = ucb_epsilon(n_cal, n_t, delta) if use_ucb else 0.0

    rows: list[dict[str, Any]] = []
    best_tau: float | None = None
    best_cov = -1.0

    for tau in thresholds:
        m = metrics_at_threshold(scores_cal, cls_cal, xai_cal, audited_cal, float(tau))
        cov = m["coverage"]
        cal_cls = m["cal_cls_risk"]
        cal_xai = m["cal_xai_risk"]
        cal_aud = m["cal_audited_risk"]
        mean_il = m["mean_il"]
        mean_ixai = m["mean_ixai"]
        n_acc = m["n_accepted"]

        if use_ucb:
            cov_lcb, ucb_cls, ucb_xai = ucb_bounds(cov, mean_il, mean_ixai, eps)
            feasible = (
                cov_lcb > 0
                and not np.isnan(ucb_cls)
                and not np.isnan(ucb_xai)
                and ucb_cls <= alpha
                and ucb_xai <= beta
                and cov >= min_empirical_coverage
            )
        else:
            cov_lcb = cov
            ucb_cls = cal_cls if n_acc > 0 else float("nan")
            ucb_xai = cal_xai if n_acc > 0 else float("nan")
            feasible = (
                n_acc > 0
                and not np.isnan(cal_cls)
                and not np.isnan(cal_xai)
                and cal_cls <= alpha
                and cal_xai <= beta
                and cov >= min_empirical_coverage
            )

        rows.append(
            {
                "threshold": float(tau),
                "cal_coverage": cov,
                "cal_cls_risk": cal_cls,
                "cal_xai_risk": cal_xai,
                "cal_audited_risk": cal_aud,
                "ucb_cls_risk": ucb_cls,
                "ucb_xai_risk": ucb_xai,
                "coverage_lcb": cov_lcb if use_ucb else cov,
                "feasible": bool(feasible),
                "n_cal": n_cal,
                "n_accepted_cal": n_acc,
            }
        )

        if feasible and cov > best_cov:
            best_cov = cov
            best_tau = float(tau)

    if best_tau is None:
        return None, rows

    return best_tau, rows
