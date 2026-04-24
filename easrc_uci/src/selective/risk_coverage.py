from __future__ import annotations

import numpy as np


def metrics_at_threshold(
    scores: np.ndarray,
    cls_loss: np.ndarray,
    xai_loss: np.ndarray,
    audited_loss: np.ndarray,
    tau: float,
) -> dict[str, float | int]:
    """
    Empirical coverage and risks on a fixed split for accept rule score >= tau.
    """
    scores = np.asarray(scores, dtype=np.float64)
    cls_loss = np.asarray(cls_loss, dtype=np.float64)
    xai_loss = np.asarray(xai_loss, dtype=np.float64)
    audited_loss = np.asarray(audited_loss, dtype=np.float64)

    accepted = scores >= tau
    n = int(len(scores))
    n_acc = int(accepted.sum())
    coverage = n_acc / n if n else 0.0

    mean_il = float((accepted.astype(np.float64) * cls_loss).mean()) if n else 0.0
    mean_ixai = float((accepted.astype(np.float64) * xai_loss).mean()) if n else 0.0

    if n_acc == 0:
        return {
            "coverage": float(coverage),
            "cal_cls_risk": float("nan"),
            "cal_xai_risk": float("nan"),
            "cal_audited_risk": float("nan"),
            "mean_il": mean_il,
            "mean_ixai": mean_ixai,
            "n_accepted": 0,
        }

    return {
        "coverage": float(coverage),
        "cal_cls_risk": float(cls_loss[accepted].mean()),
        "cal_xai_risk": float(xai_loss[accepted].mean()),
        "cal_audited_risk": float(audited_loss[accepted].mean()),
        "mean_il": mean_il,
        "mean_ixai": mean_ixai,
        "n_accepted": n_acc,
    }


def ucb_epsilon(n_cal: int, n_thresholds: int, delta: float) -> float:
    """eps = sqrt(log(4 * |T| / delta) / (2n))"""
    if n_cal <= 0:
        return float("inf")
    t = max(int(n_thresholds), 1)
    d = float(delta)
    if d <= 0:
        raise ValueError("delta must be positive for UCB calibration.")
    return float(np.sqrt(np.log(4.0 * t / d) / (2.0 * n_cal)))


def ucb_bounds(
    coverage: float,
    mean_il: float,
    mean_ixai: float,
    eps: float,
) -> tuple[float, float, float]:
    """
    Returns (coverage_lcb, ucb_cls_risk, ucb_xai_risk).
    ucb_cls = (mean_il + eps) / coverage_lcb, same for xai.
    """
    coverage_lcb = float(coverage) - eps
    if coverage_lcb <= 0:
        return coverage_lcb, float("nan"), float("nan")
    ucb_cls = (mean_il + eps) / coverage_lcb
    ucb_xai = (mean_ixai + eps) / coverage_lcb
    return coverage_lcb, float(ucb_cls), float(ucb_xai)


def test_selective_metrics(
    scores: np.ndarray,
    cls_loss: np.ndarray,
    xai_loss: np.ndarray,
    audited_loss: np.ndarray,
    tau: float,
) -> dict[str, float | int]:
    """Metrics on test at a fixed threshold (may be NaN => no selection)."""
    if tau is None or (isinstance(tau, float) and np.isnan(tau)):
        n = int(len(scores))
        return {
            "test_coverage": 0.0,
            "test_cls_risk": float("nan"),
            "test_xai_risk": float("nan"),
            "test_audited_risk": float("nan"),
            "n_accepted_test": 0,
            "n_test": n,
        }

    m = metrics_at_threshold(scores, cls_loss, xai_loss, audited_loss, float(tau))
    n_acc = int(m["n_accepted"])
    return {
        "test_coverage": float(m["coverage"]),
        "test_cls_risk": float(m["cal_cls_risk"]) if n_acc > 0 else float("nan"),
        "test_xai_risk": float(m["cal_xai_risk"]) if n_acc > 0 else float("nan"),
        "test_audited_risk": float(m["cal_audited_risk"]) if n_acc > 0 else float("nan"),
        "n_accepted_test": n_acc,
        "n_test": int(len(scores)),
    }


def area_under_risk_coverage(
    scores: np.ndarray,
    cls_loss: np.ndarray,
) -> float:
    """
    Area under risk–coverage curve on test: sort by score descending, accept prefixes,
    integrate selective cls risk vs coverage from 0 to 1.
    """
    scores = np.asarray(scores, dtype=np.float64)
    cls_loss = np.asarray(cls_loss, dtype=np.float64)
    n = len(scores)
    if n == 0:
        return float("nan")
    order = np.argsort(-scores)
    cl = cls_loss[order]
    cum_sum = np.cumsum(cl)
    k = np.arange(1, n + 1, dtype=np.float64)
    risks = cum_sum / k
    coverages = k / n
    coverages = np.concatenate([[0.0], coverages])
    risks = np.concatenate([[0.0], risks])
    return float(np.trapezoid(risks, coverages))
