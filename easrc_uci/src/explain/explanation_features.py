from __future__ import annotations

import numpy as np


def attribution_entropy(
    attributions: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Normalized entropy of absolute attribution mass.

    Returns values in [0, 1].
    """
    abs_attr = np.abs(attributions)
    d = abs_attr.shape[1]

    mass = abs_attr.sum(axis=1, keepdims=True) + eps
    p = abs_attr / mass

    entropy = -np.sum(p * np.log(p + eps), axis=1) / np.log(d)
    entropy = np.clip(entropy, 0.0, 1.0)

    return entropy


def topk_attribution_mass(
    attributions: np.ndarray,
    k: int = 100,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Fraction of attribution mass contained in top-k absolute features.

    Returns values in [0, 1].
    """
    abs_attr = np.abs(attributions)
    _, d = abs_attr.shape

    k = min(k, d)

    topk = np.partition(abs_attr, kth=d - k, axis=1)[:, d - k:]
    topk_mass = topk.sum(axis=1) / (abs_attr.sum(axis=1) + eps)
    topk_mass = np.clip(topk_mass, 0.0, 1.0)

    return topk_mass


def xai_unreliability_score(
    attr_entropy: np.ndarray,
    attr_stability: np.ndarray,
    topk_mass: np.ndarray,
    weights: dict | None = None,
) -> np.ndarray:
    """
    Composite explanation unreliability score in [0, 1].

    Higher = less reliable explanation.
    """
    if weights is None:
        weights = {
            "attr_entropy": 0.4,
            "attr_instability": 0.4,
            "inverse_topk_mass": 0.2,
        }

    w_entropy = float(weights.get("attr_entropy", 0.4))
    w_instability = float(weights.get("attr_instability", 0.4))
    w_inverse_topk = float(weights.get("inverse_topk_mass", 0.2))

    total = w_entropy + w_instability + w_inverse_topk
    if total <= 0:
        raise ValueError("XAI unreliability weights must sum to a positive value.")

    w_entropy /= total
    w_instability /= total
    w_inverse_topk /= total

    unreliability = (
        w_entropy * attr_entropy
        + w_instability * (1.0 - attr_stability)
        + w_inverse_topk * (1.0 - topk_mass)
    )

    return np.clip(unreliability, 0.0, 1.0)
