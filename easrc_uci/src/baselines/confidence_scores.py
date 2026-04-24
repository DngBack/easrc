from __future__ import annotations

import numpy as np
import pandas as pd


def make_score_only_baseline_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create non-trained method scores.

    Convention:
        Higher score = more likely to accept.
    """
    required = ["sample_id", "split", "max_prob", "entropy", "margin", "energy"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for score-only baselines: {missing}")

    rows = []

    base_cols = df[["sample_id", "split"]].copy()

    method_to_score = {
        "NoReject": np.ones(len(df), dtype=float),
        "MaxProb": df["max_prob"].to_numpy(dtype=float),
        "Entropy": -df["entropy"].to_numpy(dtype=float),
        "Margin": df["margin"].to_numpy(dtype=float),
        # energy was defined as E(x) = -logsumexp(logits).
        # More confident samples usually have lower/more negative energy,
        # so accept score is -energy.
        "Energy": -df["energy"].to_numpy(dtype=float),
    }

    for method, score in method_to_score.items():
        out = base_cols.copy()
        out["method"] = method
        out["score"] = score
        rows.append(out)

    return pd.concat(rows, axis=0, ignore_index=True)
