from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler


def fit_and_transform_splits(
    X: np.ndarray,
    split_indices: dict[str, np.ndarray],
    standardize: bool = True,
) -> tuple[np.ndarray, StandardScaler | None]:
    if not standardize:
        return X.astype(np.float32), None

    base_idx = split_indices["base_train"]
    scaler = StandardScaler()
    scaler.fit(X[base_idx])
    X_scaled = scaler.transform(X).astype(np.float32)
    return X_scaled, scaler
