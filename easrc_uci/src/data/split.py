from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split


@dataclass
class SplitConfig:
    base_train: float
    rejector_train: float
    calibration: float
    test: float
    stratify: bool = True

    def validate(self) -> None:
        total = self.base_train + self.rejector_train + self.calibration + self.test
        if not np.isclose(total, 1.0):
            raise ValueError(f"Split ratios must sum to 1.0, got {total:.6f}")
        for name, value in [
            ("base_train", self.base_train),
            ("rejector_train", self.rejector_train),
            ("calibration", self.calibration),
            ("test", self.test),
        ]:
            if value <= 0:
                raise ValueError(f"Split ratio `{name}` must be > 0, got {value}")


def _check_all_classes_present(y: np.ndarray, split_indices: dict[str, np.ndarray]) -> None:
    all_classes = set(np.unique(y).tolist())
    for split_name, idx in split_indices.items():
        split_classes = set(np.unique(y[idx]).tolist())
        missing = all_classes - split_classes
        if missing:
            raise ValueError(
                f"Split `{split_name}` is missing classes: {sorted(missing)}. "
                "Try changing seed or split ratios."
            )


def _train_test_split_stratified(
    indices: np.ndarray,
    stratify_y: np.ndarray | None,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    Stratified split when sklearn allows it; otherwise fall back to unstratified.

    Returns (train_idx, test_idx, used_stratify).
    """
    if stratify_y is None:
        a, b = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=None,
        )
        return a, b, False
    try:
        a, b = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_y,
        )
        return a, b, True
    except ValueError as e:
        msg = str(e).lower()
        if "stratify" in msg or "test_size" in msg or "least populated class" in msg:
            warnings.warn(
                f"Stratified split failed ({e!r}); using unstratified split for this stage. "
                "Increase sample size per class for stable stratification.",
                UserWarning,
                stacklevel=3,
            )
            a, b = train_test_split(
                indices,
                test_size=test_size,
                random_state=random_state,
                stratify=None,
            )
            return a, b, False
        raise


def make_splits(y: np.ndarray, cfg: SplitConfig, seed: int) -> dict[str, np.ndarray]:
    cfg.validate()
    n = y.shape[0]
    all_idx = np.arange(n)
    strat_y = y if cfg.stratify else None

    # Step 1: carve out test set.
    trainval_idx, test_idx, s1 = _train_test_split_stratified(
        all_idx, strat_y, test_size=cfg.test, random_state=seed
    )

    # Step 2: split remaining pool into base/rejector/calibration with normalized ratios.
    remaining = 1.0 - cfg.test
    base_rel = cfg.base_train / remaining
    rej_rel = cfg.rejector_train / remaining
    cal_rel = cfg.calibration / remaining

    trainval_y = y[trainval_idx]
    trainval_strat = trainval_y if cfg.stratify else None

    base_idx, rest_idx, s2 = _train_test_split_stratified(
        trainval_idx,
        trainval_strat,
        test_size=(rej_rel + cal_rel),
        random_state=seed + 1,
    )

    rest_y = y[rest_idx]
    rest_strat = rest_y if cfg.stratify else None
    rej_share_of_rest = rej_rel / (rej_rel + cal_rel)
    rejector_idx, calibration_idx, s3 = _train_test_split_stratified(
        rest_idx,
        rest_strat,
        test_size=(1.0 - rej_share_of_rest),
        random_state=seed + 2,
    )

    split_indices = {
        "base_train": np.sort(base_idx),
        "rejector_train": np.sort(rejector_idx),
        "calibration": np.sort(calibration_idx),
        "test": np.sort(test_idx),
    }
    strict_stratify = cfg.stratify and s1 and s2 and s3
    if strict_stratify:
        _check_all_classes_present(y, split_indices)
    else:
        warnings.warn(
            "Skipping per-split 'all classes present' check because stratify was disabled "
            "for at least one split (dataset too small for sklearn stratification).",
            UserWarning,
            stacklevel=2,
        )
    return split_indices
