from __future__ import annotations

import numpy as np


def build_proxy_groups(
    X: np.ndarray,
    y: np.ndarray,
    rejector_train_indices: np.ndarray,
    num_classes: int,
    topk_per_class: int = 200,
) -> dict[int, np.ndarray]:
    """
    Build class-wise proxy biological groups using only rejector_train.

    For UCI RNA-seq, feature identities may be anonymized. This module creates
    pseudo pathway groups by selecting top discriminative features per class.

    Important:
        This must not use calibration or test data.
    """
    X_rej = X[rejector_train_indices]
    y_rej = y[rejector_train_indices]

    groups: dict[int, np.ndarray] = {}
    d = X.shape[1]
    topk = min(topk_per_class, d)

    for c in range(num_classes):
        class_mask = y_rej == c

        if class_mask.sum() == 0:
            raise ValueError(f"No rejector-train samples found for class {c}.")

        X_pos = X_rej[class_mask]
        X_neg = X_rej[~class_mask]

        pos_mean = X_pos.mean(axis=0)
        neg_mean = X_neg.mean(axis=0)

        score = np.abs(pos_mean - neg_mean)

        group = np.argsort(score)[-topk:]
        groups[c] = np.sort(group)

    return groups


def make_random_groups(
    num_classes: int,
    num_features: int,
    group_size: int,
    n_random_groups: int = 10,
    seed: int = 0,
) -> dict[int, np.ndarray]:
    """
    Random size-matched groups for negative control.

    Returns:
        dict[class_id] = array [n_random_groups, group_size]
    """
    rng = np.random.default_rng(seed)

    random_groups: dict[int, np.ndarray] = {}

    for c in range(num_classes):
        groups_c = []
        for _ in range(n_random_groups):
            group = rng.choice(
                num_features,
                size=min(group_size, num_features),
                replace=False,
            )
            groups_c.append(np.sort(group))

        random_groups[c] = np.stack(groups_c, axis=0)

    return random_groups


def attribution_mass_in_predicted_group(
    attributions: np.ndarray,
    predicted_classes: np.ndarray,
    groups: dict[int, np.ndarray],
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Compute attribution mass inside predicted-class proxy group.

    This is deployable because it uses predicted class, not true label.
    """
    abs_attr = np.abs(attributions)
    total_mass = abs_attr.sum(axis=1) + eps

    alignments = np.zeros(attributions.shape[0], dtype=np.float32)

    for i, pred_class in enumerate(predicted_classes):
        group = groups[int(pred_class)]
        alignments[i] = abs_attr[i, group].sum() / total_mass[i]

    return np.clip(alignments, 0.0, 1.0)


def random_group_alignment(
    attributions: np.ndarray,
    predicted_classes: np.ndarray,
    random_groups: dict[int, np.ndarray],
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Average attribution mass inside random size-matched groups.
    """
    abs_attr = np.abs(attributions)
    total_mass = abs_attr.sum(axis=1) + eps

    alignments = np.zeros(attributions.shape[0], dtype=np.float32)

    for i, pred_class in enumerate(predicted_classes):
        groups_c = random_groups[int(pred_class)]

        masses = []
        for group in groups_c:
            masses.append(abs_attr[i, group].sum() / total_mass[i])

        alignments[i] = float(np.mean(masses))

    return np.clip(alignments, 0.0, 1.0)
