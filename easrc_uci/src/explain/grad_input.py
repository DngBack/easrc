from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def gradient_times_input(
    model: torch.nn.Module,
    X: np.ndarray,
    target_classes: np.ndarray,
    batch_size: int = 64,
    device: torch.device | None = None,
) -> np.ndarray:
    """
    Compute Gradient x Input attribution for target_classes.

    Args:
        model: trained classifier.
        X: standardized input array [n, d].
        target_classes: class index for each sample [n].
        batch_size: batch size.
        device: torch device.

    Returns:
        attributions: array [n, d].
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(target_classes, dtype=torch.long)

    loader = DataLoader(
        TensorDataset(X_tensor, y_tensor),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    attrs = []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        xb.requires_grad_(True)

        logits = model(xb)
        selected_logits = logits.gather(1, yb.view(-1, 1)).sum()

        grads = torch.autograd.grad(
            outputs=selected_logits,
            inputs=xb,
            create_graph=False,
            retain_graph=False,
            only_inputs=True,
        )[0]

        attr = grads * xb
        attrs.append(attr.detach().cpu().numpy())

    return np.concatenate(attrs, axis=0)


def gradient_times_input_stability(
    model: torch.nn.Module,
    X: np.ndarray,
    target_classes: np.ndarray,
    perturb_std: float = 0.01,
    repeats: int = 5,
    batch_size: int = 64,
    device: torch.device | None = None,
    seed: int = 0,
) -> np.ndarray:
    """
    Estimate attribution stability by perturbing input and computing
    correlation between original and perturbed absolute attributions.

    Returns:
        stability: array [n], clipped to [0, 1].
    """
    rng = np.random.default_rng(seed)

    base_attr = gradient_times_input(
        model=model,
        X=X,
        target_classes=target_classes,
        batch_size=batch_size,
        device=device,
    )
    base_abs = np.abs(base_attr)

    stability_all = []

    for _ in range(repeats):
        noise = rng.normal(loc=0.0, scale=perturb_std, size=X.shape).astype(np.float32)
        X_pert = X.astype(np.float32) + noise

        pert_attr = gradient_times_input(
            model=model,
            X=X_pert,
            target_classes=target_classes,
            batch_size=batch_size,
            device=device,
        )
        pert_abs = np.abs(pert_attr)

        corr = rowwise_correlation(base_abs, pert_abs)
        corr = np.clip(corr, 0.0, 1.0)
        stability_all.append(corr)

    return np.mean(np.stack(stability_all, axis=0), axis=0)


def rowwise_correlation(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Pearson correlation per row.
    """
    a_centered = a - a.mean(axis=1, keepdims=True)
    b_centered = b - b.mean(axis=1, keepdims=True)

    numerator = np.sum(a_centered * b_centered, axis=1)
    denom = np.sqrt(np.sum(a_centered**2, axis=1) * np.sum(b_centered**2, axis=1))

    corr = numerator / (denom + eps)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    return corr
