from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )


@dataclass
class TrainResult:
    best_epoch: int
    best_val_loss: float
    history: list[dict]


def train_classifier(
    model: torch.nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    epochs: int = 100,
    patience: int = 15,
    device: torch.device | None = None,
) -> TrainResult:
    if device is None:
        device = get_device()

    model = model.to(device)

    train_loader = make_loader(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_val, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    criterion = torch.nn.CrossEntropyLoss()

    best_state = None
    best_val_loss = float("inf")
    best_epoch = -1
    bad_epochs = 0

    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()

        train_losses = []
        train_preds = []
        train_targets = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item() * xb.size(0))
            train_preds.append(logits.argmax(dim=1).detach().cpu().numpy())
            train_targets.append(yb.detach().cpu().numpy())

        train_loss = float(np.sum(train_losses) / len(X_train))
        train_preds_np = np.concatenate(train_preds)
        train_targets_np = np.concatenate(train_targets)
        train_acc = accuracy_score(train_targets_np, train_preds_np)
        train_f1 = f1_score(train_targets_np, train_preds_np, average="macro")

        model.eval()

        val_losses = []
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                logits = model(xb)
                loss = criterion(logits, yb)

                val_losses.append(loss.item() * xb.size(0))
                val_preds.append(logits.argmax(dim=1).detach().cpu().numpy())
                val_targets.append(yb.detach().cpu().numpy())

        val_loss = float(np.sum(val_losses) / len(X_val))
        val_preds_np = np.concatenate(val_preds)
        val_targets_np = np.concatenate(val_targets)
        val_acc = accuracy_score(val_targets_np, val_preds_np)
        val_f1 = f1_score(val_targets_np, val_preds_np, average="macro")

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_macro_f1": train_f1,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_macro_f1": val_f1,
        }
        history.append(row)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return TrainResult(
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        history=history,
    )


def predict_logits(
    model: torch.nn.Module,
    X: np.ndarray,
    batch_size: int = 256,
    device: torch.device | None = None,
) -> np.ndarray:
    if device is None:
        device = get_device()

    model = model.to(device)
    model.eval()

    X_tensor = torch.tensor(X, dtype=torch.float32)
    loader = DataLoader(X_tensor, batch_size=batch_size, shuffle=False)

    logits_all = []

    with torch.no_grad():
        for xb in loader:
            xb = xb.to(device)
            logits = model(xb)
            logits_all.append(logits.detach().cpu().numpy())

    return np.concatenate(logits_all, axis=0)


def softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


def prediction_features_from_logits(logits: np.ndarray) -> dict[str, np.ndarray]:
    probs = softmax_np(logits)

    y_pred = probs.argmax(axis=1)
    max_prob = probs.max(axis=1)

    sorted_probs = np.sort(probs, axis=1)
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]

    entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1)

    # Energy definition: E(x) = -logsumexp(logits)
    # Later, accept score for Energy baseline will be -energy.
    max_logits = logits.max(axis=1, keepdims=True)
    logsumexp = max_logits.squeeze(1) + np.log(
        np.exp(logits - max_logits).sum(axis=1) + 1e-12
    )
    energy = -logsumexp

    return {
        "probs": probs,
        "y_pred": y_pred,
        "max_prob": max_prob,
        "entropy": entropy,
        "margin": margin,
        "energy": energy,
    }
