from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLPRejector(nn.Module):
    """
    Binary accept/reject scoring model.

    Forward returns logits. Sigmoid(logits) is accept probability/score.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | tuple[int, ...] = (64, 32),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def make_loader(
    X: np.ndarray,
    z: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    X_tensor = torch.tensor(X, dtype=torch.float32)
    z_tensor = torch.tensor(z, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, z_tensor)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )


@dataclass
class RejectorTrainResult:
    best_epoch: int
    best_val_loss: float
    history: list[dict]


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def train_rejector(
    model: MLPRejector,
    X_train: np.ndarray,
    z_train: np.ndarray,
    X_val: np.ndarray,
    z_val: np.ndarray,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    epochs: int = 100,
    patience: int = 15,
    device: torch.device | None = None,
) -> RejectorTrainResult:
    if device is None:
        device = get_device()

    model = model.to(device)

    train_loader = make_loader(X_train, z_train, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(X_val, z_val, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    criterion = nn.BCEWithLogitsLoss()

    best_state = None
    best_val_loss = float("inf")
    best_epoch = -1
    bad_epochs = 0

    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()

        train_losses = []
        train_logits = []
        train_targets = []

        for xb, zb in train_loader:
            xb = xb.to(device)
            zb = zb.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, zb)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item() * xb.size(0))
            train_logits.append(logits.detach().cpu().numpy())
            train_targets.append(zb.detach().cpu().numpy())

        train_loss = float(np.sum(train_losses) / len(X_train))
        train_logits_np = np.concatenate(train_logits)
        train_targets_np = np.concatenate(train_targets)
        train_scores_np = 1.0 / (1.0 + np.exp(-train_logits_np))
        train_pred_np = (train_scores_np >= 0.5).astype(int)

        train_acc = float(accuracy_score(train_targets_np, train_pred_np))
        train_auc = safe_auc(train_targets_np, train_scores_np)

        model.eval()

        val_losses = []
        val_logits = []
        val_targets = []

        with torch.no_grad():
            for xb, zb in val_loader:
                xb = xb.to(device)
                zb = zb.to(device)

                logits = model(xb)
                loss = criterion(logits, zb)

                val_losses.append(loss.item() * xb.size(0))
                val_logits.append(logits.detach().cpu().numpy())
                val_targets.append(zb.detach().cpu().numpy())

        val_loss = float(np.sum(val_losses) / len(X_val))
        val_logits_np = np.concatenate(val_logits)
        val_targets_np = np.concatenate(val_targets)
        val_scores_np = 1.0 / (1.0 + np.exp(-val_logits_np))
        val_pred_np = (val_scores_np >= 0.5).astype(int)

        val_acc = float(accuracy_score(val_targets_np, val_pred_np))
        val_auc = safe_auc(val_targets_np, val_scores_np)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_auc": train_auc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_auc": val_auc,
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

    return RejectorTrainResult(
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        history=history,
    )


def predict_rejector_scores(
    model: MLPRejector,
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

    scores = []

    with torch.no_grad():
        for xb in loader:
            xb = xb.to(device)
            logits = model(xb)
            prob = torch.sigmoid(logits)
            scores.append(prob.detach().cpu().numpy())

    return np.concatenate(scores, axis=0)
