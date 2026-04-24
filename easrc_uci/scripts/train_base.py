from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.mlp import MLPClassifier
from src.models.train_utils import (
    get_device,
    predict_logits,
    prediction_features_from_logits,
    set_seed,
    train_classifier,
)


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_processed(processed_root: Path, seed: int):
    seed_dir = processed_root / f"seed_{seed}"

    X = np.load(seed_dir / "X.npy")
    y = np.load(seed_dir / "y.npy")
    sample_ids = np.load(seed_dir / "sample_ids.npy", allow_pickle=True)

    with open(seed_dir / "splits.json", "r", encoding="utf-8") as f:
        splits = json.load(f)

    with open(seed_dir / "class_names.json", "r", encoding="utf-8") as f:
        class_names = json.load(f)

    with open(seed_dir / "feature_names.json", "r", encoding="utf-8") as f:
        feature_names = json.load(f)

    return X, y, sample_ids, splits, class_names, feature_names


def get_base_model_config(config: dict) -> dict:
    if "base_model" in config:
        return config["base_model"]
    if "model" in config:
        return config["model"]
    raise KeyError("Config must contain either `base_model` or `model` section.")


def make_predictions_dataframe(
    sample_ids: np.ndarray,
    split_name: str,
    indices: list[int] | np.ndarray,
    y_true: np.ndarray,
    logits: np.ndarray,
) -> pd.DataFrame:
    feats = prediction_features_from_logits(logits)

    probs = feats["probs"]
    y_pred = feats["y_pred"]

    df = pd.DataFrame(
        {
            "sample_id": sample_ids[indices],
            "split": split_name,
            "y_true": y_true,
            "y_pred": y_pred,
            "correct": (y_pred == y_true).astype(int),
            "max_prob": feats["max_prob"],
            "entropy": feats["entropy"],
            "margin": feats["margin"],
            "energy": feats["energy"],
        }
    )

    for c in range(logits.shape[1]):
        df[f"logit_{c}"] = logits[:, c]
        df[f"prob_{c}"] = probs[:, c]

    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--base_model", type=str, default="mlp")
    args = parser.parse_args()

    set_seed(args.seed)

    config = load_config(args.config)
    dataset_cfg = config["dataset"]
    model_cfg = get_base_model_config(config)

    if args.base_model != "mlp":
        raise ValueError(
            f"This script currently supports --base_model mlp only. "
            f"Got: {args.base_model}"
        )

    processed_root = PROJECT_ROOT / dataset_cfg["processed_dir"]
    results_root = PROJECT_ROOT / dataset_cfg["results_dir"]

    X, y, sample_ids, splits, class_names, feature_names = load_processed(
        processed_root=processed_root,
        seed=args.seed,
    )

    base_train_idx = np.array(splits["base_train"], dtype=int)
    X_base = X[base_train_idx]
    y_base = y[base_train_idx]

    # Internal validation split from base_train only.
    train_idx_local, val_idx_local = train_test_split(
        np.arange(len(base_train_idx)),
        test_size=0.20,
        random_state=args.seed,
        stratify=y_base,
    )

    X_train = X_base[train_idx_local]
    y_train = y_base[train_idx_local]
    X_val = X_base[val_idx_local]
    y_val = y_base[val_idx_local]

    input_dim = X.shape[1]
    num_classes = len(class_names)

    model = MLPClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=model_cfg.get("hidden_dims", [512, 128]),
        dropout=float(model_cfg.get("dropout", 0.2)),
    )

    device = get_device()
    result = train_classifier(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        lr=float(model_cfg.get("lr", 1e-3)),
        weight_decay=float(model_cfg.get("weight_decay", 1e-4)),
        batch_size=int(model_cfg.get("batch_size", 64)),
        epochs=int(model_cfg.get("epochs", 100)),
        patience=int(model_cfg.get("patience", 15)),
        device=device,
    )

    out_dir = ensure_dir(results_root / f"seed_{args.seed}" / "base" / args.base_model)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "num_classes": num_classes,
            "hidden_dims": model_cfg.get("hidden_dims", [512, 128]),
            "dropout": float(model_cfg.get("dropout", 0.2)),
            "class_names": class_names,
            "feature_names": feature_names,
            "seed": args.seed,
            "best_epoch": result.best_epoch,
            "best_val_loss": result.best_val_loss,
        },
        out_dir / "model.pt",
    )

    pd.DataFrame(result.history).to_csv(out_dir / "train_log.csv", index=False)

    pred_frames = []
    for split_name in ["base_train", "rejector_train", "calibration", "test"]:
        idx = np.array(splits[split_name], dtype=int)

        logits = predict_logits(
            model=model,
            X=X[idx],
            batch_size=256,
            device=device,
        )

        df_split = make_predictions_dataframe(
            sample_ids=sample_ids,
            split_name=split_name,
            indices=idx,
            y_true=y[idx],
            logits=logits,
        )
        pred_frames.append(df_split)

    pred_df = pd.concat(pred_frames, axis=0, ignore_index=True)
    pred_df.to_csv(out_dir / "predictions.csv", index=False)

    test_df = pred_df[pred_df["split"] == "test"]
    test_acc = float(test_df["correct"].mean())

    print(f"Base model saved to: {out_dir}")
    print(f"Device: {device}")
    print(f"Best epoch: {result.best_epoch}")
    print(f"Best val loss: {result.best_val_loss:.6f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print("Prediction file:")
    print(f"  {out_dir / 'predictions.csv'}")


if __name__ == "__main__":
    main()
