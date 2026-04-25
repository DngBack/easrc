from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.confidence_scores import make_score_only_baseline_scores
from src.rejectors.feature_sets import (
    FEATURE_SETS,
    LEARNED_REJECTOR_METHODS,
)
from src.rejectors.mlp_rejector import (
    MLPRejector,
    get_device,
    predict_rejector_scores,
    set_seed,
    train_rejector,
)


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def compute_training_targets(
    df: pd.DataFrame,
    gamma: float,
    cls_weight: float = 0.5,
    xai_weight: float = 0.5,
) -> pd.DataFrame:
    """
    Create rejector training target.

    cls_loss = 1[y_pred != y_true]
    audited_loss = cls_weight * cls_loss + xai_weight * xai_unreliability
    z = 1 if audited_loss <= gamma else 0

    z = 1 means acceptable sample.
    """
    required = ["y_true", "y_pred", "xai_unreliability"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for target construction: {missing}")

    out = df.copy()

    cls_loss = (out["y_pred"].to_numpy() != out["y_true"].to_numpy()).astype(float)
    xai_loss = out["xai_unreliability"].to_numpy(dtype=float)

    audited_loss = cls_weight * cls_loss + xai_weight * xai_loss
    z = (audited_loss <= gamma).astype(int)

    out["cls_loss"] = cls_loss
    out["xai_loss"] = xai_loss
    out["audited_loss"] = audited_loss
    out["accept_target"] = z

    return out


def make_train_val_split(
    X: np.ndarray,
    z: np.ndarray,
    seed: int,
    val_size: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split rejector_train into train/val.

    Uses stratification only when both classes have at least 2 samples.
    """
    unique, counts = np.unique(z, return_counts=True)

    can_stratify = len(unique) == 2 and np.min(counts) >= 2

    stratify = z if can_stratify else None

    if len(z) < 5:
        return X, X, z, z

    X_train, X_val, z_train, z_val = train_test_split(
        X,
        z,
        test_size=val_size,
        random_state=seed,
        stratify=stratify,
    )

    return X_train, X_val, z_train, z_val


def validate_feature_columns(df: pd.DataFrame, feature_cols: list[str], method: str) -> None:
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns for {method}: {missing}")

    if df[feature_cols].isna().any().any():
        bad_cols = df[feature_cols].columns[df[feature_cols].isna().any()].tolist()
        raise ValueError(f"NaN detected in {method} feature columns: {bad_cols}")


def train_one_rejector(
    method: str,
    df: pd.DataFrame,
    rejector_cfg: dict,
    seed: int,
    out_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols = FEATURE_SETS[method]
    validate_feature_columns(df, feature_cols, method)

    train_mask = df["split"] == "rejector_train"

    train_df = df.loc[train_mask].copy()
    all_df = df.copy()

    X_train_raw = train_df[feature_cols].to_numpy(dtype=np.float32)
    z_train_full = train_df["accept_target"].to_numpy(dtype=int)

    scaler = StandardScaler()
    X_train_scaled_full = scaler.fit_transform(X_train_raw).astype(np.float32)

    X_fit, X_val, z_fit, z_val = make_train_val_split(
        X=X_train_scaled_full,
        z=z_train_full,
        seed=seed,
        val_size=0.2,
    )

    method_dir = ensure_dir(out_dir / "rejectors" / method)

    joblib.dump(
        {
            "scaler": scaler,
            "feature_cols": feature_cols,
        },
        method_dir / "scaler.joblib",
    )

    model = MLPRejector(
        input_dim=len(feature_cols),
        hidden_dims=rejector_cfg.get("hidden_dims", [64, 32]),
        dropout=float(rejector_cfg.get("dropout", 0.1)),
    )

    device = get_device()

    result = train_rejector(
        model=model,
        X_train=X_fit,
        z_train=z_fit,
        X_val=X_val,
        z_val=z_val,
        lr=float(rejector_cfg.get("lr", 1e-3)),
        weight_decay=float(rejector_cfg.get("weight_decay", 1e-4)),
        batch_size=int(rejector_cfg.get("batch_size", 64)),
        epochs=int(rejector_cfg.get("epochs", 100)),
        patience=int(rejector_cfg.get("patience", 15)),
        device=device,
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": len(feature_cols),
            "hidden_dims": rejector_cfg.get("hidden_dims", [64, 32]),
            "dropout": float(rejector_cfg.get("dropout", 0.1)),
            "feature_cols": feature_cols,
            "seed": seed,
            "method": method,
            "best_epoch": result.best_epoch,
            "best_val_loss": result.best_val_loss,
        },
        method_dir / "model.pt",
    )

    all_X_raw = all_df[feature_cols].to_numpy(dtype=np.float32)
    all_X_scaled = scaler.transform(all_X_raw).astype(np.float32)

    scores = predict_rejector_scores(
        model=model,
        X=all_X_scaled,
        batch_size=256,
        device=device,
    )

    score_df = all_df[["sample_id", "split"]].copy()
    score_df["method"] = method
    score_df["score"] = scores

    log_df = pd.DataFrame(result.history)
    log_df.insert(0, "method", method)
    log_df["best_epoch"] = result.best_epoch
    log_df["best_val_loss"] = result.best_val_loss

    target_rate = float(z_train_full.mean())
    n_pos = int(z_train_full.sum())
    n_total = int(len(z_train_full))

    print(
        f"{method:14s} | target accept rate on rejector_train: "
        f"{target_rate:.3f} ({n_pos}/{n_total}) | "
        f"best epoch: {result.best_epoch}"
    )

    return score_df, log_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--base_model", type=str, default="mlp")
    parser.add_argument(
        "--target-gamma",
        type=float,
        default=None,
        help="Override rejector.target_gamma from config.",
    )
    parser.add_argument(
        "--cls-weight",
        type=float,
        default=None,
        help="Override rejector.cls_weight (default 0.5).",
    )
    parser.add_argument(
        "--xai-weight",
        type=float,
        default=None,
        help="Override rejector.xai_weight (default 0.5).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directory for scores CSVs/metadata (default: results/.../scores/<base_model>).",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    config = load_config(args.config)

    dataset_cfg = config["dataset"]
    rejector_cfg = config.get("rejector", {})

    results_root = PROJECT_ROOT / dataset_cfg["results_dir"]

    feature_dir = results_root / f"seed_{args.seed}" / "features" / args.base_model
    all_features_path = feature_dir / "all_features.csv"

    if not all_features_path.exists():
        raise FileNotFoundError(
            f"Missing all_features.csv: {all_features_path}. "
            f"Run compute_features.py first."
        )

    if args.out_dir:
        out_dir = ensure_dir(Path(args.out_dir))
    else:
        out_dir = ensure_dir(results_root / f"seed_{args.seed}" / "scores" / args.base_model)

    df = pd.read_csv(all_features_path)

    gamma = float(
        args.target_gamma
        if args.target_gamma is not None
        else rejector_cfg.get("target_gamma", 0.30)
    )

    cls_weight = float(
        args.cls_weight if args.cls_weight is not None else rejector_cfg.get("cls_weight", 0.5)
    )
    xai_weight = float(
        args.xai_weight if args.xai_weight is not None else rejector_cfg.get("xai_weight", 0.5)
    )

    df = compute_training_targets(
        df=df,
        gamma=gamma,
        cls_weight=cls_weight,
        xai_weight=xai_weight,
    )

    df.to_csv(out_dir / "all_features_with_targets.csv", index=False)

    print("Rejector target summary on rejector_train:")
    tmp = df[df["split"] == "rejector_train"]
    print(
        tmp[
            [
                "cls_loss",
                "xai_loss",
                "audited_loss",
                "accept_target",
            ]
        ].describe()
    )
    print()

    score_frames = []
    log_frames = []

    score_only_scores = make_score_only_baseline_scores(df)
    score_frames.append(score_only_scores)

    for method in LEARNED_REJECTOR_METHODS:
        score_df, log_df = train_one_rejector(
            method=method,
            df=df,
            rejector_cfg=rejector_cfg,
            seed=args.seed,
            out_dir=out_dir,
        )

        score_frames.append(score_df)
        log_frames.append(log_df)

    method_scores = pd.concat(score_frames, axis=0, ignore_index=True)
    method_scores.to_csv(out_dir / "method_scores.csv", index=False)

    if log_frames:
        rejector_logs = pd.concat(log_frames, axis=0, ignore_index=True)
    else:
        rejector_logs = pd.DataFrame()

    rejector_logs.to_csv(out_dir / "rejector_train_logs.csv", index=False)

    metadata = {
        "seed": args.seed,
        "base_model": args.base_model,
        "gamma": gamma,
        "cls_weight": cls_weight,
        "xai_weight": xai_weight,
        "methods": sorted(method_scores["method"].unique().tolist()),
        "score_direction": "higher_score_means_more_likely_to_accept",
    }

    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print()
    print(f"Scores saved to: {out_dir}")
    print("Methods:")
    for method in sorted(method_scores["method"].unique()):
        n_rows = len(method_scores[method_scores["method"] == method])
        print(f"  - {method:14s}: {n_rows} rows")

    print()
    print("Score summary:")
    print(method_scores.groupby("method")["score"].describe())


if __name__ == "__main__":
    main()
