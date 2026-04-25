#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import yaml
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_tcga import load_tcga_rnaseq
from src.data.load_uci import load_uci_rnaseq
from src.data.preprocess import fit_and_transform_splits
from src.data.split import SplitConfig, make_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare RNA-seq matrices and save processed artifacts.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument("--seed", type=int, default=None, help="Override seed from config.")
    return parser.parse_args()


def _read_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_data_dir(script_path: Path, rel_path: str) -> Path:
    # Config paths are defined relative to easrc_uci package root.
    return (script_path.parents[1] / rel_path).resolve()


def _to_serializable_splits(split_indices: dict[str, np.ndarray]) -> dict[str, list[int]]:
    return {k: v.astype(int).tolist() for k, v in split_indices.items()}


def _print_split_report(y: np.ndarray, split_indices: dict[str, np.ndarray]) -> None:
    all_classes = sorted(np.unique(y).tolist())
    print("Split check:")
    for name, idx in split_indices.items():
        labels = np.unique(y[idx]).tolist()
        print(f"  - {name:14s}: n={len(idx):5d}, classes={len(labels)} / {len(all_classes)}")


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    cfg = _read_config(config_path)
    seed = cfg["project"]["seed"] if args.seed is None else args.seed

    script_path = Path(__file__).resolve()
    raw_dir = _resolve_data_dir(script_path, cfg["dataset"]["raw_dir"])
    processed_root = _resolve_data_dir(script_path, cfg["dataset"]["processed_dir"])
    out_dir = processed_root / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = str(cfg["dataset"].get("loader", "uci")).lower()
    if loader == "uci":
        X, y_raw, sample_ids, feature_names = load_uci_rnaseq(
            raw_dir=raw_dir, dataset_name=cfg["dataset"]["name"]
        )
    elif loader == "tcga":
        X, y_raw, sample_ids, feature_names = load_tcga_rnaseq(raw_dir=raw_dir)
    else:
        raise ValueError(f"Unknown dataset.loader: {loader!r}. Use 'uci' or 'tcga'.")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw).astype(np.int64)
    class_names = label_encoder.classes_.tolist()

    split_cfg = SplitConfig(
        base_train=cfg["split"]["base_train"],
        rejector_train=cfg["split"]["rejector_train"],
        calibration=cfg["split"]["calibration"],
        test=cfg["split"]["test"],
        stratify=bool(cfg["split"].get("stratify", True)),
    )
    split_indices = make_splits(y=y, cfg=split_cfg, seed=seed)

    X_processed, scaler = fit_and_transform_splits(
        X=X,
        split_indices=split_indices,
        standardize=bool(cfg["preprocess"].get("standardize", True)),
    )

    np.save(out_dir / "X.npy", X_processed)
    np.save(out_dir / "y.npy", y)
    np.save(out_dir / "sample_ids.npy", sample_ids)

    with (out_dir / "feature_names.json").open("w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)
    with (out_dir / "class_names.json").open("w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)
    with (out_dir / "splits.json").open("w", encoding="utf-8") as f:
        json.dump(_to_serializable_splits(split_indices), f, indent=2)

    if scaler is not None:
        joblib.dump(scaler, out_dir / "scaler.joblib")
    else:
        joblib.dump({"standardize": False}, out_dir / "scaler.joblib")

    print(f"Prepared data saved to: {out_dir}")
    _print_split_report(y, split_indices)
    print("Scaler check: fitted only on split `base_train`.")


if __name__ == "__main__":
    main()
