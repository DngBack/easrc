from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def _normalize_features_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # UCI raw exports often store sample id in first unnamed column.
    if "sample_id" not in df.columns:
        if "Unnamed: 0" in df.columns:
            df = df.rename(columns={"Unnamed: 0": "sample_id"})
        elif len(df.columns) > 0:
            first_col = df.columns[0]
            if str(first_col).lower() in {"id", "sample", "sampleid"}:
                df = df.rename(columns={first_col: "sample_id"})
    return df


def _normalize_labels_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "sample_id" not in df.columns:
        if "Unnamed: 0" in df.columns:
            df = df.rename(columns={"Unnamed: 0": "sample_id"})
        elif len(df.columns) > 0:
            first_col = df.columns[0]
            if str(first_col).lower() in {"id", "sample", "sampleid"}:
                df = df.rename(columns={first_col: "sample_id"})

    if "label" not in df.columns:
        if "Class" in df.columns:
            df = df.rename(columns={"Class": "label"})
        elif "class" in df.columns:
            df = df.rename(columns={"class": "label"})
    return df


def _detect_two_file_format(raw_dir: Path) -> Tuple[Path, Path] | None:
    features_path = raw_dir / "features.csv"
    labels_path = raw_dir / "labels.csv"
    if features_path.exists() and labels_path.exists():
        return features_path, labels_path
    return None


def _detect_single_file_format(raw_dir: Path, dataset_name: str) -> Path | None:
    preferred = raw_dir / f"{dataset_name}.csv"
    if preferred.exists():
        return preferred

    csv_files = sorted(raw_dir.glob("*.csv"))
    if len(csv_files) == 1:
        return csv_files[0]
    return None


def load_uci_rnaseq(raw_dir: str | Path, dataset_name: str = "uci_rnaseq") -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

    two_file = _detect_two_file_format(raw_dir)
    if two_file is not None:
        features_path, labels_path = two_file
        features_df = _normalize_features_columns(pd.read_csv(features_path))
        labels_df = _normalize_labels_columns(pd.read_csv(labels_path))

        if "sample_id" not in features_df.columns:
            raise ValueError("features.csv must contain column: sample_id")
        if "sample_id" not in labels_df.columns or "label" not in labels_df.columns:
            raise ValueError("labels.csv must contain columns: sample_id, label")

        merged = features_df.merge(labels_df, on="sample_id", how="inner", validate="one_to_one")
        if merged.empty:
            raise ValueError("No samples after merging features.csv and labels.csv on sample_id")
    else:
        single_file = _detect_single_file_format(raw_dir, dataset_name)
        if single_file is None:
            raise FileNotFoundError(
                "Could not detect raw data format. Expect either "
                "`features.csv` + `labels.csv`, or a single `*.csv` file with "
                "columns including sample_id and label."
            )
        merged = pd.read_csv(single_file)
        merged = _normalize_labels_columns(_normalize_features_columns(merged))
        if "sample_id" not in merged.columns or "label" not in merged.columns:
            raise ValueError(f"{single_file.name} must contain columns: sample_id, label")

    gene_columns = [c for c in merged.columns if c not in {"sample_id", "label"}]
    if not gene_columns:
        raise ValueError("No gene feature columns found in raw data.")

    merged = merged.dropna(subset=["sample_id", "label"])
    merged = merged.reset_index(drop=True)

    sample_ids = merged["sample_id"].astype(str).to_numpy()
    labels_raw = merged["label"].astype(str).to_numpy()
    X = merged[gene_columns].to_numpy(dtype=np.float32)

    return X, labels_raw, sample_ids, gene_columns
