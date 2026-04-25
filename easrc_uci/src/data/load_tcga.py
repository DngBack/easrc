from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.load_uci import _normalize_labels_columns


def _normalize_expression_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "sample_id" not in df.columns:
        if "Unnamed: 0" in df.columns:
            df = df.rename(columns={"Unnamed: 0": "sample_id"})
        elif len(df.columns) > 0:
            first = df.columns[0]
            if str(first).lower() in {"id", "sample", "sampleid"}:
                df = df.rename(columns={first: "sample_id"})
    return df


def load_tcga_rnaseq(raw_dir: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Load TCGA (or TCGA-style) RNA-seq matrices from raw_dir.

    Expected layout (minimum):
        raw/expression.csv   — sample_id, <gene1>, <gene2>, ...
        raw/labels.csv       — sample_id, label

    Gene columns should be HGNC symbols (or any string ids consistent with pathway GMTs).
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

    expr_path = raw_dir / "expression.csv"
    labels_path = raw_dir / "labels.csv"

    if not expr_path.exists():
        raise FileNotFoundError(
            f"Missing {expr_path}. Place TCGA expression matrix as expression.csv "
            "(columns: sample_id, gene symbols, ...)."
        )
    if not labels_path.exists():
        raise FileNotFoundError(
            f"Missing {labels_path}. Place labels as labels.csv (columns: sample_id, label)."
        )

    expr_df = _normalize_expression_columns(pd.read_csv(expr_path))
    labels_df = _normalize_labels_columns(pd.read_csv(labels_path))

    if "sample_id" not in expr_df.columns:
        raise ValueError("expression.csv must contain column: sample_id")
    if "sample_id" not in labels_df.columns or "label" not in labels_df.columns:
        raise ValueError("labels.csv must contain columns: sample_id, label")

    merged = expr_df.merge(labels_df, on="sample_id", how="inner", validate="one_to_one")
    if merged.empty:
        raise ValueError("No samples after merging expression.csv and labels.csv on sample_id")

    gene_columns = [c for c in merged.columns if c not in {"sample_id", "label"}]
    if not gene_columns:
        raise ValueError("No gene columns found in expression.csv after sample_id.")

    merged = merged.dropna(subset=["sample_id", "label"])
    merged = merged.reset_index(drop=True)

    sample_ids = merged["sample_id"].astype(str).to_numpy()
    labels_raw = merged["label"].astype(str).to_numpy()
    X = merged[gene_columns].to_numpy(dtype=np.float32)

    return X, labels_raw, sample_ids, gene_columns
