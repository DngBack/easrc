#!/usr/bin/env python3
"""
Download TCGA primary-tumor STAR gene counts from NCI GDC (open API),
merge into expression.csv + labels.csv for easrc_uci (load_tcga.py).

Requires network access. No GDC token needed for open-access data.

Example:
  python3 easrc_uci/scripts/download_tcga_gdc.py \\
    --out-dir easrc_uci/data/tcga_rnaseq/raw \\
    --cohorts TCGA-BRCA TCGA-LUAD TCGA-KIRC \\
    --max-per-cohort 40
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

GDC_API = "https://api.gdc.cancer.gov"
GDC_DATA = f"{GDC_API}/data"


def _post_json(url: str, payload: dict, timeout: int = 120) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "User-Agent": "easrc-easrc_uci/1.0"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_bytes(url: str, timeout: int = 300) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "easrc-easrc_uci/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def project_to_label(project_id: str) -> str:
    if project_id.startswith("TCGA-"):
        return project_id[len("TCGA-") :]
    return project_id


def build_file_filters(project_id: str) -> dict:
    return {
        "op": "and",
        "content": [
            {
                "op": "in",
                "content": {"field": "cases.project.project_id", "value": [project_id]},
            },
            {
                "op": "=",
                "content": {"field": "data_category", "value": "Transcriptome Profiling"},
            },
            {
                "op": "=",
                "content": {"field": "data_type", "value": "Gene Expression Quantification"},
            },
            {
                "op": "=",
                "content": {"field": "analysis.workflow_type", "value": "STAR - Counts"},
            },
            {
                "op": "=",
                "content": {"field": "cases.samples.sample_type", "value": "Primary Tumor"},
            },
        ],
    }


def iter_file_hits_for_project(
    project_id: str,
    fields: str,
    page_size: int = 100,
):
    """Yield /files hits for one TCGA project (paginated)."""
    from_ = 0
    while True:
        payload = {
            "filters": build_file_filters(project_id),
            "fields": fields,
            "format": "JSON",
            "size": page_size,
            "from": from_,
        }
        resp = _post_json(f"{GDC_API}/files", payload)
        batch = resp["data"]["hits"]
        if not batch:
            break
        for h in batch:
            yield h
        total = resp["data"]["pagination"]["total"]
        from_ += len(batch)
        if from_ >= total:
            break


def parse_star_counts_tsv(
    raw: bytes,
    metric: str,
    coding_only: bool,
) -> pd.Series:
    """Return Series index=gene_name, value=expression."""
    text = raw.decode("utf-8", errors="replace")
    df = pd.read_csv(io.StringIO(text), sep="\t", comment="#")
    if metric not in df.columns:
        raise ValueError(f"Column {metric!r} not in file. Available: {list(df.columns)}")
    if "gene_name" not in df.columns:
        raise ValueError("Expected gene_name column in STAR counts TSV.")

    g = df[["gene_name", "gene_type", metric]].copy()
    g = g[g["gene_name"].notna() & (g["gene_name"].astype(str).str.len() > 0)]
    if coding_only:
        g = g[g["gene_type"].astype(str) == "protein_coding"]
    g[metric] = pd.to_numeric(g[metric], errors="coerce")
    g = g.dropna(subset=[metric])
    g = g.groupby("gene_name", as_index=True)[metric].mean()
    return g.astype(np.float32)


def download_and_merge(
    cohorts: list[str],
    out_dir: Path,
    max_per_cohort: int,
    metric: str,
    coding_only: bool,
    apply_log1p: bool,
    sleep_s: float,
) -> None:
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    fields = ",".join(
        [
            "file_id",
            "file_name",
            "cases.case_id",
            "cases.submitter_id",
            "cases.project.project_id",
        ]
    )

    per_sample: list[tuple[str, str, pd.Series]] = []
    for project in cohorts:
        print(f"Streaming GDC files for {project} (label={project_to_label(project)})...")
        seen_cases: set[str] = set()
        n_ok = 0
        for h in iter_file_hits_for_project(project, fields=fields):
            if max_per_cohort > 0 and n_ok >= max_per_cohort:
                break
            case_id = None
            submitter = None
            proj = project
            cases = h.get("cases") or []
            if cases:
                case_id = cases[0].get("case_id")
                submitter = cases[0].get("submitter_id")
                proj = (cases[0].get("project") or {}).get("project_id", project)
            if not case_id or case_id in seen_cases:
                continue
            fid = h["file_id"]
            url = f"{GDC_DATA}/{fid}"
            try:
                raw = _get_bytes(url)
            except urllib.error.HTTPError as e:
                print(f"  skip file {fid}: HTTP {e.code}", file=sys.stderr)
                time.sleep(sleep_s)
                continue
            try:
                ser = parse_star_counts_tsv(raw, metric=metric, coding_only=coding_only)
            except Exception as e:
                print(f"  skip file {fid}: parse error {e}", file=sys.stderr)
                time.sleep(sleep_s)
                continue
            sid = str(submitter) if submitter else str(case_id)
            if apply_log1p:
                ser = np.log1p(ser.clip(lower=0.0))
            per_sample.append((sid, project_to_label(proj), ser))
            seen_cases.add(case_id)
            n_ok += 1
            print(f"  [{n_ok}/{max_per_cohort if max_per_cohort > 0 else '∞'}] {sid} ({len(ser)} genes)")
            time.sleep(sleep_s)

    if not per_sample:
        raise RuntimeError("No samples downloaded. Check cohort IDs and network.")

    # Intersection of genes across samples (fair merge).
    common: set[str] | None = None
    for _, _, ser in per_sample:
        common = set(ser.index) if common is None else common & set(ser.index)
    assert common is not None
    genes_sorted = sorted(common)
    print(f"Common protein-coding genes: {len(genes_sorted)}")

    rows = []
    labels_rows = []
    for sid, lab, ser in per_sample:
        vec = ser.reindex(genes_sorted)
        if vec.isna().any():
            vec = vec.fillna(0.0)
        row = {"sample_id": sid, **{g: float(vec.loc[g]) for g in genes_sorted}}
        rows.append(row)
        labels_rows.append({"sample_id": sid, "label": lab})

    expr_df = pd.DataFrame(rows)
    labels_df = pd.DataFrame(labels_rows)
    expr_path = out_dir / "expression.csv"
    lab_path = out_dir / "labels.csv"
    expr_df.to_csv(expr_path, index=False)
    labels_df.to_csv(lab_path, index=False)
    print(f"Wrote {expr_path} shape={expr_df.shape}")
    print(f"Wrote {lab_path} (n={len(labels_df)})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download TCGA RNA-seq from GDC into expression.csv + labels.csv")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/tcga_rnaseq/raw",
        help="Relative to easrc_uci root unless absolute.",
    )
    parser.add_argument(
        "--cohorts",
        nargs="+",
        default=["TCGA-BRCA", "TCGA-LUAD", "TCGA-KIRC", "TCGA-COAD", "TCGA-PRAD"],
        help="TCGA project_ids to include.",
    )
    parser.add_argument(
        "--max-per-cohort",
        type=int,
        default=40,
        help="Max primary-tumor samples per cohort (0 = all; can be very slow/large).",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="tpm_unstranded",
        help="Column in STAR augmented gene counts TSV (e.g. tpm_unstranded, fpkm_unstranded).",
    )
    parser.add_argument(
        "--all-gene-types",
        action="store_true",
        help="Include all gene_type rows (default: protein_coding only).",
    )
    parser.add_argument(
        "--log1p",
        action="store_true",
        default=True,
        help="Apply log1p to expression values (default: on).",
    )
    parser.add_argument(
        "--no-log1p",
        action="store_true",
        help="Store raw TPM (or chosen metric) without log1p.",
    )
    parser.add_argument("--sleep", type=float, default=0.35, help="Pause between GDC downloads (seconds).")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        parts = out_dir.parts
        # Avoid easrc_uci/easrc_uci/data/... when user passes --out-dir easrc_uci/data/... from repo root.
        if parts and parts[0] == root.name:
            out_dir = Path(*parts[1:]) if len(parts) > 1 else Path(".")
        out_dir = (root / out_dir).resolve()

    coding_only = not args.all_gene_types
    apply_log1p = args.log1p and not args.no_log1p

    download_and_merge(
        cohorts=list(args.cohorts),
        out_dir=out_dir,
        max_per_cohort=int(args.max_per_cohort),
        metric=args.metric,
        coding_only=coding_only,
        apply_log1p=apply_log1p,
        sleep_s=float(args.sleep),
    )


if __name__ == "__main__":
    main()
