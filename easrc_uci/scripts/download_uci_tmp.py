#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import shutil
import tarfile
import urllib.request
import zipfile

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/401/gene+expression+cancer+rna+seq.zip"


def main() -> None:
    out_dir = Path(__file__).resolve().parents[1] / "data" / "uci_rnaseq" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / "uci_rnaseq.zip"
    extracted_dir = out_dir / "_tmp_extracted"

    print(f"Downloading dataset from: {UCI_ZIP_URL}")
    urllib.request.urlretrieve(UCI_ZIP_URL, zip_path)
    print(f"Saved archive to: {zip_path}")

    if extracted_dir.exists():
        shutil.rmtree(extracted_dir)
    extracted_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extracted_dir)

    tar_candidates = list(extracted_dir.rglob("*.tar.gz"))
    for tar_path in tar_candidates:
        unpack_dir = tar_path.parent / tar_path.stem.replace(".tar", "")
        unpack_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(unpack_dir)

    data_candidates = list(extracted_dir.rglob("data.csv"))
    label_candidates = list(extracted_dir.rglob("labels.csv"))
    if not data_candidates or not label_candidates:
        raise FileNotFoundError("Could not find data.csv and labels.csv in extracted archive.")

    features_src = data_candidates[0]
    labels_src = label_candidates[0]
    features_dst = out_dir / "features.csv"
    labels_dst = out_dir / "labels.csv"
    shutil.copy2(features_src, features_dst)
    shutil.copy2(labels_src, labels_dst)

    shutil.rmtree(extracted_dir, ignore_errors=True)
    zip_path.unlink(missing_ok=True)

    print(f"Saved features to: {features_dst}")
    print(f"Saved labels to:   {labels_dst}")


if __name__ == "__main__":
    main()
