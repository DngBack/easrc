# easrc
Explainable Learning-to-Reject for Bioinformatics with Risk-Controlled Calibration?

## UCI RNA-seq Data Preparation (Smoke Test)

Data pipeline code lives under `easrc_uci/`.

### 1) Setup environment

```bash
cd easrc_uci
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Download raw UCI dataset

```bash
cd /home/admin1/Desktop/easrc
source .venv/bin/activate
python3 easrc_uci/scripts/download_uci_tmp.py
```

This creates raw files at:

- `easrc_uci/data/uci_rnaseq/raw/features.csv`
- `easrc_uci/data/uci_rnaseq/raw/labels.csv`

### 3) Supported raw input formats

`src/data/load_uci.py` auto-detects both:

- Two-file format:
  - `features.csv`: `sample_id, gene_0, gene_1, ..., gene_d`
  - `labels.csv`: `sample_id, label`
- One-file format:
  - `uci_rnaseq.csv`: `sample_id, label, gene_0, gene_1, ..., gene_d`

Also supported for UCI-original naming:

- `Unnamed: 0` is treated as `sample_id`
- `Class`/`class` is treated as `label`

### 4) Prepare processed data

```bash
cd /home/admin1/Desktop/easrc
source .venv/bin/activate
python3 easrc_uci/scripts/prepare_data.py --config easrc_uci/configs/uci_rnaseq.yaml --seed 0
```

Script behavior:

1. Loads raw UCI RNA-seq data
2. Encodes labels to integer IDs
3. Creates stratified splits: `base_train`, `rejector_train`, `calibration`, `test`
4. Fits `StandardScaler` on `base_train` only
5. Transforms all splits with that scaler
6. Saves processed artifacts

Output directory:

- `easrc_uci/data/uci_rnaseq/processed/seed_0/`
  - `X.npy`
  - `y.npy`
  - `sample_ids.npy`
  - `feature_names.json`
  - `class_names.json`
  - `splits.json`
  - `scaler.joblib`

### 5) Quick validation checklist

- Split sample counts match config ratios (approximately, due to integer rounding)
- Every split contains all classes
- Scaler is fit on `base_train` only

### 6) Step 1: Train base model (MLP)

Run:

```bash
cd /home/admin1/Desktop/easrc
source .venv/bin/activate
python3 easrc_uci/scripts/train_base.py \
  --config easrc_uci/configs/uci_rnaseq.yaml \
  --seed 0 \
  --base_model mlp
```

Expected output directory:

- `easrc_uci/results/uci_rnaseq/seed_0/base/mlp/`
  - `model.pt`
  - `train_log.csv`
  - `predictions.csv`

### 7) Step 2: Compute explanation + proxy-bio features

Run:

```bash
cd /home/admin1/Desktop/easrc
source .venv/bin/activate
python3 easrc_uci/scripts/compute_features.py \
  --config easrc_uci/configs/uci_rnaseq.yaml \
  --seed 0 \
  --base_model mlp
```

Expected output directory:

- `easrc_uci/results/uci_rnaseq/seed_0/features/mlp/`
  - `attribution_features.csv`
  - `proxy_bio_features.csv`
  - `all_features.csv`
  - `attributions_pred_class.npy`
  - `proxy_groups.json`
