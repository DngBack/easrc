from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.explain.explanation_features import (
    attribution_entropy,
    topk_attribution_mass,
    xai_unreliability_score,
)
from src.explain.grad_input import gradient_times_input, gradient_times_input_stability
from src.explain.pathway_alignment import (
    attribution_mass_in_predicted_pathway,
    build_class_pathway_groups,
    feature_name_to_index,
    load_gmt,
    make_random_groups_matched_size,
    pathway_groups_to_jsonable,
    random_pathway_control_alignment,
)
from src.explain.proxy_bio import (
    attribution_mass_in_predicted_group,
    build_proxy_groups,
    make_random_groups,
    random_group_alignment,
)
from src.models.mlp import MLPClassifier
from src.models.train_utils import get_device, set_seed


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


def load_mlp_model(model_path: Path, device: torch.device) -> MLPClassifier:
    ckpt = torch.load(model_path, map_location=device)

    model = MLPClassifier(
        input_dim=int(ckpt["input_dim"]),
        num_classes=int(ckpt["num_classes"]),
        hidden_dims=ckpt["hidden_dims"],
        dropout=float(ckpt["dropout"]),
    )

    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model


def ids_to_original_indices(
    prediction_df: pd.DataFrame,
    sample_ids: np.ndarray,
) -> np.ndarray:
    id_to_idx = {str(sample_id): i for i, sample_id in enumerate(sample_ids)}

    indices = []
    for sample_id in prediction_df["sample_id"].tolist():
        key = str(sample_id)
        if key not in id_to_idx:
            raise KeyError(f"sample_id {sample_id} not found in processed sample_ids.")
        indices.append(id_to_idx[key])

    return np.array(indices, dtype=int)


def save_proxy_groups(groups: dict[int, np.ndarray], path: Path) -> None:
    serializable = {str(k): v.astype(int).tolist() for k, v in groups.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--base_model", type=str, default="mlp")
    args = parser.parse_args()

    if args.base_model != "mlp":
        raise ValueError("compute_features.py currently supports --base_model mlp only.")

    set_seed(args.seed)

    config = load_config(args.config)
    dataset_cfg = config["dataset"]
    explain_cfg = config.get("explain", {})
    proxy_cfg = config.get("proxy_bio", {})

    processed_root = PROJECT_ROOT / dataset_cfg["processed_dir"]
    results_root = PROJECT_ROOT / dataset_cfg["results_dir"]

    X, y, sample_ids, splits, class_names, feature_names = load_processed(
        processed_root=processed_root,
        seed=args.seed,
    )

    base_dir = results_root / f"seed_{args.seed}" / "base" / args.base_model
    model_path = base_dir / "model.pt"
    pred_path = base_dir / "predictions.csv"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing predictions file: {pred_path}")

    device = get_device()
    model = load_mlp_model(model_path=model_path, device=device)

    pred_df = pd.read_csv(pred_path)

    # We compute features for all splits already present in predictions.csv.
    row_indices = ids_to_original_indices(pred_df, sample_ids)
    X_rows = X[row_indices].astype(np.float32)

    y_pred = pred_df["y_pred"].to_numpy(dtype=int)

    batch_size = int(explain_cfg.get("batch_size", 64))
    perturb_std = float(explain_cfg.get("perturb_std", 0.01))
    stability_repeats = int(explain_cfg.get("stability_repeats", 5))
    topk_mass_k = int(explain_cfg.get("topk_mass", 100))
    xai_weights = explain_cfg.get("xai_weights", None)

    print("Computing Gradient x Input attributions...")
    attributions = gradient_times_input(
        model=model,
        X=X_rows,
        target_classes=y_pred,
        batch_size=batch_size,
        device=device,
    )

    print("Computing attribution stability...")
    attr_stability = gradient_times_input_stability(
        model=model,
        X=X_rows,
        target_classes=y_pred,
        perturb_std=perturb_std,
        repeats=stability_repeats,
        batch_size=batch_size,
        device=device,
        seed=args.seed,
    )

    print("Computing attribution summary features...")
    attr_entropy = attribution_entropy(attributions)
    topk_mass = topk_attribution_mass(attributions, k=topk_mass_k)
    xai_unrel = xai_unreliability_score(
        attr_entropy=attr_entropy,
        attr_stability=attr_stability,
        topk_mass=topk_mass,
        weights=xai_weights,
    )

    attr_features = pd.DataFrame(
        {
            "sample_id": pred_df["sample_id"].values,
            "split": pred_df["split"].values,
            "attr_entropy": attr_entropy,
            "attr_stability": attr_stability,
            "topk_mass": topk_mass,
            "xai_unreliability": xai_unrel,
        }
    )

    bio_mode = str(proxy_cfg.get("mode", "proxy")).lower()
    rejector_train_indices = np.array(splits["rejector_train"], dtype=int)

    num_classes = len(class_names)
    num_features = X.shape[1]
    topk_per_class = int(proxy_cfg.get("topk_per_class", 200))
    n_random_groups = int(proxy_cfg.get("random_groups", 10))

    if bio_mode == "pathway":
        print("Building pathway gene groups from GMT + class_pathways (not from rejector labels)...")
        gmt_rel = proxy_cfg.get("gmt_path")
        if not gmt_rel:
            raise ValueError("proxy_bio.mode='pathway' requires proxy_bio.gmt_path (MSigDB .gmt file).")
        gmt_path = (PROJECT_ROOT / str(gmt_rel)).resolve()
        class_pathways = proxy_cfg.get("class_pathways") or {}
        if not class_pathways:
            raise ValueError(
                "proxy_bio.mode='pathway' requires proxy_bio.class_pathways "
                "(map class label string -> list of GMT pathway names)."
            )
        pathway_defs = load_gmt(gmt_path)
        gene_to_idx = feature_name_to_index(list(feature_names))
        class_pathways_norm = {str(k): [str(p) for p in v] for k, v in class_pathways.items()}
        proxy_groups = build_class_pathway_groups(
            class_names=list(class_names),
            class_pathways=class_pathways_norm,
            pathway_defs=pathway_defs,
            gene_to_idx=gene_to_idx,
        )
        target_sizes = {c: int(proxy_groups[c].size) for c in range(num_classes)}
        random_groups = make_random_groups_matched_size(
            num_classes=num_classes,
            num_features=num_features,
            target_sizes=target_sizes,
            n_random_groups=n_random_groups,
            seed=args.seed,
        )
        print("Computing pathway attribution alignment (predicted class)...")
        proxy_alignment = attribution_mass_in_predicted_pathway(
            attributions=attributions,
            predicted_classes=y_pred,
            groups=proxy_groups,
        )
        random_alignment = random_pathway_control_alignment(
            attributions=attributions,
            predicted_classes=y_pred,
            random_groups=random_groups,
        )
    elif bio_mode == "proxy":
        print("Building proxy-bio groups from rejector_train only...")
        proxy_groups = build_proxy_groups(
            X=X,
            y=y,
            rejector_train_indices=rejector_train_indices,
            num_classes=num_classes,
            topk_per_class=topk_per_class,
        )

        random_groups = make_random_groups(
            num_classes=num_classes,
            num_features=num_features,
            group_size=min(topk_per_class, num_features),
            n_random_groups=n_random_groups,
            seed=args.seed,
        )

        print("Computing proxy-bio alignment...")
        proxy_alignment = attribution_mass_in_predicted_group(
            attributions=attributions,
            predicted_classes=y_pred,
            groups=proxy_groups,
        )

        random_alignment = random_group_alignment(
            attributions=attributions,
            predicted_classes=y_pred,
            random_groups=random_groups,
        )
    else:
        raise ValueError(f"Unknown proxy_bio.mode: {bio_mode!r}. Use 'proxy' or 'pathway'.")

    proxy_bio_unrel = 1.0 - proxy_alignment
    proxy_gap = proxy_alignment - random_alignment

    proxy_features = pd.DataFrame(
        {
            "sample_id": pred_df["sample_id"].values,
            "split": pred_df["split"].values,
            "proxy_alignment": proxy_alignment,
            "proxy_bio_unreliability": proxy_bio_unrel,
            "random_proxy_alignment": random_alignment,
            "proxy_alignment_gap": proxy_gap,
        }
    )

    # Merge with prediction/confidence features.
    keep_pred_cols = [
        "sample_id",
        "split",
        "y_true",
        "y_pred",
        "correct",
        "max_prob",
        "entropy",
        "margin",
        "energy",
    ]

    all_features = pred_df[keep_pred_cols].merge(
        attr_features,
        on=["sample_id", "split"],
        how="inner",
    )

    all_features = all_features.merge(
        proxy_features,
        on=["sample_id", "split"],
        how="inner",
    )

    # Basic safety checks.
    required_cols = [
        "max_prob",
        "entropy",
        "margin",
        "energy",
        "attr_entropy",
        "attr_stability",
        "topk_mass",
        "xai_unreliability",
        "proxy_alignment",
        "proxy_bio_unreliability",
        "random_proxy_alignment",
        "proxy_alignment_gap",
    ]

    if all_features[required_cols].isna().any().any():
        bad_cols = all_features[required_cols].columns[
            all_features[required_cols].isna().any()
        ].tolist()
        raise ValueError(f"NaN detected in feature columns: {bad_cols}")

    out_dir = ensure_dir(
        results_root / f"seed_{args.seed}" / "features" / args.base_model
    )

    np.save(out_dir / "attributions_pred_class.npy", attributions.astype(np.float32))

    attr_features.to_csv(out_dir / "attribution_features.csv", index=False)
    proxy_features.to_csv(out_dir / "proxy_bio_features.csv", index=False)
    all_features.to_csv(out_dir / "all_features.csv", index=False)

    if bio_mode == "pathway":
        groups_payload = pathway_groups_to_jsonable(
            proxy_groups, list(class_names), list(feature_names)
        )
        with (out_dir / "proxy_groups.json").open("w", encoding="utf-8") as f:
            json.dump(groups_payload, f, indent=2)
    else:
        save_proxy_groups(proxy_groups, out_dir / "proxy_groups.json")

    print(f"Features saved to: {out_dir}")
    print("Feature summary:")
    print(
        all_features[
            [
                "attr_entropy",
                "attr_stability",
                "topk_mass",
                "xai_unreliability",
                "proxy_alignment",
                "random_proxy_alignment",
                "proxy_alignment_gap",
            ]
        ].describe()
    )


if __name__ == "__main__":
    main()
