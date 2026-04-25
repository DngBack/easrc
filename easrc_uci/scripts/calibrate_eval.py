from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.selective.calibrate import sweep_and_pick_threshold
from src.selective.risk_coverage import area_under_risk_coverage, test_selective_metrics


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--base_model", type=str, default="mlp")
    parser.add_argument(
        "--no-ucb",
        action="store_true",
        help="Use empirical risks only for feasibility (ignore UCB upper bounds).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Override calibration.alpha from config.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="Override calibration.beta from config.",
    )
    parser.add_argument(
        "--scores-dir",
        type=str,
        default=None,
        help="Directory with method_scores.csv and all_features_with_targets.csv (default: results/.../scores/<base_model>).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directory for eval outputs (default: results/.../eval/<base_model>).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_cfg = config["dataset"]
    cal_cfg = config.get("calibration", {})

    alpha = float(args.alpha if args.alpha is not None else cal_cfg.get("alpha", 0.05))
    beta = float(args.beta if args.beta is not None else cal_cfg.get("beta", 0.30))
    delta = float(cal_cfg.get("delta", 0.10))
    n_thresholds = int(cal_cfg.get("n_thresholds", 200))
    use_ucb = bool(cal_cfg.get("use_ucb", True)) and not args.no_ucb
    min_cov = float(cal_cfg.get("min_empirical_coverage", 0.05))

    results_root = PROJECT_ROOT / dataset_cfg["results_dir"]
    if args.scores_dir:
        scores_dir = Path(args.scores_dir)
    else:
        scores_dir = results_root / f"seed_{args.seed}" / "scores" / args.base_model
    method_scores_path = scores_dir / "method_scores.csv"
    targets_path = scores_dir / "all_features_with_targets.csv"

    if not method_scores_path.exists():
        raise FileNotFoundError(f"Missing {method_scores_path}")
    if not targets_path.exists():
        raise FileNotFoundError(f"Missing {targets_path}")

    method_scores = pd.read_csv(method_scores_path)
    targets = pd.read_csv(targets_path)

    merge_cols = [
        "sample_id",
        "split",
        "y_true",
        "y_pred",
        "cls_loss",
        "xai_loss",
        "audited_loss",
    ]
    missing = [c for c in merge_cols if c not in targets.columns]
    if missing:
        raise ValueError(f"all_features_with_targets missing columns: {missing}")

    merged = method_scores.merge(
        targets[merge_cols],
        on=["sample_id", "split"],
        how="inner",
        validate="many_to_one",
    )

    calib_one = merged[merged["split"] == "calibration"].drop_duplicates(subset=["sample_id"])
    cal_diag = {
        "n_calibration_samples": int(len(calib_one)),
        "xai_loss_min": float(calib_one["xai_loss"].min()),
        "xai_loss_max": float(calib_one["xai_loss"].max()),
        "xai_loss_mean": float(calib_one["xai_loss"].mean()),
        "cls_loss_mean": float(calib_one["cls_loss"].mean()),
    }

    calib_rows_out: list[dict] = []
    test_rows_out: list[dict] = []
    accepted_frames: list[pd.DataFrame] = []

    methods = sorted(merged["method"].unique())

    for method in methods:
        sub = merged[merged["method"] == method]
        cal_df = sub[sub["split"] == "calibration"].copy()
        test_df = sub[sub["split"] == "test"].copy()

        scores_cal = cal_df["score"].to_numpy(dtype=np.float64)
        cls_cal = cal_df["cls_loss"].to_numpy(dtype=np.float64)
        xai_cal = cal_df["xai_loss"].to_numpy(dtype=np.float64)
        audited_cal = cal_df["audited_loss"].to_numpy(dtype=np.float64)

        best_tau, sweep_rows = sweep_and_pick_threshold(
            scores_cal,
            cls_cal,
            xai_cal,
            audited_cal,
            alpha=alpha,
            beta=beta,
            delta=delta,
            n_thresholds=n_thresholds,
            use_ucb=use_ucb,
            min_empirical_coverage=min_cov,
        )

        for row in sweep_rows:
            out = {"method": method, **{k: v for k, v in row.items() if k != "coverage_lcb"}}
            calib_rows_out.append(out)

        feasible = best_tau is not None
        tau_val = float(best_tau) if feasible else float("nan")

        scores_test = test_df["score"].to_numpy(dtype=np.float64)
        cls_test = test_df["cls_loss"].to_numpy(dtype=np.float64)
        xai_test = test_df["xai_loss"].to_numpy(dtype=np.float64)
        audited_test = test_df["audited_loss"].to_numpy(dtype=np.float64)

        aurc = area_under_risk_coverage(scores_test, cls_test)

        tm = test_selective_metrics(
            scores_test,
            cls_test,
            xai_test,
            audited_test,
            tau_val if feasible else float("nan"),
        )

        t_cls = tm["test_cls_risk"]
        t_xai = tm["test_xai_risk"]

        violate_cls = bool(
            feasible
            and not np.isnan(t_cls)
            and t_cls > alpha
        )
        violate_xai = bool(
            feasible
            and not np.isnan(t_xai)
            and t_xai > beta
        )

        test_rows_out.append(
            {
                "method": method,
                "alpha": alpha,
                "beta": beta,
                "delta": delta,
                "threshold": tau_val,
                "feasible": feasible,
                "test_coverage": tm["test_coverage"],
                "test_cls_risk": t_cls,
                "test_xai_risk": t_xai,
                "test_audited_risk": tm["test_audited_risk"],
                "test_aurc": aurc,
                "violate_cls": violate_cls,
                "violate_xai": violate_xai,
                "n_test": tm["n_test"],
                "n_accepted_test": tm["n_accepted_test"],
            }
        )

        thr_for_accept = tau_val if feasible else np.nan
        acc = (test_df["score"].to_numpy() >= thr_for_accept) if feasible else np.zeros(
            len(test_df), dtype=bool
        )

        acc_df = pd.DataFrame(
            {
                "sample_id": test_df["sample_id"].values,
                "split": test_df["split"].values,
                "method": method,
                "score": test_df["score"].values,
                "threshold": thr_for_accept,
                "accepted": acc.astype(int),
                "y_true": test_df["y_true"].values,
                "y_pred": test_df["y_pred"].values,
                "cls_loss": test_df["cls_loss"].values,
                "xai_loss": test_df["xai_loss"].values,
                "audited_loss": test_df["audited_loss"].values,
            }
        )
        accepted_frames.append(acc_df)

    if args.out_dir:
        out_dir = ensure_dir(Path(args.out_dir))
    else:
        out_dir = ensure_dir(results_root / f"seed_{args.seed}" / "eval" / args.base_model)

    calib_df = pd.DataFrame(calib_rows_out)
    cal_cols = [
        "method",
        "threshold",
        "cal_coverage",
        "cal_cls_risk",
        "cal_xai_risk",
        "cal_audited_risk",
        "ucb_cls_risk",
        "ucb_xai_risk",
        "feasible",
        "n_cal",
        "n_accepted_cal",
    ]
    calib_df = calib_df[[c for c in cal_cols if c in calib_df.columns]]
    calib_df.to_csv(out_dir / "calibration_results.csv", index=False, na_rep="NA")

    test_df_out = pd.DataFrame(test_rows_out)
    test_cols = [
        "method",
        "alpha",
        "beta",
        "delta",
        "threshold",
        "feasible",
        "test_coverage",
        "test_cls_risk",
        "test_xai_risk",
        "test_audited_risk",
        "test_aurc",
        "violate_cls",
        "violate_xai",
        "n_test",
        "n_accepted_test",
    ]
    test_df_out = test_df_out[[c for c in test_cols if c in test_df_out.columns]]
    test_df_out.to_csv(out_dir / "test_metrics.csv", index=False, na_rep="NA")

    accepted_all = pd.concat(accepted_frames, axis=0, ignore_index=True)
    accepted_all.to_csv(out_dir / "accepted_samples.csv", index=False)

    meta = {
        "seed": args.seed,
        "base_model": args.base_model,
        "alpha": alpha,
        "beta": beta,
        "delta": delta,
        "n_thresholds": n_thresholds,
        "use_ucb": use_ucb,
        "min_empirical_coverage": min_cov,
        "cli_no_ucb": bool(args.no_ucb),
        "alpha_from_config": args.alpha is None,
        "beta_from_config": args.beta is None,
        "calibration_diagnostics": cal_diag,
        "infeasible_hint": (
            "If all methods infeasible: (1) raise beta above calibration xai_loss mean; "
            "(2) try use_ucb: false or larger calibration split; "
            "(3) see calibration_diagnostics in this file."
        ),
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote evaluation to: {out_dir}")
    print(test_df_out[["method", "feasible", "threshold", "test_coverage", "test_cls_risk", "test_xai_risk"]].to_string(index=False))

    if not test_df_out["feasible"].any():
        print(
            "\nNote: no feasible method under current alpha/beta/UCB settings. "
            "On UCI with ~160 calibration points and xai_unreliability ≈ 0.55, "
            "beta often must exceed ~0.55 for non-empty acceptance; with use_ucb, "
            "alpha=0.05 may require a much larger calibration set. "
            "Try: --no-ucb --beta 0.58",
            flush=True,
        )


if __name__ == "__main__":
    main()
