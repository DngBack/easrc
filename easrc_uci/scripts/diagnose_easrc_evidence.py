"""
Ordered diagnostics for selective EASRC vs baselines (user checklist).

1. Beta sweep (calibrate_eval only; reuses current scores/).
2. accept_target distribution on rejector_train.
3. corr(score, -xai_loss) per method (rejector_train by default).
4. Score saturation heuristic -> suggest lower target_gamma if needed.
5. n_test / cohort size hints if test set is tiny.
6. Report proxy_bio.mode (pathway vs proxy) from config.

Writes under results/<dataset>/seed_<s>/diagnostics/ by default.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_cmd(argv: list[str]) -> None:
    r = subprocess.run(argv, cwd=str(PROJECT_ROOT), check=False)
    if r.returncode != 0:
        raise RuntimeError(f"Command failed ({r.returncode}): {' '.join(argv)}")


def pearsonr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan"), float("nan")
    x, y = x[m], y[m]
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan"), float("nan")
    r = np.corrcoef(x, y)[0, 1]
    # two-sided p-value approx not needed; return r only as second nan for API simplicity
    return float(r), float("nan")


def saturation_flags(scores: np.ndarray) -> dict:
    s = np.asarray(scores, dtype=np.float64)
    s = s[np.isfinite(s)]
    if len(s) < 5:
        return {"n": len(s), "strong_saturation": True, "reason": "too_few_points"}
    std = float(np.std(s))
    rng = float(np.max(s) - np.min(s))
    mean = float(np.mean(s))
    cv = std / (abs(mean) + 1e-8)
    # Heuristic: very low spread relative to [0,1] scale for learned rejectors
    narrow_range = rng < 0.08
    low_std = std < 0.03
    low_cv = cv < 0.05
    strong = bool(narrow_range or low_std or (low_cv and rng < 0.15))
    return {
        "n": len(s),
        "std": std,
        "range": rng,
        "mean": mean,
        "cv": cv,
        "strong_saturation": strong,
        "narrow_range_lt_0_08": narrow_range,
        "low_std_lt_0_03": low_std,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--base-model", type=str, default="mlp")
    ap.add_argument(
        "--betas",
        type=str,
        default="0.52,0.56,0.58,0.60,0.62,0.66",
        help="Comma-separated beta values for sweep (step 1).",
    )
    ap.add_argument(
        "--corr-split",
        type=str,
        default="rejector_train",
        help="Split for corr(score, -xai_loss) (step 3).",
    )
    ap.add_argument("--no-ucb", action="store_true")
    ap.add_argument(
        "--skip-beta-sweep",
        action="store_true",
        help="Skip step 1 (reuse existing diag dirs or faster re-run).",
    )
    args = ap.parse_args()

    cfg_path = (PROJECT_ROOT / args.config).resolve()
    config = load_config(cfg_path)
    ds = config["dataset"]
    results_root = PROJECT_ROOT / ds["results_dir"]
    diag_root = results_root / f"seed_{args.seed}" / "diagnostics"
    diag_root.mkdir(parents=True, exist_ok=True)

    scores_dir = results_root / f"seed_{args.seed}" / "scores" / args.base_model
    targets_path = scores_dir / "all_features_with_targets.csv"
    scores_path = scores_dir / "method_scores.csv"
    if not targets_path.exists() or not scores_path.exists():
        raise FileNotFoundError(f"Need {targets_path} and {scores_path}")

    targets = pd.read_csv(targets_path)
    method_scores = pd.read_csv(scores_path)

    report: dict = {}

    # --- 6. Pathway / proxy_bio ---
    pb = config.get("proxy_bio", {})
    report["proxy_bio"] = {
        "enabled": pb.get("enabled"),
        "mode": pb.get("mode"),
        "gmt_path": pb.get("gmt_path"),
        "note": (
            "mode=pathway uses GMT pathway alignment (not synthetic proxy features). "
            "Use a full MSigDB GMT for production evidence."
        ),
    }

    # --- 2. accept_target on rejector_train ---
    rt = targets[targets["split"] == "rejector_train"]
    if len(rt) == 0:
        accept_summary = {"error": "no rejector_train rows in all_features_with_targets"}
    else:
        vc = rt["accept_target"].value_counts().sort_index()
        accept_summary = {
            "n": int(len(rt)),
            "value_counts": {str(k): int(v) for k, v in vc.items()},
            "positive_rate": float(rt["accept_target"].mean()),
        }
    report["accept_target_rejector_train"] = accept_summary
    with open(diag_root / "accept_target_rejector_train.json", "w", encoding="utf-8") as f:
        json.dump(accept_summary, f, indent=2)

    # --- 3. corr(score, -xai_loss) ---
    merge_cols = ["sample_id", "split", "xai_loss"]
    tsub = targets[merge_cols].drop_duplicates(subset=["sample_id", "split"])
    merged = method_scores.merge(tsub, on=["sample_id", "split"], how="inner", validate="many_to_one")
    sub = merged[merged["split"] == args.corr_split].copy()
    sub["neg_xai_loss"] = -sub["xai_loss"].astype(np.float64)
    corr_rows = []
    for method in sorted(sub["method"].unique()):
        m = sub[sub["method"] == method]
        r, _ = pearsonr(m["score"].to_numpy(), m["neg_xai_loss"].to_numpy())
        corr_rows.append(
            {
                "method": method,
                "split": args.corr_split,
                "n": len(m),
                "corr_score_neg_xai_loss": None if (r is None or (isinstance(r, float) and math.isnan(r))) else r,
            }
        )
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(diag_root / "corr_score_neg_xai_loss.csv", index=False)
    report["corr_score_neg_xai_loss"] = corr_rows

    # --- 4. Saturation (EASRCFullProxy on rejector_train) ---
    eas = sub[sub["method"] == "EASRCFullProxy"]["score"].to_numpy()
    sat = saturation_flags(eas)
    gamma = float(config.get("rejector", {}).get("target_gamma", 0.30))
    if sat.get("strong_saturation"):
        sat["recommendation"] = (
            f"Score spread looks saturated; try decreasing rejector.target_gamma "
            f"(currently {gamma}). Example: max(0.22, {gamma} - 0.04)."
        )
    else:
        sat["recommendation"] = "Saturation heuristic not triggered; gamma change optional."
    report["easrc_score_saturation_rejector_train"] = sat
    with open(diag_root / "saturation_easrc.json", "w", encoding="utf-8") as f:
        json.dump(sat, f, indent=2)

    # --- 5. n_test and cohort ---
    n_by_split = targets.groupby("split").size().to_dict()
    n_by_split = {str(k): int(v) for k, v in n_by_split.items()}
    n_test = int(targets[targets["split"] == "test"].shape[0])
    report["split_counts"] = n_by_split
    report["n_test"] = n_test
    if n_test <= 40:
        report["cohort_recommendation"] = (
            f"n_test={n_test} is small; selective metrics are noisy. "
            "Download more TCGA samples (scripts/download_tcga_gdc.py), add tumor types, "
            "or relax test fraction in configs to increase test size only if scientifically justified."
        )
    else:
        report["cohort_recommendation"] = "n_test acceptable for rough stability; multi-seed still advised."

    # --- 1. Beta sweep ---
    betas = [float(x) for x in args.betas.split(",") if x.strip()]
    beta_rows = []
    py = sys.executable
    cal_py = str(PROJECT_ROOT / "scripts" / "calibrate_eval.py")
    cfg_arg = str(cfg_path.relative_to(PROJECT_ROOT)) if cfg_path.is_relative_to(PROJECT_ROOT) else str(cfg_path)

    if not args.skip_beta_sweep:
        for beta in betas:
            out_dir = diag_root / "beta_sweep" / f"b_{beta:.3f}".replace(".", "p")
            out_dir.mkdir(parents=True, exist_ok=True)
            argv = [
                py,
                cal_py,
                "--config",
                cfg_arg,
                "--seed",
                str(args.seed),
                "--base_model",
                args.base_model,
                "--beta",
                str(beta),
                "--out-dir",
                str(out_dir),
            ]
            if args.no_ucb:
                argv.append("--no-ucb")
            run_cmd(argv)
            tm = pd.read_csv(out_dir / "test_metrics.csv")
            for _, row in tm.iterrows():
                beta_rows.append(
                    {
                        "beta": beta,
                        "method": row["method"],
                        "feasible": bool(row["feasible"]),
                        "test_coverage": float(row["test_coverage"]),
                        "test_xai_risk": float(row["test_xai_risk"]),
                        "test_audited_risk": float(row["test_audited_risk"]),
                        "n_test": int(row["n_test"]),
                    }
                )
        beta_df = pd.DataFrame(beta_rows)
        beta_df.to_csv(diag_root / "beta_sweep_summary.csv", index=False)
        report["beta_sweep_csv"] = str(diag_root / "beta_sweep_summary.csv")

    # EASRC vs MaxProb gap at config beta (from last sweep row or main eval)
    main_eval = results_root / f"seed_{args.seed}" / "eval" / args.base_model / "test_metrics.csv"
    if main_eval.exists():
        me = pd.read_csv(main_eval)
        eas = me[me["method"] == "EASRCFullProxy"].iloc[0]
        mp = me[me["method"] == "MaxProb"].iloc[0]
        report["main_eval_gap"] = {
            "test_xai_risk_EASRC_minus_MaxProb": float(eas["test_xai_risk"] - mp["test_xai_risk"]),
            "test_coverage_EASRC_minus_MaxProb": float(eas["test_coverage"] - mp["test_coverage"]),
            "n_test": int(eas["n_test"]),
        }

    def _json_default(o: object):
        if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
            return None
        raise TypeError(type(o))

    with open(diag_root / "diagnostics_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=_json_default, allow_nan=False)

    # Short narrative for papers / appendices
    eas_c = next((x["corr_score_neg_xai_loss"] for x in corr_rows if x["method"] == "EASRCFullProxy"), None)
    mp_c = next((x["corr_score_neg_xai_loss"] for x in corr_rows if x["method"] == "MaxProb"), None)
    lines = [
        "EASRC evidence diagnostics (automated)",
        "",
        f"1. Beta sweep: see {diag_root / 'beta_sweep_summary.csv'} (beta too low -> infeasible selective rules).",
        f"2. accept_target on rejector_train: n={accept_summary.get('n')}, "
        f"positive_rate={accept_summary.get('positive_rate')}, counts={accept_summary.get('value_counts')}.",
        f"3. corr(score, -xai_loss) on {args.corr_split}: EASRCFullProxy={eas_c}, MaxProb={mp_c}. "
        "Positive correlation means higher accept score associates with lower xai_loss (more reliable explanations).",
        f"4. EASRC score saturation (rejector_train): strong_saturation={sat.get('strong_saturation')}. {sat.get('recommendation')}",
        f"5. Cohort: n_test={n_test}. {report.get('cohort_recommendation')}",
        f"6. proxy_bio: mode={pb.get('mode')}, gmt={pb.get('gmt_path')}.",
        "",
        "Main eval (current eval/mlp/test_metrics.csv): "
        f"test_xai_risk (EASRC - MaxProb) = {report.get('main_eval_gap', {}).get('test_xai_risk_EASRC_minus_MaxProb')}; "
        "negative means lower selective xai risk for EASRC on the accepted test subset.",
    ]
    summary_path = diag_root / "evidence_summary.txt"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2, default=_json_default, allow_nan=False))
    print(f"\nWrote diagnostics under: {diag_root}", flush=True)


if __name__ == "__main__":
    main()
