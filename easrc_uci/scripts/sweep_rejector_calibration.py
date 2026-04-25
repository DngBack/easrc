"""
Grid search: target_gamma (rejector targets) × beta (calibration).

For each gamma, retrains rejectors into a dedicated scores directory; for each
(gamma, beta), runs calibrate_eval into a dedicated eval directory. Writes
sweep_summary.csv and optionally reapplies the best setting to the default
results/.../scores/mlp and eval/mlp paths.

Example:
  cd easrc_uci && python scripts/sweep_rejector_calibration.py \\
    --config configs/tcga_rnaseq.yaml --seed 0 --apply-best
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

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


def row_metrics(test_csv: Path, method: str) -> dict | None:
    if not test_csv.exists():
        return None
    df = pd.read_csv(test_csv)
    sub = df[df["method"] == method]
    if len(sub) != 1:
        return None
    return sub.iloc[0].to_dict()


def score_pair(easrc: dict | None, baseline: dict | None) -> float:
    """
    Higher is better: lower EASRC xai than baseline, similar or higher coverage,
    lower audited risk on EASRC.
    """
    if easrc is None or baseline is None:
        return float("-inf")
    if not easrc.get("feasible") or not baseline.get("feasible"):
        return float("-inf")
    gap_xai = float(baseline["test_xai_risk"]) - float(easrc["test_xai_risk"])
    gap_cov = float(easrc["test_coverage"]) - float(baseline["test_coverage"])
    gap_aud = float(baseline["test_audited_risk"]) - float(easrc["test_audited_risk"])
    return gap_xai + 0.35 * gap_cov + 0.5 * gap_aud


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--base-model", type=str, default="mlp")
    ap.add_argument(
        "--gammas",
        type=str,
        default="0.24,0.28,0.32,0.36,0.40",
        help="Comma-separated target_gamma values.",
    )
    ap.add_argument(
        "--betas",
        type=str,
        default="0.56,0.58,0.60,0.62,0.64",
        help="Comma-separated calibration beta overrides.",
    )
    ap.add_argument(
        "--baseline-method",
        type=str,
        default="MaxProb",
        help="Baseline method name in test_metrics.csv for comparison.",
    )
    ap.add_argument("--no-ucb", action="store_true", help="Pass --no-ucb to calibrate_eval.")
    ap.add_argument(
        "--apply-best",
        action="store_true",
        help="Retrain rejectors + calibrate with best (gamma, beta) into default result dirs.",
    )
    ap.add_argument(
        "--sweep-root",
        type=str,
        default=None,
        help="Root for sweep artifacts (default: results/<dataset>/seed_<s>/sweep_rc).",
    )
    args = ap.parse_args()

    cfg_path = (PROJECT_ROOT / args.config).resolve()
    config = load_config(cfg_path)
    dataset = config["dataset"]["name"]
    results_rel = config["dataset"]["results_dir"]
    results_root = PROJECT_ROOT / results_rel

    if args.sweep_root:
        sweep_root = Path(args.sweep_root)
    else:
        sweep_root = results_root / f"seed_{args.seed}" / "sweep_rc"
    sweep_root.mkdir(parents=True, exist_ok=True)

    gammas = [float(x) for x in args.gammas.split(",") if x.strip()]
    betas = [float(x) for x in args.betas.split(",") if x.strip()]

    py = sys.executable
    train_py = str(PROJECT_ROOT / "scripts" / "train_rejectors.py")
    cal_py = str(PROJECT_ROOT / "scripts" / "calibrate_eval.py")
    cfg_arg = str(cfg_path.relative_to(PROJECT_ROOT)) if cfg_path.is_relative_to(PROJECT_ROOT) else str(cfg_path)

    rows: list[dict] = []

    for gamma in gammas:
        gtag = f"g{gamma:.3f}".replace(".", "p")
        scores_dir = sweep_root / f"scores_{gtag}"
        train_argv = [
            py,
            train_py,
            "--config",
            cfg_arg,
            "--seed",
            str(args.seed),
            "--base_model",
            args.base_model,
            "--target-gamma",
            str(gamma),
            "--out-dir",
            str(scores_dir),
        ]
        print(f"\n=== train_rejectors gamma={gamma} -> {scores_dir} ===", flush=True)
        run_cmd(train_argv)

        for beta in betas:
            btag = f"b{beta:.3f}".replace(".", "p")
            eval_dir = sweep_root / f"eval_{gtag}_{btag}"
            cal_argv = [
                py,
                cal_py,
                "--config",
                cfg_arg,
                "--seed",
                str(args.seed),
                "--base_model",
                args.base_model,
                "--scores-dir",
                str(scores_dir),
                "--out-dir",
                str(eval_dir),
                "--beta",
                str(beta),
            ]
            if args.no_ucb:
                cal_argv.append("--no-ucb")
            print(f"  calibrate beta={beta} -> {eval_dir}", flush=True)
            run_cmd(cal_argv)

            tm = eval_dir / "test_metrics.csv"
            easrc = row_metrics(tm, "EASRCFullProxy")
            base = row_metrics(tm, args.baseline_method)
            obj = score_pair(easrc, base)
            rows.append(
                {
                    "gamma": gamma,
                    "beta": beta,
                    "objective": obj,
                    "easrc_feasible": easrc["feasible"] if easrc else False,
                    "baseline_feasible": base["feasible"] if base else False,
                    "easrc_test_xai": easrc["test_xai_risk"] if easrc else float("nan"),
                    "easrc_test_cov": easrc["test_coverage"] if easrc else float("nan"),
                    "baseline_test_xai": base["test_xai_risk"] if base else float("nan"),
                    "baseline_test_cov": base["test_coverage"] if base else float("nan"),
                    "eval_dir": str(eval_dir),
                }
            )

    out_df = pd.DataFrame(rows)
    summary_path = sweep_root / "sweep_summary.csv"
    out_df.to_csv(summary_path, index=False)
    print(f"\nWrote {summary_path}", flush=True)

    feasible = out_df[out_df["objective"] > float("-inf")].copy()
    if len(feasible) == 0:
        print("No feasible (gamma, beta) pairs for EASRC and baseline.", flush=True)
        sys.exit(1)

    ref_beta = float(config.get("calibration", {}).get("beta", 0.60))
    feasible["_beta_dist"] = (feasible["beta"] - ref_beta).abs()
    best = feasible.sort_values(
        ["objective", "_beta_dist", "gamma"],
        ascending=[False, True, True],
    ).iloc[0]
    print("\nBest row:", flush=True)
    print(best.to_string(), flush=True)

    best_meta = {
        "best_gamma": float(best["gamma"]),
        "best_beta": float(best["beta"]),
        "best_objective": float(best["objective"]),
        "baseline_method": args.baseline_method,
    }
    with open(sweep_root / "sweep_best.json", "w", encoding="utf-8") as f:
        json.dump(best_meta, f, indent=2)

    if args.apply_best:
        default_scores = results_root / f"seed_{args.seed}" / "scores" / args.base_model
        default_eval = results_root / f"seed_{args.seed}" / "eval" / args.base_model
        print(
            f"\nApplying best gamma={best_meta['best_gamma']} beta={best_meta['best_beta']} "
            f"to {default_scores} and {default_eval}",
            flush=True,
        )
        apply_train = [
            py,
            train_py,
            "--config",
            cfg_arg,
            "--seed",
            str(args.seed),
            "--base_model",
            args.base_model,
            "--target-gamma",
            str(best_meta["best_gamma"]),
            "--out-dir",
            str(default_scores),
        ]
        run_cmd(apply_train)
        apply_cal = [
            py,
            cal_py,
            "--config",
            cfg_arg,
            "--seed",
            str(args.seed),
            "--base_model",
            args.base_model,
            "--beta",
            str(best_meta["best_beta"]),
        ]
        if args.no_ucb:
            apply_cal.append("--no-ucb")
        run_cmd(apply_cal)

        print(
            f"\nTo pin these defaults in YAML (edit by hand to preserve comments):\n"
            f"  rejector.target_gamma: {best_meta['best_gamma']}\n"
            f"  calibration.beta: {best_meta['best_beta']}\n",
            flush=True,
        )


if __name__ == "__main__":
    main()
