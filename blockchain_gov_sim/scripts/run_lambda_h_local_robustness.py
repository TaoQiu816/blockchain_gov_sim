"""只对 chain.lambda_h 做最小局部鲁棒性复查。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_yaml(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(payload, file, sort_keys=False, allow_unicode=True)


def _fmt_lambda(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _ensure_run(
    *,
    lambda_h: float,
    seed: int,
    base_config: Path,
    train_config: Path,
    output_root: Path,
    override_dir: Path,
) -> Path:
    run_name = f"lambdah{_fmt_lambda(lambda_h).replace('.', '')}_seed{seed}"
    run_dir = output_root / "hierarchical" / run_name
    summary_path = run_dir / "stage2_high_train" / "training_export_summary.json"
    compare_path = run_dir / "hard_eval" / "hard_compare.json"
    if summary_path.exists() and compare_path.exists():
        return run_dir

    override = {
        "run_name": run_name,
        "output_root": str(output_root),
        "env": {
            "h_min": 0.64,
            "h_warn": 0.72,
            "cost_weights": {
                "timeout": 0.08,
                "margin": 0.15,
            },
        },
        "chain": {
            "lambda_h": float(lambda_h),
            "h_warn": 0.72,
        },
    }
    override_path = override_dir / f"{run_name}.yaml"
    _write_yaml(override_path, override)
    cmd = [
        sys.executable,
        "scripts/train_hierarchical_formal.py",
        "--config",
        str(base_config),
        "--train-config",
        str(train_config),
        "--override",
        "configs/default_theory_best_workpoint.yaml",
        "--override",
        str(override_path),
        "--variant",
        "final",
        "--seed",
        str(seed),
        "--run-name",
        run_name,
        "--output-root",
        str(output_root),
    ]
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    return run_dir


def _load_row(lambda_h: float, seed: int, run_dir: Path) -> dict:
    train = json.load((run_dir / "stage2_high_train" / "training_export_summary.json").open("r", encoding="utf-8"))
    ep = train["episode_stats"]
    step = train["step_stats"]
    hard = json.load((run_dir / "hard_eval" / "hard_compare.json").open("r", encoding="utf-8"))["scenarios"]
    top_ratio = max((item.get("ratio", 0.0) for item in step.get("high_template_usage", [])), default=1.0)
    top3 = "; ".join(f"{item['template']}:{item['ratio']:.2f}" for item in step.get("high_template_usage", [])[:3])
    return {
        "lambda_h": float(lambda_h),
        "seed": int(seed),
        "stage2_unsafe": float(ep["mean_unsafe_rate"]),
        "stage2_timeout": float(ep.get("mean_timeout_rate", 0.0)),
        "stage2_structural_infeasible": float(ep.get("mean_structural_infeasible_rate", 0.0)),
        "lambda_final": float(ep["final_lambda"]),
        "malicious_burst_unsafe": float(hard["malicious_burst"]["hierarchical_learned"]["unsafe_rate"]),
        "high_rtt_burst_unsafe": float(hard["high_rtt_burst"]["hierarchical_learned"]["unsafe_rate"]),
        "high_rtt_burst_timeout": float(hard["high_rtt_burst"]["hierarchical_learned"]["timeout_failure_rate"]),
        "top3_high_templates": top3,
        "dominant_high_template_ratio": float(top_ratio),
        "single_template_collapsed": int(top_ratio >= 0.90),
        "run_dir": str(run_dir),
    }


def _reuse_or_run(
    *,
    lambda_h: float,
    seed: int,
    base_config: Path,
    train_config: Path,
    output_root: Path,
    override_dir: Path,
) -> Path:
    if abs(lambda_h - 0.25) < 1e-9 and seed in (42, 43):
        existing = PROJECT_ROOT / "outputs/theory_workpoint_calibration/hierarchical" / f"recheck_seed{seed}"
        summary_path = existing / "stage2_high_train" / "training_export_summary.json"
        compare_path = existing / "hard_eval" / "hard_compare.json"
        if summary_path.exists() and compare_path.exists():
            return existing
    return _ensure_run(
        lambda_h=lambda_h,
        seed=seed,
        base_config=base_config,
        train_config=train_config,
        output_root=output_root,
        override_dir=override_dir,
    )


def _needs_midpoint(frame: pd.DataFrame) -> bool:
    agg = (
        frame[frame["lambda_h"].isin([0.20, 0.30])]
        .groupby("lambda_h")
        .agg(
            malicious_mean=("malicious_burst_unsafe", "mean"),
            high_rtt_unsafe_mean=("high_rtt_burst_unsafe", "mean"),
            high_rtt_timeout_mean=("high_rtt_burst_timeout", "mean"),
            stage2_unsafe_mean=("stage2_unsafe", "mean"),
            lambda_final_mean=("lambda_final", "mean"),
        )
        .reset_index()
        .sort_values("lambda_h")
    )
    if len(agg) != 2:
        return False
    low = agg.iloc[0]
    high = agg.iloc[1]
    better_low = 0
    better_high = 0
    for left_key, right_key in (
        ("malicious_mean", "malicious_mean"),
        ("high_rtt_unsafe_mean", "high_rtt_unsafe_mean"),
        ("high_rtt_timeout_mean", "high_rtt_timeout_mean"),
        ("stage2_unsafe_mean", "stage2_unsafe_mean"),
        ("lambda_final_mean", "lambda_final_mean"),
    ):
        if float(low[left_key]) + 1e-9 < float(high[right_key]):
            better_low += 1
        elif float(high[right_key]) + 1e-9 < float(low[left_key]):
            better_high += 1
    return better_low > 0 and better_high > 0


def _pick_best(agg: pd.DataFrame) -> dict:
    return min(
        agg.to_dict(orient="records"),
        key=lambda row: (
            round(float(row["malicious_burst_unsafe_mean"]), 6),
            round(float(row["high_rtt_burst_unsafe_mean"]), 6),
            round(float(row["high_rtt_burst_timeout_mean"]), 6),
            round(float(row["stage2_unsafe_mean"]), 6),
            round(float(row["lambda_final_mean"]), 6),
            round(float(row["dominant_high_template_ratio_mean"]), 6),
            round(float(row["stage2_structural_infeasible_mean"]), 6),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--train-config", default="configs/train_hierarchical_theory_smoke.yaml")
    parser.add_argument("--output-root", default="outputs/lambda_h_local_robustness")
    parser.add_argument("--analysis-dir", default="outputs/final_chapter4_closure")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = parser.parse_args()

    base_config = PROJECT_ROOT / args.config
    train_config = PROJECT_ROOT / args.train_config
    output_root = PROJECT_ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    override_dir = output_root / "overrides"
    override_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir = PROJECT_ROOT / args.analysis_dir
    analysis_dir.mkdir(parents=True, exist_ok=True)

    lambda_candidates = [0.20, 0.25, 0.30]
    rows: list[dict] = []

    for lambda_h in lambda_candidates:
        for seed in args.seeds:
            run_dir = _reuse_or_run(
                lambda_h=float(lambda_h),
                seed=int(seed),
                base_config=base_config,
                train_config=train_config,
                output_root=output_root,
                override_dir=override_dir,
            )
            rows.append(_load_row(lambda_h=float(lambda_h), seed=int(seed), run_dir=run_dir))

    frame = pd.DataFrame(rows).sort_values(["lambda_h", "seed"])
    if _needs_midpoint(frame):
        for seed in args.seeds:
            run_dir = _ensure_run(
                lambda_h=0.275,
                seed=int(seed),
                base_config=base_config,
                train_config=train_config,
                output_root=output_root,
                override_dir=override_dir,
            )
            rows.append(_load_row(lambda_h=0.275, seed=int(seed), run_dir=run_dir))
        frame = pd.DataFrame(rows).sort_values(["lambda_h", "seed"])

    frame.to_csv(analysis_dir / "lambda_h_local_robustness_summary.csv", index=False)

    agg = (
        frame.groupby("lambda_h")
        .agg(
            stage2_unsafe_mean=("stage2_unsafe", "mean"),
            stage2_timeout_mean=("stage2_timeout", "mean"),
            stage2_structural_infeasible_mean=("stage2_structural_infeasible", "mean"),
            lambda_final_mean=("lambda_final", "mean"),
            malicious_burst_unsafe_mean=("malicious_burst_unsafe", "mean"),
            high_rtt_burst_unsafe_mean=("high_rtt_burst_unsafe", "mean"),
            high_rtt_burst_timeout_mean=("high_rtt_burst_timeout", "mean"),
            dominant_high_template_ratio_mean=("dominant_high_template_ratio", "mean"),
            collapsed_runs=("single_template_collapsed", "sum"),
        )
        .reset_index()
        .sort_values("lambda_h")
    )

    best_row = _pick_best(agg)
    lines = [
        "# Lambda H Judgment",
        "",
        "## Aggregate",
        "",
        agg.to_markdown(index=False),
        "",
        "## Judgment",
        "",
        f"- 推荐 `chain.lambda_h = {best_row['lambda_h']:.3f}`。",
        f"- 该点 `malicious_burst unsafe / high_rtt_burst unsafe / high_rtt_burst timeout` 均值分别为 `{best_row['malicious_burst_unsafe_mean']:.4f} / {best_row['high_rtt_burst_unsafe_mean']:.4f} / {best_row['high_rtt_burst_timeout_mean']:.4f}`。",
        f"- `Stage2 unsafe` 均值 `{best_row['stage2_unsafe_mean']:.4f}`，`lambda_final` 均值 `{best_row['lambda_final_mean']:.4f}`，主导模板占比均值 `{best_row['dominant_high_template_ratio_mean']:.4f}`。",
        f"- 单模板塌缩次数 `{int(best_row['collapsed_runs'])}`，`structural_infeasible` 均值 `{best_row['stage2_structural_infeasible_mean']:.4f}`。",
    ]
    (analysis_dir / "LAMBDA_H_JUDGMENT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
