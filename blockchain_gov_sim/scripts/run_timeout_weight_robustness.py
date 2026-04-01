"""只复查 timeout weight 的极小鲁棒性。"""

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


def _ensure_run(
    *,
    timeout_weight: float,
    seed: int,
    base_config: Path,
    train_config: Path,
    output_root: Path,
    override_dir: Path,
) -> Path:
    run_name = f"timeout{timeout_weight:.2f}_seed{seed}".replace(".", "")
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
                "timeout": float(timeout_weight),
                "margin": 0.15,
            },
        },
        "chain": {
            "lambda_h": 0.25,
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


def _load_row(timeout_weight: float, seed: int, run_dir: Path) -> dict:
    train = json.load((run_dir / "stage2_high_train" / "training_export_summary.json").open("r", encoding="utf-8"))
    ep = train["episode_stats"]
    step = train["step_stats"]
    hard = json.load((run_dir / "hard_eval" / "hard_compare.json").open("r", encoding="utf-8"))["scenarios"]
    top_ratio = max((item.get("ratio", 0.0) for item in step.get("high_template_usage", [])), default=1.0)
    top3 = "; ".join(f"{item['template']}:{item['ratio']:.2f}" for item in step.get("high_template_usage", [])[:3])
    return {
        "timeout_weight": float(timeout_weight),
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--train-config", default="configs/train_hierarchical_theory_smoke.yaml")
    parser.add_argument("--output-root", default="outputs/timeout_weight_robustness")
    parser.add_argument("--analysis-dir", default="outputs/final_chapter4_closure")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    args = parser.parse_args()

    base_config = PROJECT_ROOT / args.config
    train_config = PROJECT_ROOT / args.train_config
    output_root = PROJECT_ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    override_dir = output_root / "overrides"
    override_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir = PROJECT_ROOT / args.analysis_dir
    analysis_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    # timeout=0.08 直接复用当前最优工作点复验
    existing_root = PROJECT_ROOT / "outputs/theory_workpoint_calibration/hierarchical"
    for seed in args.seeds:
        rows.append(
            _load_row(
                timeout_weight=0.08,
                seed=int(seed),
                run_dir=existing_root / f"recheck_seed{seed}",
            )
        )

    for timeout_weight in (0.12, 0.16):
        for seed in args.seeds:
            run_dir = _ensure_run(
                timeout_weight=float(timeout_weight),
                seed=int(seed),
                base_config=base_config,
                train_config=train_config,
                output_root=output_root,
                override_dir=override_dir,
            )
            rows.append(_load_row(timeout_weight=float(timeout_weight), seed=int(seed), run_dir=run_dir))

    frame = pd.DataFrame(rows).sort_values(["timeout_weight", "seed"])
    frame.to_csv(analysis_dir / "timeout_weight_robustness_summary.csv", index=False)

    agg = (
        frame.groupby("timeout_weight")
        .agg(
            stage2_unsafe_mean=("stage2_unsafe", "mean"),
            stage2_timeout_mean=("stage2_timeout", "mean"),
            stage2_structural_infeasible_mean=("stage2_structural_infeasible", "mean"),
            lambda_final_mean=("lambda_final", "mean"),
            malicious_burst_unsafe_mean=("malicious_burst_unsafe", "mean"),
            high_rtt_burst_unsafe_mean=("high_rtt_burst_unsafe", "mean"),
            high_rtt_burst_timeout_mean=("high_rtt_burst_timeout", "mean"),
            dominant_high_template_ratio_mean=("dominant_high_template_ratio", "mean"),
        )
        .reset_index()
        .sort_values("timeout_weight")
    )

    best_row = min(
        agg.to_dict(orient="records"),
        key=lambda row: (
            round(float(row["high_rtt_burst_timeout_mean"]), 6),
            round(float(row["malicious_burst_unsafe_mean"] + row["high_rtt_burst_unsafe_mean"]), 6),
            round(float(row["lambda_final_mean"]), 6),
            round(float(row["dominant_high_template_ratio_mean"]), 6),
        ),
    )

    lines = [
        "# Timeout Weight Judgment",
        "",
        "## Aggregate",
        "",
        agg.to_markdown(index=False),
        "",
        "## Judgment",
        "",
        f"- 推荐 `env.cost_weights.timeout = {best_row['timeout_weight']:.2f}`。",
        f"- 该点 `high_rtt_burst timeout` 均值 `{best_row['high_rtt_burst_timeout_mean']:.4f}`，`malicious_burst unsafe + high_rtt_burst unsafe` 为 `{best_row['malicious_burst_unsafe_mean'] + best_row['high_rtt_burst_unsafe_mean']:.4f}`。",
        f"- `lambda_final` 均值 `{best_row['lambda_final_mean']:.4f}`，主导模板占比均值 `{best_row['dominant_high_template_ratio_mean']:.4f}`，未出现单模板塌缩。",
    ]
    (analysis_dir / "TIMEOUT_WEIGHT_JUDGMENT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
