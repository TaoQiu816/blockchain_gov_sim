"""汇总治理 DQN 搜索结果并按冻结规则选优。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gov_sim.experiments.benchmark_runner import SCENARIO_ORDER
from gov_sim.utils.io import ensure_dir, write_json


FULL_BUDGET = {
    "num_episodes": 800,
    "episode_length": 120,
    "total_env_steps": 96000,
    "eval_every": 20,
    "eval_episodes": 20,
}

FORMAL_SEEDS = [42, 52, 62]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _csv_block(frame: pd.DataFrame) -> str:
    return "```text\n" + frame.to_csv(index=False).strip() + "\n```"


def _metric(summary: dict[str, Any], primary: str, fallback: str, default: float) -> float:
    return float(summary.get(primary, summary.get(fallback, default)))


def _adv_tuple(dqn: dict[str, Any], baseline: dict[str, Any]) -> tuple[float, float, float, float, float]:
    return (
        _metric(baseline, "unsafe_rate", "unsafe_rate_all_steps", 1.0) - _metric(dqn, "unsafe_rate", "unsafe_rate_all_steps", 1.0),
        _metric(baseline, "structural_infeasible_rate", "structural_infeasible_rate_all_steps", 1.0)
        - _metric(dqn, "structural_infeasible_rate", "structural_infeasible_rate_all_steps", 1.0),
        _metric(baseline, "timeout_rate", "timeout_rate_all_steps", 1.0) - _metric(dqn, "timeout_rate", "timeout_rate_all_steps", 1.0),
        _metric(baseline, "mean_latency", "mean_latency", float("inf")) - _metric(dqn, "mean_latency", "mean_latency", float("inf")),
        _metric(dqn, "TPS", "tps", 0.0) - _metric(baseline, "TPS", "tps", 0.0),
    )


def _selection_key(row: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(row["max_unsafe"]),
        float(row["max_structural_infeasible"]),
        float(row["max_timeout"]),
        float(row["max_mean_latency"]),
        -float(row["min_tps"]),
        -float(row["worst_adv_unsafe"]),
        -float(row["worst_adv_structural_infeasible"]),
        -float(row["worst_adv_timeout"]),
        -float(row["worst_adv_mean_latency"]),
        -float(row["worst_adv_tps"]),
    )


def _load_eval_metrics(eval_summary_path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    scenario_summaries = _load_json(eval_summary_path)
    per_scenario = {str(name): dict(summary) for name, summary in scenario_summaries.items()}
    metrics = {
        "max_unsafe": max(_metric(summary, "unsafe_rate", "unsafe_rate_all_steps", 1.0) for summary in per_scenario.values()),
        "max_structural_infeasible": max(
            _metric(summary, "structural_infeasible_rate", "structural_infeasible_rate_all_steps", 1.0) for summary in per_scenario.values()
        ),
        "max_timeout": max(_metric(summary, "timeout_rate", "timeout_rate_all_steps", 1.0) for summary in per_scenario.values()),
        "max_mean_latency": max(_metric(summary, "mean_latency", "mean_latency", float("inf")) for summary in per_scenario.values()),
        "min_tps": min(_metric(summary, "TPS", "tps", 0.0) for summary in per_scenario.values()),
    }
    return metrics, per_scenario


def _load_baseline_advantage(benchmark_csv: Path) -> dict[str, float]:
    frame = pd.read_csv(benchmark_csv)
    dqn_frame = frame[frame["method"] == "DQN"]
    base_frame = frame[frame["method"] == "Global-Static-Fixed"]
    dqn_by_scenario = {str(row["scenario"]): row.to_dict() for _, row in dqn_frame.iterrows()}
    base_by_scenario = {str(row["scenario"]): row.to_dict() for _, row in base_frame.iterrows()}
    tuples = [
        _adv_tuple(dqn_by_scenario[scenario], base_by_scenario[scenario])
        for scenario in SCENARIO_ORDER
        if scenario in dqn_by_scenario and scenario in base_by_scenario
    ]
    if not tuples:
        return {
            "worst_adv_unsafe": float("-inf"),
            "worst_adv_structural_infeasible": float("-inf"),
            "worst_adv_timeout": float("-inf"),
            "worst_adv_mean_latency": float("-inf"),
            "worst_adv_tps": float("-inf"),
        }
    worst = min(tuples)
    return {
        "worst_adv_unsafe": float(worst[0]),
        "worst_adv_structural_infeasible": float(worst[1]),
        "worst_adv_timeout": float(worst[2]),
        "worst_adv_mean_latency": float(worst[3]),
        "worst_adv_tps": float(worst[4]),
    }


def _formal_commands(best: dict[str, Any]) -> str:
    hp = {
        "lr": best["lr"],
        "target_update_period": int(best["target_update_period"]),
        "warmup_steps": int(best["warmup_steps"]),
        "epsilon_decay_steps": int(best["epsilon_decay_steps"]),
        "train_freq": int(best["train_freq"]),
    }

    def _train_cmd(seed: int, output_dir: str, mixed: bool, extra_override: str | None = None) -> str:
        parts = [
            sys.executable,
            "scripts/train_governance_dqn.py",
            "--config",
            "configs/default.yaml",
        ]
        if extra_override is not None:
            parts.extend(["--override", extra_override])
        parts.extend(
            [
                "--output-dir",
                output_dir,
                "--num-episodes",
                str(FULL_BUDGET["num_episodes"]),
                "--episode-length",
                str(FULL_BUDGET["episode_length"]),
                "--total-env-steps",
                str(FULL_BUDGET["total_env_steps"]),
                "--eval-every",
                str(FULL_BUDGET["eval_every"]),
                "--eval-episodes",
                str(FULL_BUDGET["eval_episodes"]),
                "--seed",
                str(seed),
                "--lr",
                str(hp["lr"]),
                "--target-update-period",
                str(hp["target_update_period"]),
                "--warmup-steps",
                str(hp["warmup_steps"]),
                "--epsilon-decay-steps",
                str(hp["epsilon_decay_steps"]),
                "--train-freq",
                str(hp["train_freq"]),
            ]
        )
        if mixed:
            parts.append("--mixed-scenario")
        return " ".join(parts)

    lines = ["set -euo pipefail", f"cd {ROOT}"]
    lines.append("# Part A: normal-only 3-seed")
    for seed in FORMAL_SEEDS:
        lines.append(_train_cmd(seed, f"outputs/governance_dqn_formal/part_a_normal/seed{seed}/train", mixed=False))
    lines.append("")
    lines.append("# Part B: mixed 3-seed")
    for seed in FORMAL_SEEDS:
        lines.append(_train_cmd(seed, f"outputs/governance_dqn_formal/part_b_mixed/seed{seed}/train", mixed=True))
    lines.append("")
    lines.append("# Part C: checkpoint-based 分场景趋势（建议先用 mixed seed42）")
    lines.append(
        " ".join(
            [
                sys.executable,
                "scripts/eval_governance_dqn_checkpoints.py",
                "--config",
                "configs/default.yaml",
                "--checkpoint-dir",
                "outputs/governance_dqn_formal/part_b_mixed/seed42/train/checkpoints",
                "--output-dir",
                "outputs/governance_dqn_formal/part_c_checkpoint_trend/seed42",
                "--episodes",
                str(FULL_BUDGET["eval_episodes"]),
                "--seed",
                "42",
            ]
        )
    )
    lines.append("")
    lines.append("# Part D: retrain ablation")
    for override_name, tag in (
        ("configs/formal_ablation_no_context_fusion.yaml", "no_context_fusion"),
        ("configs/formal_ablation_single_dim_trust.yaml", "single_dim_trust"),
    ):
        for seed in FORMAL_SEEDS:
            lines.append(_train_cmd(seed, f"outputs/governance_dqn_formal/part_d_ablation/{tag}/seed{seed}/train", mixed=True, extra_override=override_name))
    lines.append("")
    lines.append("# Part E: 两组鲁棒性")
    lines.append("# 需在确认具体 override 组后补命令；当前 DQN 训练/评估脚本已支持重复 --override 叠加。")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-root", default="outputs/governance_dqn_search")
    args = parser.parse_args()

    search_root = ROOT / args.search_root if not Path(args.search_root).is_absolute() else Path(args.search_root)
    output_dir = ensure_dir(search_root / "analysis")
    manifest_payload = _load_json(search_root / "search_manifest.json")
    manifest = manifest_payload["runs"]
    shared_baseline_csv = Path(str(manifest_payload.get("shared_baseline_output_dir", search_root / "shared_baseline"))) / "benchmark_by_scenario.csv"

    rows: list[dict[str, Any]] = []
    for entry in manifest:
        row = dict(entry)
        eval_summary_path = Path(str(entry["eval_output_dir"])) / "eval_summary_by_scenario.json"
        baseline_csv_path = Path(str(entry["eval_output_dir"])) / "baseline_compare" / "benchmark_by_scenario.csv"
        resolved_baseline_csv = baseline_csv_path if baseline_csv_path.exists() else shared_baseline_csv
        if eval_summary_path.exists():
            metrics, _ = _load_eval_metrics(eval_summary_path)
            row.update(metrics)
            row["status"] = "complete" if resolved_baseline_csv.exists() else "missing_baseline_compare"
        else:
            row.update(
                {
                    "max_unsafe": float("inf"),
                    "max_structural_infeasible": float("inf"),
                    "max_timeout": float("inf"),
                    "max_mean_latency": float("inf"),
                    "min_tps": float("-inf"),
                }
            )
            row["status"] = "missing_eval"
        if resolved_baseline_csv.exists():
            row.update(_load_baseline_advantage(resolved_baseline_csv))
        else:
            row.update(
                {
                    "worst_adv_unsafe": float("-inf"),
                    "worst_adv_structural_infeasible": float("-inf"),
                    "worst_adv_timeout": float("-inf"),
                    "worst_adv_mean_latency": float("-inf"),
                    "worst_adv_tps": float("-inf"),
                }
            )
        rows.append(row)

    frame = pd.DataFrame(rows)
    sortable = frame.copy()
    sortable["selection_key"] = sortable.apply(lambda item: _selection_key(item.to_dict()), axis=1)
    sortable = sortable.sort_values("selection_key").drop(columns=["selection_key"])
    sortable.to_csv(output_dir / "search_results_summary.csv", index=False)

    complete_rows = [row for row in sortable.to_dict(orient="records") if row["status"] == "complete"]
    best_row = complete_rows[0] if complete_rows else None

    markdown_lines = [
        "# Governance DQN Search Summary",
        "",
        f"- 搜索根目录: `{search_root}`",
        f"- 已发现运行数: `{len(rows)}`",
        f"- 已完成评估数: `{sum(int(row['status'] == 'complete') for row in rows)}`",
        "",
        "## Top 10",
        "",
        _csv_block(
            sortable[
                [
                    "run_id",
                    "lr",
                    "target_update_period",
                    "warmup_steps",
                    "epsilon_decay_steps",
                    "train_freq",
                    "status",
                    "max_unsafe",
                    "max_structural_infeasible",
                    "max_timeout",
                    "max_mean_latency",
                    "min_tps",
                ]
            ].head(10)
        ),
        "",
    ]
    if best_row is not None:
        write_json(output_dir / "best_config.json", best_row)
        (output_dir / "formal_commands.sh").write_text(_formal_commands(best_row), encoding="utf-8")
        markdown_lines.extend(
            [
                "## Best Config",
                "",
                _csv_block(
                    pd.DataFrame([best_row])[
                        [
                            "run_id",
                            "lr",
                            "target_update_period",
                            "warmup_steps",
                            "epsilon_decay_steps",
                            "train_freq",
                            "max_unsafe",
                            "max_structural_infeasible",
                            "max_timeout",
                            "max_mean_latency",
                            "min_tps",
                            "worst_adv_unsafe",
                            "worst_adv_structural_infeasible",
                            "worst_adv_timeout",
                            "worst_adv_mean_latency",
                            "worst_adv_tps",
                        ]
                    ]
                ),
                "",
            ]
        )
    else:
        markdown_lines.extend(["## Best Config", "", "尚无完整结果，无法确定最优配置。", ""])

    (output_dir / "search_results_summary.md").write_text("\n".join(markdown_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
