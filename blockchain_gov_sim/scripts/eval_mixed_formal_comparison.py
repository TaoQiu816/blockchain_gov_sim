"""评估 mixed-scenario 正式对比并汇总七方法五场景结果表。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gov_sim.agent.constrained_dueling_dqn import ConstrainedDoubleDuelingDQN
from gov_sim.env.gov_env import BlockchainGovEnv
from gov_sim.experiments.benchmark_runner import SCENARIO_ORDER, build_scenario_config, run_benchmark_by_scenario
from gov_sim.modules.metrics_tracker import MetricsTracker
from gov_sim.utils.io import ensure_dir, load_config, write_json


LEARNED_METHODS = (
    ("Proposed", "proposed_checkpoint"),
    ("Vanilla DQN", "vanilla_checkpoint"),
    ("No-Constraint Dueling Double DQN", "no_constraint_checkpoint"),
)

BASELINE_METHODS = [
    "Global-Static-Fixed",
    "BTSR-2026",
    "Zhao-2024",
    "Scenario-Oracle-Static",
]

METHOD_ORDER = [
    "Proposed",
    "Vanilla DQN",
    "No-Constraint Dueling Double DQN",
    "Global-Static-Fixed",
    "BTSR-2026",
    "Zhao-2024",
    "Scenario-Oracle-Static",
]


def _metric(summary: dict[str, Any], primary: str, fallback: str, default: float) -> float:
    return float(summary.get(primary, summary.get(fallback, default)))


def _evaluate_checkpoint_by_scenario(
    checkpoint: Path,
    config: dict[str, Any],
    episodes: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    agent = ConstrainedDoubleDuelingDQN.load(checkpoint, device="cpu")
    agent.epsilon = 0.0
    rows: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}

    for scenario_name in SCENARIO_ORDER:
        tracker = MetricsTracker()
        scenario_rows: list[dict[str, Any]] = []
        scenario_cfg = build_scenario_config(config, scenario_name)
        env = BlockchainGovEnv(scenario_cfg)
        for episode in range(int(episodes)):
            state, legal_mask, _ = env.reset(seed=int(seed) + episode)
            done = False
            step = 0
            while not done:
                action = agent.select_action(state, legal_mask, deterministic=True)
                state, legal_mask, reward, cost, done, info = env.step(action)
                row = info.copy()
                row["scenario"] = scenario_name
                row["episode"] = int(episode)
                row["step"] = int(step)
                row["reward"] = float(reward)
                row["cost"] = float(cost)
                scenario_rows.append(row)
                rows.append(row)
                tracker.update(row)
                step += 1
        summary = tracker.summary()
        summary["scenario"] = scenario_name
        summary["unsafe_rate"] = _metric(summary, "unsafe_rate", "unsafe_rate_all_steps", 1.0)
        summary["structural_infeasible_rate"] = _metric(
            summary,
            "structural_infeasible_rate",
            "structural_infeasible_rate_all_steps",
            1.0,
        )
        summary["timeout_rate"] = _metric(summary, "timeout_rate", "timeout_failure_rate", 1.0)
        summary["mean_latency"] = _metric(summary, "mean_latency", "mean_latency", float("inf"))
        summary["TPS"] = _metric(summary, "TPS", "tps", 0.0)
        summaries[scenario_name] = summary

    return pd.DataFrame(rows), summaries


def _summaries_to_rows(method_name: str, summaries: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario_name in SCENARIO_ORDER:
        summary = dict(summaries[scenario_name])
        rows.append(
            {
                "method": method_name,
                "scenario": scenario_name,
                "unsafe_rate": _metric(summary, "unsafe_rate", "unsafe_rate_all_steps", 1.0),
                "structural_infeasible_rate": _metric(
                    summary,
                    "structural_infeasible_rate",
                    "structural_infeasible_rate_all_steps",
                    1.0,
                ),
                "timeout_rate": _metric(summary, "timeout_rate", "timeout_failure_rate", 1.0),
                "mean_latency": _metric(summary, "mean_latency", "mean_latency", float("inf")),
                "TPS": _metric(summary, "TPS", "tps", 0.0),
            }
        )
    return rows


def _baseline_rows(table: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in table.iterrows():
        method_name = str(row["method"])
        rows.append(
            {
                "method": method_name,
                "scenario": str(row["scenario"]),
                "unsafe_rate": _metric(row.to_dict(), "unsafe_rate", "unsafe_rate_all_steps", 1.0),
                "structural_infeasible_rate": _metric(
                    row.to_dict(),
                    "structural_infeasible_rate",
                    "structural_infeasible_rate_all_steps",
                    1.0,
                ),
                "timeout_rate": _metric(row.to_dict(), "timeout_rate", "timeout_failure_rate", 1.0),
                "mean_latency": _metric(row.to_dict(), "mean_latency", "mean_latency", float("inf")),
                "TPS": _metric(row.to_dict(), "TPS", "tps", 0.0),
            }
        )
    return rows


def _markdown_table(frame: pd.DataFrame) -> str:
    return "```text\n" + frame.to_csv(index=False).strip() + "\n```"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--proposed-checkpoint", required=True)
    parser.add_argument("--vanilla-checkpoint", required=True)
    parser.add_argument("--no-constraint-checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/governance_dqn_formal/part_b_comparison_seed42")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config, *args.override)
    config["seed"] = int(args.seed)
    output_dir = ensure_dir(ROOT / args.output_dir if not Path(args.output_dir).is_absolute() else args.output_dir)

    combined_rows: list[dict[str, Any]] = []
    learned_payload: dict[str, Any] = {}
    for method_name, attr_name in LEARNED_METHODS:
        checkpoint = ROOT / getattr(args, attr_name) if not Path(getattr(args, attr_name)).is_absolute() else Path(getattr(args, attr_name))
        step_df, summaries = _evaluate_checkpoint_by_scenario(
            checkpoint=checkpoint,
            config=config,
            episodes=int(args.episodes),
            seed=int(args.seed),
        )
        step_df.to_csv(output_dir / f"{method_name.lower().replace(' ', '_').replace('-', '_')}_steps.csv", index=False)
        learned_payload[method_name] = summaries
        combined_rows.extend(_summaries_to_rows(method_name, summaries))

    baseline_result = run_benchmark_by_scenario(
        config=config,
        checkpoint_path=None,
        methods=BASELINE_METHODS,
        episodes=int(args.episodes),
        output_dir=output_dir / "baseline_compare",
        scenarios=SCENARIO_ORDER,
    )
    combined_rows.extend(_baseline_rows(baseline_result["table"]))

    result_df = pd.DataFrame(combined_rows)
    result_df["method"] = pd.Categorical(result_df["method"], categories=METHOD_ORDER, ordered=True)
    result_df["scenario"] = pd.Categorical(result_df["scenario"], categories=SCENARIO_ORDER, ordered=True)
    result_df = result_df.sort_values(["scenario", "method"]).reset_index(drop=True)
    result_df.to_csv(output_dir / "comparison_by_scenario.csv", index=False)

    markdown_lines = [
        "# Mixed Formal Comparison",
        "",
        f"- episodes_per_scenario: `{int(args.episodes)}`",
        f"- seed: `{int(args.seed)}`",
        "",
        "## Five-Scenario Table",
        "",
        _markdown_table(result_df),
        "",
    ]
    (output_dir / "comparison_by_scenario.md").write_text("\n".join(markdown_lines), encoding="utf-8")
    write_json(
        output_dir / "comparison_metadata.json",
        {
            "episodes_per_scenario": int(args.episodes),
            "seed": int(args.seed),
            "proposed_checkpoint": str(args.proposed_checkpoint),
            "vanilla_checkpoint": str(args.vanilla_checkpoint),
            "no_constraint_checkpoint": str(args.no_constraint_checkpoint),
            "learned_methods": learned_payload,
        },
    )


if __name__ == "__main__":
    main()
