"""评估治理 DQN。"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gov_sim.agent.constrained_dueling_dqn import ConstrainedDoubleDuelingDQN
from gov_sim.env.gov_env import BlockchainGovEnv
from gov_sim.experiments.benchmark_runner import SCENARIO_ORDER, build_scenario_config, run_benchmark_by_scenario
from gov_sim.modules.metrics_tracker import MetricsTracker
from gov_sim.utils.io import ensure_dir, load_config, write_json


def _action_entropy(actions: list[str]) -> float:
    if not actions:
        return 0.0
    counts = np.asarray(list(Counter(actions).values()), dtype=np.float64)
    probs = counts / max(np.sum(counts), 1.0)
    return float(-np.sum(probs * np.log(probs + 1.0e-12)))


def _action_switch_count_per_episode(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    counts: list[int] = []
    for _, ep_df in df.groupby("episode"):
        actions = ep_df["executed_action"].astype(str).tolist()
        counts.append(sum(int(actions[idx] != actions[idx - 1]) for idx in range(1, len(actions))))
    return float(np.mean(counts)) if counts else 0.0


def _top_k_true_actions(df: pd.DataFrame, k: int = 10) -> list[dict[str, Any]]:
    if df.empty:
        return []
    counter = df["executed_action"].astype(str).value_counts()
    total = int(counter.sum())
    return [
        {"action": str(action), "count": int(count), "ratio": float(count / max(total, 1))}
        for action, count in counter.head(int(k)).items()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/governance_dqn_eval")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compare-baselines", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config, *args.override)
    config["seed"] = int(args.seed)
    output_dir = ensure_dir(ROOT / args.output_dir if not Path(args.output_dir).is_absolute() else args.output_dir)
    checkpoint = ROOT / args.checkpoint if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint)
    agent = ConstrainedDoubleDuelingDQN.load(checkpoint, device="cpu")
    agent.epsilon = 0.0
    all_rows: list[dict[str, Any]] = []
    scenario_summaries: dict[str, dict[str, Any]] = {}

    for scenario_name in SCENARIO_ORDER:
        scenario_cfg = build_scenario_config(config, scenario_name)
        env = BlockchainGovEnv(scenario_cfg)
        tracker = MetricsTracker()
        rows: list[dict[str, Any]] = []
        for episode in range(int(args.episodes)):
            state, legal_mask, _ = env.reset(seed=int(args.seed) + episode)
            done = False
            step = 0
            while not done:
                action = agent.select_action(state, legal_mask, deterministic=True)
                state, legal_mask, reward, cost, done, info = env.step(action)
                row = info.copy()
                row["scenario_name"] = scenario_name
                row["episode"] = int(episode)
                row["step"] = int(step)
                row["reward"] = float(reward)
                row["cost"] = float(cost)
                rows.append(row)
                all_rows.append(row)
                tracker.update(row)
                step += 1
        df = pd.DataFrame(rows)
        summary = tracker.summary()
        summary["timeout_rate"] = float(summary.get("timeout_failure_rate", 0.0))
        summary["qualified_node_count_mean"] = float(summary.get("eligible_size_mean", 0.0))
        summary["committee_average_trust"] = float(summary.get("committee_mean_trust_mean", 0.0))
        summary["top_k_true_action_distribution"] = _top_k_true_actions(df)
        summary["action_entropy"] = _action_entropy(df["executed_action"].astype(str).tolist())
        summary["action_switch_count_per_episode"] = _action_switch_count_per_episode(df)
        scenario_summaries[scenario_name] = summary
        df.to_csv(output_dir / f"eval_log_{scenario_name}.csv", index=False)

    all_df = pd.DataFrame(all_rows)
    all_df.to_csv(output_dir / "eval_log.csv", index=False)
    write_json(output_dir / "eval_summary_by_scenario.json", scenario_summaries)

    if bool(args.compare_baselines):
        benchmark = run_benchmark_by_scenario(
            config=config,
            checkpoint_path=checkpoint,
            methods=["DQN", "Global-Static-Fixed", "Scenario-Oracle-Static", "BTSR-2026", "Zhao-2024"],
            episodes=int(args.episodes),
            output_dir=output_dir / "baseline_compare",
            scenarios=SCENARIO_ORDER,
        )
        write_json(output_dir / "baseline_compare_summary.json", {"rows": benchmark["rows"]})


if __name__ == "__main__":
    main()
