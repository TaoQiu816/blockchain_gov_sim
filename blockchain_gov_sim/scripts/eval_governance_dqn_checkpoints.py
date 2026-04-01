"""批量评估治理 DQN checkpoints 的分场景趋势。"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
import sys
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gov_sim.agent.constrained_dueling_dqn import ConstrainedDoubleDuelingDQN
from gov_sim.env.gov_env import BlockchainGovEnv
from gov_sim.experiments.benchmark_runner import SCENARIO_ORDER, build_scenario_config
from gov_sim.modules.metrics_tracker import MetricsTracker
from gov_sim.utils.io import ensure_dir, load_config, write_json


def _extract_checkpoint_step(path: Path) -> int:
    matched = re.search(r"(\d+)", path.stem)
    return int(matched.group(1)) if matched else -1


def _evaluate_checkpoint(
    checkpoint: Path,
    config: dict[str, Any],
    episodes: int,
    seed: int,
) -> list[dict[str, Any]]:
    agent = ConstrainedDoubleDuelingDQN.load(checkpoint, device="cpu")
    agent.epsilon = 0.0
    rows: list[dict[str, Any]] = []
    for scenario_name in SCENARIO_ORDER:
        scenario_cfg = build_scenario_config(config, scenario_name)
        env = BlockchainGovEnv(scenario_cfg)
        tracker = MetricsTracker()
        for episode in range(int(episodes)):
            state, legal_mask, _ = env.reset(seed=int(seed) + episode)
            done = False
            while not done:
                action = agent.select_action(state, legal_mask, deterministic=True)
                state, legal_mask, _, _, done, info = env.step(action)
                tracker.update(info)
        summary = tracker.summary()
        rows.append(
            {
                "checkpoint": checkpoint.name,
                "checkpoint_order": int(_extract_checkpoint_step(checkpoint)),
                "scenario_name": scenario_name,
                "unsafe_rate": float(summary.get("unsafe_rate_all_steps", summary.get("unsafe_rate", 1.0))),
                "structural_infeasible_rate": float(
                    summary.get("structural_infeasible_rate_all_steps", summary.get("structural_infeasible_rate", 1.0))
                ),
                "timeout_rate": float(summary.get("timeout_rate_all_steps", summary.get("timeout_failure_rate", 1.0))),
                "mean_latency": float(summary.get("mean_latency", float("inf"))),
                "TPS": float(summary.get("TPS", summary.get("tps", 0.0))),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/governance_dqn_checkpoint_eval")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config, *args.override)
    config["seed"] = int(args.seed)
    checkpoint_dir = ROOT / args.checkpoint_dir if not Path(args.checkpoint_dir).is_absolute() else Path(args.checkpoint_dir)
    output_dir = ensure_dir(ROOT / args.output_dir if not Path(args.output_dir).is_absolute() else args.output_dir)
    checkpoints = sorted(checkpoint_dir.glob("*.pt"), key=_extract_checkpoint_step)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    rows: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        rows.extend(_evaluate_checkpoint(checkpoint=checkpoint, config=config, episodes=int(args.episodes), seed=int(args.seed)))

    frame = pd.DataFrame(rows).sort_values(["checkpoint_order", "scenario_name"])
    frame.to_csv(output_dir / "checkpoint_trend_by_scenario.csv", index=False)
    write_json(
        output_dir / "checkpoint_trend_manifest.json",
        {
            "checkpoint_dir": str(checkpoint_dir),
            "episodes": int(args.episodes),
            "seed": int(args.seed),
            "num_checkpoints": int(len(checkpoints)),
        },
    )


if __name__ == "__main__":
    main()
