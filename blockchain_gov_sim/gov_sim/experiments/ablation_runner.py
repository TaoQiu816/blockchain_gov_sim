"""消融实验 runner。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from gov_sim.experiments.benchmark_runner import evaluate_method
from gov_sim.utils.io import ensure_dir, load_config, write_json


def _no_tof_hook(env: Any) -> None:
    env.reputation_model.delta_r = 0.0
    env.reputation_model.delta_plus = 0.0


def _no_safety_cost_hook(controller: Any, method: str) -> None:
    if method == "DQN":
        controller.lambda_value = 0.0


def run_ablation(
    config: dict[str, Any],
    checkpoint_path: str | Path | None,
    method: str = "DQN",
    episodes: int | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    output = ensure_dir(output_dir or Path(config.get("output_root", "outputs")) / "ablation" / str(config.get("run_name", "default_run")))
    rows: list[dict[str, Any]] = []

    df, summary = evaluate_method(
        method=method,
        config=config,
        checkpoint_path=checkpoint_path,
        episodes=episodes,
        deterministic=True,
        env_hook=_no_tof_hook,
    )
    summary["ablation"] = "no_tof"
    rows.append(summary)
    df.to_csv(output / "no_tof_steps.csv", index=False)

    if method == "DQN":
        from gov_sim.agent.constrained_dueling_dqn import ConstrainedDoubleDuelingDQN

        controller = ConstrainedDoubleDuelingDQN.load(checkpoint_path, device="cpu")
        controller.epsilon = 0.0
        _no_safety_cost_hook(controller, method)
        from gov_sim.experiments.benchmark_runner import evaluate_controller

        df2, summary2 = evaluate_controller(
            controller=controller,
            config=config,
            episodes=int(episodes if episodes is not None else config.get("eval", {}).get("episodes", 1)),
            deterministic=True,
            is_baseline=False,
            env_hook=None,
        )
        summary2["ablation"] = "no_safety_cost"
        rows.append(summary2)
        df2.to_csv(output / "no_safety_cost_steps.csv", index=False)

    table = pd.DataFrame(rows)
    table.to_csv(output / "ablation_table.csv", index=False)
    write_json(output / "ablation_summary.json", {"rows": rows})
    return {"output_dir": str(output), "rows": rows}


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--method", default="DQN")
    args = parser.parse_args()

    config = load_config(args.config)
    run_ablation(
        config=config,
        checkpoint_path=args.checkpoint,
        method=args.method,
        episodes=int(args.episodes),
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    _main()
