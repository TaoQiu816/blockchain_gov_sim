"""统一 benchmark runner。"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from gov_sim.agent.constrained_dueling_dqn import ConstrainedDoubleDuelingDQN
from gov_sim.baselines import BaselinePolicy, instantiate_baseline
from gov_sim.constants import ACTION_DIM
from gov_sim.env.gov_env import BlockchainGovEnv
from gov_sim.modules.metrics_tracker import MetricsTracker
from gov_sim.utils.io import deep_update, ensure_dir, load_config, write_json


DEFAULT_METHODS = [
    "DQN",
    "Global-Static-Fixed",
    "Scenario-Oracle-Static",
    "BTSR-2026",
    "Zhao-2024",
    "Yadav-2025",
    "DVRC-2025",
]

SCENARIO_MIX = {
    "normal": 0.15,
    "load_shock": 0.20,
    "high_rtt_burst": 0.20,
    "churn_burst": 0.25,
    "malicious_burst": 0.20,
}

SCENARIO_ORDER = list(SCENARIO_MIX.keys())

MIXED_TRAINING_CURRICULUM = (
    {
        "stage_index": 1,
        "stage_name": "stage1",
        "fraction": 0.25,
        "scenario_mix": {
            "normal": 0.55,
            "load_shock": 0.20,
            "high_rtt_burst": 0.10,
            "churn_burst": 0.10,
            "malicious_burst": 0.05,
        },
    },
    {
        "stage_index": 2,
        "stage_name": "stage2",
        "fraction": 0.35,
        "scenario_mix": {
            "normal": 0.25,
            "load_shock": 0.20,
            "high_rtt_burst": 0.20,
            "churn_burst": 0.20,
            "malicious_burst": 0.15,
        },
    },
    {
        "stage_index": 3,
        "stage_name": "stage3",
        "fraction": 0.40,
        "scenario_mix": {
            "normal": 0.10,
            "load_shock": 0.15,
            "high_rtt_burst": 0.25,
            "churn_burst": 0.25,
            "malicious_burst": 0.25,
        },
    },
)


def build_scenario_config(config: dict[str, Any], scenario_name: str) -> dict[str, Any]:
    cfg = copy.deepcopy(config)
    scenario_cfg = cfg.setdefault("scenario", {})
    training_mix = scenario_cfg.setdefault("training_mix", {})
    profiles = copy.deepcopy(training_mix.get("profiles", {}))
    if scenario_name == "normal":
        scenario_cfg["default_name"] = "stable"
        training_mix["enabled"] = False
        training_mix["profiles"] = profiles
        return cfg
    if scenario_name not in profiles:
        raise ValueError(f"Unknown scenario profile: {scenario_name}")
    training_mix["enabled"] = True
    training_mix["profiles"] = {scenario_name: {**profiles[scenario_name], "weight": 1.0}}
    scenario_cfg["default_name"] = "stable"
    return cfg


def mixed_scenario_schedule(num_episodes: int, seed: int) -> list[str]:
    rng = np.random.default_rng(int(seed))
    names = list(SCENARIO_MIX.keys())
    weights = np.asarray([SCENARIO_MIX[name] for name in names], dtype=np.float64)
    weights = weights / np.sum(weights)
    return [str(rng.choice(names, p=weights)) for _ in range(int(num_episodes))]


def mixed_scenario_curriculum(num_episodes: int, seed: int) -> list[dict[str, Any]]:
    total_episodes = max(int(num_episodes), 0)
    if total_episodes == 0:
        return []

    rng = np.random.default_rng(int(seed))
    raw_counts = [float(stage["fraction"]) * total_episodes for stage in MIXED_TRAINING_CURRICULUM]
    counts = [int(np.floor(value)) for value in raw_counts]
    residual = total_episodes - sum(counts)
    remainders = sorted(
        ((raw_counts[idx] - counts[idx], idx) for idx in range(len(raw_counts))),
        key=lambda item: (-item[0], item[1]),
    )
    for _, idx in remainders[:residual]:
        counts[idx] += 1

    schedule: list[dict[str, Any]] = []
    episode_cursor = 0
    for stage, stage_count in zip(MIXED_TRAINING_CURRICULUM, counts):
        stage_mix = dict(stage["scenario_mix"])
        names = list(stage_mix.keys())
        weights = np.asarray([float(stage_mix[name]) for name in names], dtype=np.float64)
        weights = weights / max(np.sum(weights), 1.0)
        stage_start = episode_cursor
        for _ in range(stage_count):
            scenario_name = str(rng.choice(names, p=weights))
            schedule.append(
                {
                    "episode": int(episode_cursor),
                    "stage_index": int(stage["stage_index"]),
                    "stage_name": str(stage["stage_name"]),
                    "stage_start_episode": int(stage_start),
                    "stage_end_episode": int(stage_start + stage_count - 1),
                    "scenario_name": scenario_name,
                    "stage_fraction": float(stage["fraction"]),
                    "stage_mix": dict(stage_mix),
                }
            )
            episode_cursor += 1
    return schedule


def action_entropy(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    counts = pd.Series([str(row.get("executed_action", row.get("action_tuple", ""))) for row in rows]).value_counts().to_numpy(dtype=np.float64)
    probs = counts / max(np.sum(counts), 1.0)
    return float(-np.sum(probs * np.log(probs + 1.0e-12)))


def action_switch_count_per_episode(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    df = pd.DataFrame(rows)
    counts: list[int] = []
    for _, ep_df in df.groupby("episode"):
        actions = ep_df["executed_action"].astype(str).tolist()
        switches = sum(int(actions[idx] != actions[idx - 1]) for idx in range(1, len(actions)))
        counts.append(switches)
    return float(np.mean(counts)) if counts else 0.0


def top_k_true_actions(rows: list[dict[str, Any]], k: int = 10) -> list[dict[str, Any]]:
    if not rows:
        return []
    counter = pd.Series([str(row.get("executed_action", row.get("action_tuple", ""))) for row in rows]).value_counts()
    total = int(counter.sum())
    output: list[dict[str, Any]] = []
    for action, count in counter.head(int(k)).items():
        output.append({"action": str(action), "count": int(count), "ratio": float(count / max(total, 1))})
    return output


def _resolve_output_dir(output_dir: str | Path | None, stage: str, config: dict[str, Any]) -> Path:
    if output_dir is not None:
        return ensure_dir(output_dir)
    return ensure_dir(Path(config.get("output_root", "outputs")) / stage / str(config.get("run_name", "default_run")))


def _load_dqn(checkpoint_path: str | Path, device: str = "cpu") -> ConstrainedDoubleDuelingDQN:
    return ConstrainedDoubleDuelingDQN.load(checkpoint_path, device=device)


def _scenario_weight(scenario_name: str) -> float:
    return float(SCENARIO_MIX.get(str(scenario_name), 0.0))


def _summary_sort_key(summary: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(summary.get("unsafe_rate", summary.get("unsafe_rate_all_steps", 1.0))),
        float(summary.get("timeout_rate", summary.get("timeout_rate_all_steps", 1.0))),
        float(summary.get("mean_latency", float("inf"))),
        -float(summary.get("tps", summary.get("TPS", 0.0))),
    )


def _global_static_scan(config: dict[str, Any], episodes: int) -> dict[str, Any]:
    controller = instantiate_baseline("Static-Best-Fixed", config)
    scan_episodes = int(config.get("baselines", {}).get("static_scan_episodes", 1))
    best_row: dict[str, Any] | None = None
    weighted_rows: list[dict[str, Any]] = []
    for action_idx in range(int(ACTION_DIM)):
        aggregate = {
            "unsafe_rate": 0.0,
            "timeout_rate": 0.0,
            "mean_latency": 0.0,
            "tps": 0.0,
            "mean_reward": 0.0,
            "mean_cost": 0.0,
        }
        per_scenario: dict[str, dict[str, Any]] = {}
        for scenario_name in SCENARIO_ORDER:
            scenario_cfg = build_scenario_config(config, scenario_name)
            tracker = MetricsTracker()
            rewards: list[float] = []
            costs: list[float] = []
            base_seed = int(config.get("seed", 42)) + 70000 + action_idx * 1000
            for episode in range(int(scan_episodes)):
                env = BlockchainGovEnv(scenario_cfg)
                _, _, _ = env.reset(seed=base_seed + episode)
                done = False
                while not done:
                    _, _, reward, cost, done, info = env.step(action_idx)
                    rewards.append(float(reward))
                    costs.append(float(cost))
                    tracker.update(info)
            summary = tracker.summary()
            scenario_row = {
                "unsafe_rate": float(summary.get("unsafe_rate", summary.get("unsafe_rate_all_steps", 1.0))),
                "timeout_rate": float(summary.get("timeout_rate", summary.get("timeout_failure_rate", summary.get("timeout_rate_all_steps", 1.0)))),
                "mean_latency": float(summary.get("mean_latency", float("inf"))),
                "tps": float(summary.get("tps", summary.get("TPS", 0.0))),
                "mean_reward": float(np.mean(rewards)) if rewards else float("-inf"),
                "mean_cost": float(np.mean(costs)) if costs else float("inf"),
            }
            per_scenario[scenario_name] = scenario_row
            weight = _scenario_weight(scenario_name)
            for key in aggregate:
                aggregate[key] += weight * float(scenario_row[key])
        row = {
            "action_idx": int(action_idx),
            "action_tuple": (
                float(controller.codec.decode(action_idx).rho_m),
                float(controller.codec.decode(action_idx).theta),
                int(controller.codec.decode(action_idx).b),
                int(controller.codec.decode(action_idx).tau),
            ),
            "per_scenario": per_scenario,
            **aggregate,
        }
        weighted_rows.append(row)
        if best_row is None or controller._score_key(row) < controller._score_key(best_row):
            best_row = row
    if best_row is None:
        raise RuntimeError("Failed to find global static fixed action.")
    return {"best_row": best_row, "scan_rows": weighted_rows}


def _evaluate_fixed_action(
    config: dict[str, Any],
    action_idx: int,
    episodes: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    tracker = MetricsTracker()
    rows: list[dict[str, Any]] = []
    env = BlockchainGovEnv(config)
    for episode in range(int(episodes)):
        _, _, _ = env.reset(seed=int(config["seed"]) + episode)
        done = False
        step = 0
        while not done:
            _, _, reward, cost, done, info = env.step(int(action_idx))
            row = info.copy()
            row["episode"] = int(episode)
            row["step"] = int(step)
            row["reward"] = float(reward)
            row["cost"] = float(cost)
            rows.append(row)
            tracker.update(row)
            step += 1
    df = pd.DataFrame(rows)
    summary = tracker.summary()
    summary["timeout_rate"] = float(summary.get("timeout_failure_rate", 0.0))
    summary["qualified_node_count"] = float(summary.get("eligible_size_mean", 0.0))
    summary["committee_average_trust"] = float(summary.get("committee_mean_trust_mean", 0.0))
    summary["action_distribution_stability"] = float(summary.get("dominant_action_ratio", 0.0))
    summary["mean_reward"] = float(df["reward"].mean()) if not df.empty else 0.0
    summary["mean_cost"] = float(df["cost"].mean()) if not df.empty else 0.0
    summary["top_k_true_action_distribution"] = top_k_true_actions(rows)
    summary["action_entropy"] = action_entropy(rows)
    summary["action_switch_count_per_episode"] = action_switch_count_per_episode(rows)
    return df, summary


def _controller_action(
    controller: Any,
    is_baseline: bool,
    env: BlockchainGovEnv,
    obs: Any,
    legal_mask: Any,
    deterministic: bool,
) -> int:
    if is_baseline:
        return int(controller.select_action(env, obs, legal_mask))
    return int(controller.select_action(obs, legal_mask, deterministic=deterministic))


def evaluate_controller(
    controller: Any,
    config: dict[str, Any],
    episodes: int,
    deterministic: bool,
    is_baseline: bool,
    env_hook: Callable[[BlockchainGovEnv], None] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    env = BlockchainGovEnv(config)
    if env_hook is not None:
        env_hook(env)
    tracker = MetricsTracker()
    rows: list[dict[str, Any]] = []
    if is_baseline and hasattr(controller, "reset"):
        controller.reset()
    for episode in range(int(episodes)):
        obs, legal_mask, _ = env.reset(seed=int(config["seed"]) + episode)
        if env_hook is not None:
            env_hook(env)
        done = False
        step = 0
        while not done:
            action_idx = _controller_action(controller, is_baseline, env, obs, legal_mask, deterministic)
            obs, legal_mask, reward, cost, done, info = env.step(action_idx)
            row = info.copy()
            row["episode"] = int(episode)
            row["step"] = int(step)
            row["reward"] = float(reward)
            row["cost"] = float(cost)
            rows.append(row)
            tracker.update(row)
            step += 1
    df = pd.DataFrame(rows)
    summary = tracker.summary()
    summary["timeout_rate"] = float(summary.get("timeout_failure_rate", 0.0))
    summary["qualified_node_count"] = float(summary.get("eligible_size_mean", 0.0))
    summary["committee_average_trust"] = float(summary.get("committee_mean_trust_mean", 0.0))
    summary["action_distribution_stability"] = float(summary.get("dominant_action_ratio", 0.0))
    summary["mean_reward"] = float(df["reward"].mean()) if not df.empty else 0.0
    summary["mean_cost"] = float(df["cost"].mean()) if not df.empty else 0.0
    summary["top_k_true_action_distribution"] = top_k_true_actions(rows)
    summary["action_entropy"] = action_entropy(rows)
    summary["action_switch_count_per_episode"] = action_switch_count_per_episode(rows)
    return df, summary


def evaluate_method(
    method: str,
    config: dict[str, Any],
    checkpoint_path: str | Path | None,
    episodes: int | None = None,
    deterministic: bool = True,
    env_hook: Callable[[BlockchainGovEnv], None] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    eval_episodes = int(episodes if episodes is not None else config.get("eval", {}).get("episodes", 1))
    if method == "DQN":
        if checkpoint_path is None:
            raise ValueError("checkpoint_path is required for DQN evaluation")
        controller = _load_dqn(checkpoint_path)
        controller.epsilon = 0.0
        is_baseline = False
    elif method == "Scenario-Oracle-Static":
        controller = instantiate_baseline("Static-Best-Fixed", config)
        is_baseline = True
    else:
        controller = instantiate_baseline(method, config)
        is_baseline = True
    return evaluate_controller(
        controller=controller,
        config=config,
        episodes=eval_episodes,
        deterministic=bool(deterministic),
        is_baseline=is_baseline,
        env_hook=env_hook,
    )


def run_benchmark(
    config: dict[str, Any],
    checkpoint_path: str | Path | None,
    methods: list[str] | None = None,
    episodes: int | None = None,
    output_dir: str | Path | None = None,
    env_hook: Callable[[BlockchainGovEnv], None] | None = None,
) -> dict[str, Any]:
    methods = list(DEFAULT_METHODS if methods is None else methods)
    output = _resolve_output_dir(output_dir, "benchmark", config)
    rows: list[dict[str, Any]] = []
    raw_logs: dict[str, pd.DataFrame] = {}
    for method in methods:
        df, summary = evaluate_method(
            method=method,
            config=config,
            checkpoint_path=checkpoint_path,
            episodes=episodes,
            deterministic=True,
            env_hook=env_hook,
        )
        summary["method"] = method
        rows.append(summary)
        raw_logs[method] = df
        df.to_csv(output / f"{method.lower().replace('-', '_')}_steps.csv", index=False)
    table = pd.DataFrame(rows).sort_values(["unsafe_rate", "timeout_rate", "mean_latency", "tps"], ascending=[True, True, True, False])
    table.to_csv(output / "benchmark_table.csv", index=False)
    write_json(output / "benchmark_summary.json", {"rows": rows})
    return {"output_dir": str(output), "rows": rows, "table": table}


def run_benchmark_by_scenario(
    config: dict[str, Any],
    checkpoint_path: str | Path | None,
    methods: list[str] | None = None,
    episodes: int | None = None,
    output_dir: str | Path | None = None,
    scenarios: list[str] | None = None,
) -> dict[str, Any]:
    methods = list(DEFAULT_METHODS if methods is None else methods)
    scenarios = list(SCENARIO_ORDER if scenarios is None else scenarios)
    output = _resolve_output_dir(output_dir, "benchmark", config)
    rows: list[dict[str, Any]] = []
    global_static_cache: dict[str, Any] | None = None
    if "Global-Static-Fixed" in methods:
        global_static_cache = _global_static_scan(config, int(episodes if episodes is not None else config.get("eval", {}).get("episodes", 1)))
        pd.DataFrame(global_static_cache["scan_rows"]).drop(columns=["per_scenario"]).to_csv(output / "global_static_fixed_scan.csv", index=False)
    for scenario_name in scenarios:
        scenario_cfg = build_scenario_config(config, scenario_name)
        for method in methods:
            if method == "Global-Static-Fixed":
                if global_static_cache is None:
                    raise RuntimeError("Global static cache missing.")
                chosen_idx = int(global_static_cache["best_row"]["action_idx"])
                df, summary = _evaluate_fixed_action(
                    config=scenario_cfg,
                    action_idx=chosen_idx,
                    episodes=int(episodes if episodes is not None else config.get("eval", {}).get("episodes", 1)),
                )
                summary["global_fixed_action_idx"] = chosen_idx
                summary["global_fixed_action_tuple"] = global_static_cache["best_row"]["action_tuple"]
            else:
                df, summary = evaluate_method(
                    method=method,
                    config=scenario_cfg,
                    checkpoint_path=checkpoint_path,
                    episodes=episodes,
                    deterministic=True,
                )
            summary["method"] = method
            summary["scenario"] = scenario_name
            rows.append(summary)
            df.to_csv(output / f"{scenario_name}_{method.lower().replace('-', '_')}_steps.csv", index=False)
    table = pd.DataFrame(rows).sort_values(["scenario", "unsafe_rate", "timeout_rate", "mean_latency", "tps"], ascending=[True, True, True, True, False])
    table.to_csv(output / "benchmark_by_scenario.csv", index=False)
    write_json(output / "benchmark_by_scenario_summary.json", {"rows": rows})
    return {"output_dir": str(output), "rows": rows, "table": table}


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--by-scenario", action="store_true")
    parser.add_argument("--scenarios", nargs="*", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if bool(args.by_scenario):
        run_benchmark_by_scenario(
            config=config,
            checkpoint_path=args.checkpoint,
            methods=args.methods,
            episodes=int(args.episodes),
            output_dir=args.output_dir,
            scenarios=args.scenarios,
        )
    else:
        run_benchmark(
            config=config,
            checkpoint_path=args.checkpoint,
            methods=args.methods,
            episodes=int(args.episodes),
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    _main()
