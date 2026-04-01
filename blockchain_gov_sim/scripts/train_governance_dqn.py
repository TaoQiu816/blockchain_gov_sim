"""训练治理 DQN。"""

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
from gov_sim.experiments.benchmark_runner import (
    MIXED_TRAINING_CURRICULUM,
    SCENARIO_ORDER,
    build_scenario_config,
    mixed_scenario_curriculum,
)
from gov_sim.modules.metrics_tracker import MetricsTracker
from gov_sim.utils.io import ensure_dir, load_config, write_json
from gov_sim.utils.plotting import save_bar_plot, save_line_plot


def _evaluate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tracker = MetricsTracker()
    for row in rows:
        tracker.update(row)
    summary = tracker.summary()
    summary["timeout_rate"] = float(summary.get("timeout_failure_rate", 0.0))
    summary["committee_average_trust"] = float(summary.get("committee_mean_trust_mean", 0.0))
    return summary


def evaluate_agent(agent: ConstrainedDoubleDuelingDQN, config: dict[str, Any], episodes: int = 2) -> tuple[pd.DataFrame, dict[str, Any]]:
    env = BlockchainGovEnv(config)
    rows: list[dict[str, Any]] = []
    for episode in range(int(episodes)):
        state, legal_mask, _ = env.reset(seed=int(config["seed"]) + 1000 + episode)
        done = False
        step = 0
        while not done:
            action = agent.select_action(state, legal_mask, deterministic=True)
            state, legal_mask, reward, cost, done, info = env.step(action)
            row = info.copy()
            row["episode"] = int(episode)
            row["step"] = int(step)
            row["reward"] = float(reward)
            row["cost"] = float(cost)
            rows.append(row)
            step += 1
    df = pd.DataFrame(rows)
    return df, _evaluate_rows(rows)


def evaluate_agent_by_scenario(
    agent: ConstrainedDoubleDuelingDQN,
    config: dict[str, Any],
    episodes_per_scenario: int = 2,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    per_scenario_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in SCENARIO_ORDER}
    seed_base = int(config["seed"]) + 1000
    seed_cursor = 0
    for scenario_name in SCENARIO_ORDER:
        scenario_cfg = build_scenario_config(config, scenario_name)
        env = BlockchainGovEnv(scenario_cfg)
        for episode in range(int(episodes_per_scenario)):
            state, legal_mask, _ = env.reset(seed=seed_base + seed_cursor)
            seed_cursor += 1
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
                per_scenario_rows[scenario_name].append(row)
                step += 1
    overall_df = pd.DataFrame(rows)
    overall_summary = _evaluate_rows(rows)
    scenario_summaries = {name: _evaluate_rows(per_scenario_rows[name]) for name in SCENARIO_ORDER}
    return overall_df, overall_summary, scenario_summaries


def _best_key(summary: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(summary.get("unsafe_rate", 1.0)),
        float(summary.get("timeout_rate", summary.get("timeout_failure_rate", 1.0))),
        float(summary.get("mean_latency", float("inf"))),
        -float(summary.get("tps", 0.0)),
    )


def _worst_case_key(summaries: dict[str, dict[str, Any]]) -> tuple[float, float, float, float, float]:
    unsafe_values = [float(summary.get("unsafe_rate_all_steps", summary.get("unsafe_rate", 1.0))) for summary in summaries.values()]
    structural_values = [float(summary.get("structural_infeasible_rate_all_steps", summary.get("structural_infeasible_rate", 1.0))) for summary in summaries.values()]
    timeout_values = [float(summary.get("timeout_rate_all_steps", summary.get("timeout_rate", summary.get("timeout_failure_rate", 1.0)))) for summary in summaries.values()]
    latency_values = [float(summary.get("mean_latency", float("inf"))) for summary in summaries.values()]
    tps_values = [float(summary.get("TPS", summary.get("tps", 0.0))) for summary in summaries.values()]
    return (
        max(unsafe_values) if unsafe_values else 1.0,
        max(structural_values) if structural_values else 1.0,
        max(timeout_values) if timeout_values else 1.0,
        max(latency_values) if latency_values else float("inf"),
        -min(tps_values) if tps_values else 0.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--output-dir", default="outputs/governance_dqn")
    parser.add_argument("--num-episodes", type=int, default=800)
    parser.add_argument("--episode-length", type=int, default=120)
    parser.add_argument("--total-env-steps", type=int, default=96000)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed-scenario", action="store_true")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--target-update-period", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--epsilon-decay-steps", type=int, default=None)
    parser.add_argument("--train-freq", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config, *args.override)
    config["seed"] = int(args.seed)
    config.setdefault("env", {})["episode_length"] = int(args.episode_length)
    output_dir = ensure_dir(ROOT / args.output_dir if not Path(args.output_dir).is_absolute() else args.output_dir)

    env = BlockchainGovEnv(config)
    agent = ConstrainedDoubleDuelingDQN(
        state_dim=int(env.observation_space.shape[0]),
        action_dim=int(env.action_space.n),
        device="cpu",
        cost_limit=0.10,
        lr=args.lr,
        target_update_period=args.target_update_period,
        warmup_steps=args.warmup_steps,
        epsilon_decay_steps=args.epsilon_decay_steps,
        train_freq=args.train_freq,
    )
    checkpoint_dir = ensure_dir(output_dir / "checkpoints")
    train_hyperparameters = {
        "lr": float(agent.lr),
        "target_update_period": int(agent.target_update_period),
        "warmup_steps": int(agent.warmup_steps),
        "epsilon_decay_steps": int(agent.epsilon_decay_steps),
        "train_freq": int(agent.train_freq),
    }

    episode_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    eval_rows_by_scenario: list[dict[str, Any]] = []
    schedule_rows: list[dict[str, Any]] = []
    action_counter: Counter[str] = Counter()
    best_summary: dict[str, Any] | None = None
    best_summary_by_scenario: dict[str, dict[str, Any]] | None = None
    best_eval_df = pd.DataFrame()
    total_steps = 0
    if bool(args.mixed_scenario):
        episode_schedule = mixed_scenario_curriculum(int(args.num_episodes), int(args.seed))
    else:
        episode_schedule = [
            {
                "episode": int(episode),
                "stage_index": 0,
                "stage_name": "normal_only",
                "stage_start_episode": 0,
                "stage_end_episode": int(args.num_episodes) - 1,
                "scenario_name": "normal",
                "stage_fraction": 1.0,
                "stage_mix": {"normal": 1.0},
            }
            for episode in range(int(args.num_episodes))
        ]

    for episode in range(int(args.num_episodes)):
        schedule_entry = dict(episode_schedule[episode])
        scenario_name = str(schedule_entry["scenario_name"])
        stage_index = int(schedule_entry["stage_index"])
        stage_name = str(schedule_entry["stage_name"])
        schedule_rows.append(
            {
                "episode": int(schedule_entry["episode"]),
                "stage_index": stage_index,
                "stage_name": stage_name,
                "stage_start_episode": int(schedule_entry["stage_start_episode"]),
                "stage_end_episode": int(schedule_entry["stage_end_episode"]),
                "scenario_name": scenario_name,
            }
        )
        episode_config = build_scenario_config(config, scenario_name) if bool(args.mixed_scenario) else config
        env = BlockchainGovEnv(episode_config)
        state, legal_mask, _ = env.reset(seed=int(args.seed) + episode)
        done = False
        ep_reward = 0.0
        ep_cost = 0.0
        ep_infos: list[dict[str, Any]] = []
        while not done and total_steps < int(args.total_env_steps):
            action = agent.select_action(state, legal_mask, deterministic=False)
            next_state, next_legal_mask, reward, cost, done, info = env.step(action)
            agent.store_transition(state, legal_mask, action, reward, cost, next_state, next_legal_mask, done)
            train_info = None
            if agent.global_step >= agent.warmup_steps and agent.global_step % agent.train_freq == 0:
                train_info = agent.train_step()
            state, legal_mask = next_state, next_legal_mask
            ep_reward += float(reward)
            ep_cost += float(cost)
            total_steps += 1
            action_counter[info["action_distribution_key"]] += 1
            row = {
                "episode": int(episode),
                "global_step": int(total_steps),
                "scenario_name": scenario_name,
                "stage_index": stage_index,
                "stage_name": stage_name,
                "reward": float(reward),
                "cost": float(cost),
                "lambda": float(agent.lambda_value),
                "epsilon": float(agent.epsilon),
                "unsafe": int(info["unsafe"]),
                "timeout": int(info["timeout"]),
                "structural_infeasible": int(info["structural_infeasible"]),
                "committee_mean_trust": float(info["committee_mean_trust"]),
                "qualified_node_count": int(info["qualified_node_count"]),
                "tps": float(info["tps"]),
                "latency": float(info["latency"]),
                "action_key": str(info["action_distribution_key"]),
            }
            if train_info is not None:
                row.update(train_info)
            step_rows.append(row)
            ep_infos.append(info)
            if total_steps >= int(args.total_env_steps):
                break

        if ep_infos:
            episode_rows.append(
                {
                    "episode": int(episode),
                    "stage_index": stage_index,
                    "stage_name": stage_name,
                    "scenario_name": scenario_name,
                    "episode_reward": float(ep_reward),
                    "episode_cost": float(ep_cost),
                    "unsafe_rate": float(np.mean([info["unsafe"] for info in ep_infos])),
                    "timeout_rate": float(np.mean([info["timeout"] for info in ep_infos])),
                    "structural_infeasible_rate": float(np.mean([info["structural_infeasible"] for info in ep_infos])),
                    "committee_mean_trust_mean": float(np.mean([info["committee_mean_trust"] for info in ep_infos])),
                    "eligible_size_mean": float(np.mean([info["eligible_size"] for info in ep_infos])),
                    "tps": float(np.mean([info["tps"] for info in ep_infos])),
                    "latency": float(np.mean([info["latency"] for info in ep_infos])),
                    "lagrangian_lambda": float(agent.lambda_value),
                    "epsilon": float(agent.epsilon),
                }
            )

        if (episode + 1) % int(args.eval_every) == 0 or total_steps >= int(args.total_env_steps):
            eval_df, summary, by_scenario = evaluate_agent_by_scenario(agent, config, episodes_per_scenario=int(args.eval_episodes))
            summary["episode"] = int(episode)
            summary["global_step"] = int(total_steps)
            eval_rows.append(summary)
            for scenario_name, scenario_summary in by_scenario.items():
                row = dict(scenario_summary)
                row["episode"] = int(episode)
                row["global_step"] = int(total_steps)
                row["scenario_name"] = scenario_name
                eval_rows_by_scenario.append(row)
            agent.save(checkpoint_dir / f"episode_{episode + 1:04d}.pt")
            if best_summary_by_scenario is None or _worst_case_key(by_scenario) < _worst_case_key(best_summary_by_scenario):
                best_summary = summary
                best_summary_by_scenario = by_scenario
                best_eval_df = eval_df.copy()
                agent.save(output_dir / "best_checkpoint.pt")
            agent.save(output_dir / "latest_checkpoint.pt")
        if total_steps >= int(args.total_env_steps):
            break

    episode_df = pd.DataFrame(episode_rows)
    step_df = pd.DataFrame(step_rows)
    eval_df = pd.DataFrame(eval_rows)
    eval_df_by_scenario = pd.DataFrame(eval_rows_by_scenario)
    schedule_df = pd.DataFrame(schedule_rows)
    episode_df.to_csv(output_dir / "train_log.csv", index=False)
    step_df.to_csv(output_dir / "train_log_steps.csv", index=False)
    eval_df.to_csv(output_dir / "eval_log.csv", index=False)
    eval_df_by_scenario.to_csv(output_dir / "eval_log_by_scenario.csv", index=False)
    schedule_df.to_csv(output_dir / "episode_schedule.csv", index=False)
    if not best_eval_df.empty:
        best_eval_df.to_csv(output_dir / "best_eval_episode.csv", index=False)

    action_df = pd.DataFrame(
        [{"action": action, "count": int(count), "ratio": float(count / max(sum(action_counter.values()), 1))} for action, count in action_counter.items()]
    ).sort_values("count", ascending=False)
    action_df.to_csv(output_dir / "action_distribution.csv", index=False)

    if not episode_df.empty:
        x = episode_df["episode"].tolist()
        save_line_plot(x, {"reward": episode_df["episode_reward"]}, output_dir / "reward_curve.png", "Reward Curve", "Episode", "Reward")
        save_line_plot(x, {"cost": episode_df["episode_cost"]}, output_dir / "cost_curve.png", "Cost Curve", "Episode", "Cost")
        save_line_plot(x, {"lambda": episode_df["lagrangian_lambda"]}, output_dir / "lambda_curve.png", "Lambda Trajectory", "Episode", "Lambda")
        save_line_plot(x, {"epsilon": episode_df["epsilon"]}, output_dir / "epsilon_curve.png", "Epsilon Trajectory", "Episode", "Epsilon")
        save_line_plot(
            x,
            {
                "unsafe": episode_df["unsafe_rate"],
                "timeout": episode_df["timeout_rate"],
                "structural_infeasible": episode_df["structural_infeasible_rate"],
            },
            output_dir / "safety_breakdown_curve.png",
            "Safety Breakdown",
            "Episode",
            "Rate",
        )
        save_line_plot(
            x,
            {"committee_mean_trust": episode_df["committee_mean_trust_mean"]},
            output_dir / "committee_mean_trust_curve.png",
            "Committee Mean Trust",
            "Episode",
            "Trust",
        )
        save_line_plot(
            x,
            {"qualified_node_count": episode_df["eligible_size_mean"]},
            output_dir / "qualified_node_count_curve.png",
            "Qualified Node Count",
            "Episode",
            "Count",
        )
    if not action_df.empty:
        top_df = action_df.head(15)
        save_bar_plot(top_df["action"].tolist(), top_df["count"].astype(float).tolist(), output_dir / "action_distribution.png", "Action Distribution", "Count")

    summary_payload = {
        "episodes_completed": int(len(episode_df)),
        "total_env_steps": int(total_steps),
        "mixed_scenario": bool(args.mixed_scenario),
        "training_hyperparameters": train_hyperparameters,
        "mixed_training_protocol": {
            "type": "three_stage_curriculum" if bool(args.mixed_scenario) else "normal_only",
            "stages": list(MIXED_TRAINING_CURRICULUM) if bool(args.mixed_scenario) else [],
        },
        "schedule_counts": (
            schedule_df.groupby(["stage_name", "scenario_name"], as_index=False).size().rename(columns={"size": "episodes"}).to_dict(orient="records")
            if not schedule_df.empty
            else []
        ),
        "best_checkpoint": str(output_dir / "best_checkpoint.pt") if best_summary is not None else "",
        "best_eval_summary": best_summary or {},
        "best_eval_summary_by_scenario": best_summary_by_scenario or {},
    }
    write_json(output_dir / "train_summary.json", summary_payload)


if __name__ == "__main__":
    main()
