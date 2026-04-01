#!/usr/bin/env python3
"""第四章链侧分层动态治理理论一致性审计。"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gov_sim.env.action_codec import ActionCodec, GovernanceAction
from gov_sim.experiments import make_env
from gov_sim.utils.io import ensure_dir, load_config, write_json


CORE_METRICS = [
    "tps",
    "L_bar_e",
    "unsafe",
    "timeout_failure",
    "structural_infeasible",
    "eligible_size",
    "h_LCB_e",
    "committee_honest_ratio",
    "B_e",
    "S_e",
    "L_queue_e",
    "L_batch_e",
    "L_cons_e",
]


def build_scene_config(base_config: dict[str, Any], scene: str) -> dict[str, Any]:
    if scene == "stable":
        config = copy.deepcopy(base_config)
        config["scenario"]["training_mix"]["enabled"] = False
        return config
    config = copy.deepcopy(base_config)
    profile = copy.deepcopy(config["scenario"]["training_mix"]["profiles"][scene])
    profile["weight"] = 1.0
    config["scenario"]["training_mix"] = {
        "enabled": True,
        "enabled_in_train": False,
        "profiles": {scene: profile},
    }
    return config


def summarize_episode(step_rows: list[dict[str, float]]) -> dict[str, float]:
    frame = pd.DataFrame(step_rows)
    return {
        "mean_tps": float(frame["tps"].mean()),
        "mean_latency": float(frame["L_bar_e"].mean()),
        "unsafe_rate": float(frame["unsafe"].mean()),
        "timeout_rate": float(frame["timeout_failure"].mean()),
        "structural_infeasible_rate": float(frame["structural_infeasible"].mean()),
        "eligible_size": float(frame["eligible_size"].mean()),
        "h_LCB": float(frame["h_LCB_e"].mean()),
        "committee_honest_ratio": float(frame["committee_honest_ratio"].mean()),
        "B": float(frame["B_e"].mean()),
        "S": float(frame["S_e"].mean()),
        "L_queue": float(frame["L_queue_e"].mean()),
        "L_batch": float(frame["L_batch_e"].mean()),
        "L_cons": float(frame["L_cons_e"].mean()),
    }


def run_fixed_action(config: dict[str, Any], action: GovernanceAction, scene: str, seeds: int, episodes: int) -> dict[str, Any]:
    codec = ActionCodec()
    action_idx = codec.encode(action)
    scenario_config = build_scene_config(config, scene=scene)
    env = make_env(scenario_config)
    rows: list[dict[str, float]] = []
    for seed_idx in range(seeds):
        for episode_idx in range(episodes):
            eval_seed = 12000 + seed_idx * 100 + episode_idx
            _, _ = env.reset(seed=eval_seed)
            done = False
            truncated = False
            step_rows: list[dict[str, float]] = []
            while not (done or truncated):
                _, _, done, truncated, info = env.step(action_idx)
                step_rows.append({metric: float(info.get(metric, 0.0)) for metric in CORE_METRICS})
            rows.append(summarize_episode(step_rows))
    env.close()
    frame = pd.DataFrame(rows)
    summary = {}
    for column in frame.columns:
        summary[f"{column}_mean"] = float(frame[column].mean())
        summary[f"{column}_std"] = float(frame[column].std(ddof=0))
    return summary


def flatten_record(base: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    row = dict(base)
    row.update(summary)
    return row


def audit_m_scan(config: dict[str, Any], seeds: int, episodes: int) -> pd.DataFrame:
    rows = []
    for m in (5, 7, 9):
        action = GovernanceAction(m=m, b=384, tau=60, theta=0.55)
        summary = run_fixed_action(config, action=action, scene="churn_burst", seeds=seeds, episodes=episodes)
        rows.append(flatten_record({"audit": "m_scan_churn_burst", "scene": "churn_burst", "m": m}, summary))
    return pd.DataFrame(rows)


def audit_theta_grid(config: dict[str, Any], seeds: int, episodes: int) -> pd.DataFrame:
    rows = []
    for m in (5, 7, 9):
        for theta in (0.45, 0.50, 0.55, 0.60):
            action = GovernanceAction(m=m, b=384, tau=60, theta=theta)
            summary = run_fixed_action(config, action=action, scene="malicious_burst", seeds=seeds, episodes=episodes)
            rows.append(
                flatten_record(
                    {"audit": "m_theta_grid_malicious_burst", "scene": "malicious_burst", "m": m, "theta": theta},
                    summary,
                )
            )
    return pd.DataFrame(rows)


def audit_b_scan(config: dict[str, Any], scene: str, seeds: int, episodes: int) -> pd.DataFrame:
    rows = []
    for b in (256, 320, 384, 448, 512):
        action = GovernanceAction(m=7, b=b, tau=60, theta=0.55)
        summary = run_fixed_action(config, action=action, scene=scene, seeds=seeds, episodes=episodes)
        rows.append(flatten_record({"audit": f"b_scan_{scene}", "scene": scene, "b": b}, summary))
    return pd.DataFrame(rows)


def audit_tau_scan(config: dict[str, Any], scene: str, seeds: int, episodes: int) -> pd.DataFrame:
    rows = []
    for tau in (40, 60, 80, 100):
        action = GovernanceAction(m=7, b=384, tau=tau, theta=0.55)
        summary = run_fixed_action(config, action=action, scene=scene, seeds=seeds, episodes=episodes)
        rows.append(flatten_record({"audit": f"tau_scan_{scene}", "scene": scene, "tau": tau}, summary))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir", default="audits/theory_consistency")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--episodes", type=int, default=4)
    args = parser.parse_args()

    base_config = load_config(args.config)
    output_dir = ensure_dir(Path(args.output_dir))

    m_scan = audit_m_scan(base_config, seeds=args.seeds, episodes=args.episodes)
    theta_grid = audit_theta_grid(base_config, seeds=args.seeds, episodes=args.episodes)
    b_stable = audit_b_scan(base_config, scene="stable", seeds=args.seeds, episodes=args.episodes)
    b_load = audit_b_scan(base_config, scene="load_shock", seeds=args.seeds, episodes=args.episodes)
    tau_stable = audit_tau_scan(base_config, scene="stable", seeds=args.seeds, episodes=args.episodes)
    tau_high_rtt = audit_tau_scan(base_config, scene="high_rtt_burst", seeds=args.seeds, episodes=args.episodes)

    artifacts = {
        "m_scan": str(output_dir / "m_scan_churn_burst.csv"),
        "theta_grid": str(output_dir / "m_theta_grid_malicious_burst.csv"),
        "b_scan_stable": str(output_dir / "b_scan_stable.csv"),
        "b_scan_load_shock": str(output_dir / "b_scan_load_shock.csv"),
        "tau_scan_stable": str(output_dir / "tau_scan_stable.csv"),
        "tau_scan_high_rtt_burst": str(output_dir / "tau_scan_high_rtt_burst.csv"),
    }
    m_scan.to_csv(artifacts["m_scan"], index=False)
    theta_grid.to_csv(artifacts["theta_grid"], index=False)
    b_stable.to_csv(artifacts["b_scan_stable"], index=False)
    b_load.to_csv(artifacts["b_scan_load_shock"], index=False)
    tau_stable.to_csv(artifacts["tau_scan_stable"], index=False)
    tau_high_rtt.to_csv(artifacts["tau_scan_high_rtt_burst"], index=False)

    summary = {
        "config": str(args.config),
        "seeds": int(args.seeds),
        "episodes": int(args.episodes),
        "artifacts": artifacts,
    }
    write_json(output_dir / "summary.json", summary)
    print("Theory consistency audit completed.")
    for name, path in artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
