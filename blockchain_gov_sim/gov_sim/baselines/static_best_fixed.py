"""Static best fixed-action baseline。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from gov_sim.baselines import BaselineBase
from gov_sim.constants import ACTION_DIM
from gov_sim.env.gov_env import BlockchainGovEnv
from gov_sim.modules.metrics_tracker import MetricsTracker
from gov_sim.utils.io import deep_update


class StaticBestFixedBaseline(BaselineBase):
    """全扫 240 固定动作后选择最优固定动作。"""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config=config, name="Static-Best-Fixed")
        self.cost_limit = 0.10
        self.scan_episodes = int(config.get("baselines", {}).get("static_scan_episodes", 1))
        self.best_action_idx: int | None = None
        self.best_action_tuple: tuple[float, float, int, int] | None = None
        self.scan_rows: list[dict[str, Any]] = []

    def reset(self) -> None:
        return None

    def _score_key(self, row: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
        return (
            float(row["mean_cost"] > self.cost_limit),
            float(row["unsafe_rate"]),
            float(row["timeout_rate"]),
            float(row["mean_latency"]),
            -float(row["tps"]),
            -float(row["mean_reward"]),
        )

    def fit(self, episodes: int | None = None) -> int:
        if self.best_action_idx is not None:
            return int(self.best_action_idx)
        scan_episodes = int(self.scan_episodes if episodes is None else episodes)
        cfg = deep_update(self.config, {"env": {"episode_length": int(self.config["env"]["episode_length"])}})
        best_row: dict[str, Any] | None = None
        base_seed = int(cfg.get("seed", 42))
        self.scan_rows = []
        for action_idx in range(ACTION_DIM):
            tracker = MetricsTracker()
            rewards: list[float] = []
            costs: list[float] = []
            for episode in range(scan_episodes):
                env = BlockchainGovEnv(cfg)
                _, _, _ = env.reset(seed=base_seed + 50000 + action_idx * max(scan_episodes, 1) + episode)
                done = False
                while not done:
                    _, _, reward, cost, done, info = env.step(action_idx)
                    rewards.append(float(reward))
                    costs.append(float(cost))
                    tracker.update(info)
            summary = tracker.summary()
            row = {
                "action_idx": int(action_idx),
                "action_tuple": (
                    float(self.codec.decode(action_idx).rho_m),
                    float(self.codec.decode(action_idx).theta),
                    int(self.codec.decode(action_idx).b),
                    int(self.codec.decode(action_idx).tau),
                ),
                "unsafe_rate": float(summary.get("unsafe_rate", 1.0)),
                "timeout_rate": float(summary.get("timeout_failure_rate", 1.0)),
                "mean_latency": float(summary.get("mean_latency", float("inf"))),
                "tps": float(summary.get("tps", 0.0)),
                "mean_reward": float(np.mean(rewards)) if rewards else float("-inf"),
                "mean_cost": float(np.mean(costs)) if costs else float("inf"),
            }
            self.scan_rows.append(row)
            if best_row is None or self._score_key(row) < self._score_key(best_row):
                best_row = row
        if best_row is None:
            raise RuntimeError("Failed to scan fixed actions.")
        self.best_action_idx = int(best_row["action_idx"])
        self.best_action_tuple = tuple(best_row["action_tuple"])
        return int(self.best_action_idx)

    def export_scan_csv(self, path: str | Path) -> None:
        import pandas as pd

        if not self.scan_rows:
            return
        pd.DataFrame(self.scan_rows).to_csv(path, index=False)

    def select_action(self, env: Any, obs: np.ndarray, legal_mask: np.ndarray | None = None) -> int:
        del env, obs
        action_idx = self.fit()
        if legal_mask is None:
            return int(action_idx)
        if int(legal_mask[action_idx]) == 1:
            return int(action_idx)
        action = self.codec.decode(action_idx)
        return self._nearest_legal_action(action, legal_mask)
