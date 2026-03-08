"""训练回调。

该文件负责把 step 级 rollout 信息整理成 episode 级 csv，
便于答辩时直接展示“训练奖励/成本/不安全率/吞吐/时延”曲线。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback


class TrainLoggingCallback(BaseCallback):
    """记录 episode 级训练指标。

    之所以不直接依赖 SB3 默认 logger，是因为论文实验通常更需要结构化 csv，
    后续作图、统计与对比都会更方便。
    """

    def __init__(self, log_path: str | Path, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self.log_path = Path(log_path)
        self.rows: list[dict[str, Any]] = []
        self._ep_reward = 0.0
        self._ep_cost = 0.0
        self._ep_len = 0
        self._ep_unsafe = 0.0
        self._ep_tps = 0.0
        self._ep_latency = 0.0

    def _on_step(self) -> bool:
        """在每个环境步结束后更新当前 episode 的聚合统计。"""
        rewards = self.locals.get("rewards")
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones")
        if rewards is not None:
            self._ep_reward += float(rewards[0])
        if infos:
            self._ep_cost += float(infos[0].get("cost", 0.0))
            self._ep_len += 1
            self._ep_unsafe += float(infos[0].get("unsafe", 0.0))
            self._ep_tps += float(infos[0].get("tps", 0.0))
            self._ep_latency += float(infos[0].get("L_bar_e", 0.0))
        if dones is not None and bool(dones[0]):
            info = infos[0] if infos else {}
            ep_len = max(self._ep_len, 1)
            self.rows.append(
                {
                    "timesteps": int(self.num_timesteps),
                    "episode_reward": self._ep_reward,
                    "episode_cost": self._ep_cost,
                    "episode_len": self._ep_len,
                    "unsafe_rate": self._ep_unsafe / ep_len,
                    "tps": self._ep_tps / ep_len,
                    "latency": self._ep_latency / ep_len,
                    "lagrangian_lambda": float(getattr(self.model, "lambda_value", 0.0)),
                    "constraint_violation": float(max(0.0, self._ep_cost / ep_len - float(getattr(self.model, "cost_limit", 0.0)))),
                }
            )
            self._ep_reward = 0.0
            self._ep_cost = 0.0
            self._ep_len = 0
            self._ep_unsafe = 0.0
            self._ep_tps = 0.0
            self._ep_latency = 0.0
        return True

    def _on_training_end(self) -> None:
        if not self.rows:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.rows).to_csv(self.log_path, index=False)
