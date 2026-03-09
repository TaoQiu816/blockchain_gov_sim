"""训练回调。

该文件负责把 step 级 rollout 信息整理成 episode 级 csv，
便于答辩时直接展示“训练奖励/成本/不安全率/吞吐/时延”曲线。
"""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
import time
from typing import Any

import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback

from gov_sim.utils.io import write_json


class TrainLoggingCallback(BaseCallback):
    """记录 episode 级训练指标。

    之所以不直接依赖 SB3 默认 logger，是因为论文实验通常更需要结构化 csv，
    后续作图、统计与对比都会更方便。
    """

    def __init__(self, log_path: str | Path, audit_path: str | Path | None = None, recent_window: int = 50, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self.log_path = Path(log_path)
        self.audit_path = Path(audit_path) if audit_path is not None else None
        self.recent_window = int(recent_window)
        self.rows: list[dict[str, Any]] = []
        self._start_time = 0.0
        self._total_timesteps = 0
        self._ep_reward = 0.0
        self._ep_cost = 0.0
        self._ep_len = 0
        self._ep_unsafe = 0.0
        self._ep_tps = 0.0
        self._ep_latency = 0.0
        self._ep_mask_ratio = 0.0
        self._ep_eligible_sum = 0.0
        self._ep_eligible_sq_sum = 0.0
        self._episode_seed = 0
        self._episode_hasher = hashlib.sha256()
        self._trajectory_fingerprints: list[str] = []
        self._repeat_count = 0
        self._action_counts = {
            "m": Counter(),
            "b": Counter(),
            "tau": Counter(),
            "theta": Counter(),
        }
        self._eligible_total = 0.0
        self._eligible_sq_total = 0.0
        self._eligible_count = 0

    def _reset_episode_tracking(self) -> None:
        self._ep_reward = 0.0
        self._ep_cost = 0.0
        self._ep_len = 0
        self._ep_unsafe = 0.0
        self._ep_tps = 0.0
        self._ep_latency = 0.0
        self._ep_mask_ratio = 0.0
        self._ep_eligible_sum = 0.0
        self._ep_eligible_sq_sum = 0.0
        self._episode_seed = 0
        self._episode_hasher = hashlib.sha256()

    def _fingerprint_step(self, info: dict[str, Any]) -> None:
        payload = (
            int(info.get("A_e", 0)),
            round(float(info.get("Q_e", 0.0)), 6),
            round(float(info.get("S_e", 0.0)), 6),
            round(float(info.get("L_bar_e", 0.0)), 6),
            round(float(info.get("RTT_e", 0.0)), 6),
            round(float(info.get("chi_e", 0.0)), 6),
            round(float(info.get("h_e", 0.0)), 6),
            int(info.get("U_e", 0)),
            int(info.get("Z_e", 0)),
            int(info.get("m_e", 0)),
            int(info.get("b_e", 0)),
            int(info.get("tau_e", 0)),
            round(float(info.get("theta_e", 0.0)), 6),
            int(info.get("eligible_size", 0)),
            round(float(info.get("queue_next", 0.0)), 6),
            tuple(int(v) for v in info.get("committee_members", [])),
        )
        self._episode_hasher.update(repr(payload).encode("utf-8"))

    def _window_mean(self, key: str, window: list[dict[str, Any]]) -> float:
        return sum(float(item[key]) for item in window) / max(len(window), 1)

    def audit_summary(self) -> dict[str, Any]:
        repeat_ratio = float(self._repeat_count / max(len(self._trajectory_fingerprints) - 1, 1)) if self._trajectory_fingerprints else 0.0
        eligible_mean = float(self._eligible_total / max(self._eligible_count, 1))
        eligible_var = max(self._eligible_sq_total / max(self._eligible_count, 1) - eligible_mean**2, 0.0)
        action_frequencies = {
            key: {str(action): float(count / max(sum(counter.values()), 1)) for action, count in sorted(counter.items())}
            for key, counter in self._action_counts.items()
        }
        return {
            "episodes": int(len(self.rows)),
            "episode_repeat_ratio": repeat_ratio,
            "unique_trajectory_count": int(len(set(self._trajectory_fingerprints))),
            "recent_unique_trajectory_count": int(len(set(self._trajectory_fingerprints[-self.recent_window :]))),
            "recent_window": int(self.recent_window),
            "eligible_size_mean": eligible_mean,
            "eligible_size_std": float(eligible_var**0.5),
            "action_frequency": action_frequencies,
        }

    def _on_training_start(self) -> None:
        """记录训练开始时间与目标步数，用于控制台实时进度输出。"""
        self._start_time = time.monotonic()
        self._total_timesteps = int(getattr(self.model, "_total_timesteps", 0))
        self._reset_episode_tracking()

    def _on_step(self) -> bool:
        """在每个环境步结束后更新当前 episode 的聚合统计。"""
        rewards = self.locals.get("rewards")
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones")
        if rewards is not None:
            self._ep_reward += float(rewards[0])
        if infos:
            info = infos[0]
            self._ep_cost += float(info.get("cost", 0.0))
            self._ep_len += 1
            self._ep_unsafe += float(info.get("unsafe", 0.0))
            self._ep_tps += float(info.get("tps", 0.0))
            self._ep_latency += float(info.get("L_bar_e", 0.0))
            self._ep_mask_ratio += float(info.get("mask_ratio", 0.0))
            eligible = float(info.get("eligible_size", 0.0))
            self._ep_eligible_sum += eligible
            self._ep_eligible_sq_sum += eligible * eligible
            self._eligible_total += eligible
            self._eligible_sq_total += eligible * eligible
            self._eligible_count += 1
            self._episode_seed = int(info.get("episode_seed", self._episode_seed))
            self._fingerprint_step(info)
            for key, info_key in [("m", "m_e"), ("b", "b_e"), ("tau", "tau_e"), ("theta", "theta_e")]:
                self._action_counts[key][info.get(info_key)] += 1
        if dones is not None and bool(dones[0]):
            info = infos[0] if infos else {}
            ep_len = max(self._ep_len, 1)
            fingerprint = self._episode_hasher.hexdigest()
            if self._trajectory_fingerprints and fingerprint == self._trajectory_fingerprints[-1]:
                self._repeat_count += 1
            self._trajectory_fingerprints.append(fingerprint)
            repeat_ratio = float(self._repeat_count / max(len(self._trajectory_fingerprints) - 1, 1)) if len(self._trajectory_fingerprints) > 1 else 0.0
            eligible_mean = self._ep_eligible_sum / ep_len
            eligible_var = max(self._ep_eligible_sq_sum / ep_len - eligible_mean**2, 0.0)
            row = {
                "timesteps": int(self.num_timesteps),
                "episode_reward": self._ep_reward,
                "episode_cost": self._ep_cost,
                "episode_len": self._ep_len,
                "unsafe_rate": self._ep_unsafe / ep_len,
                "tps": self._ep_tps / ep_len,
                "latency": self._ep_latency / ep_len,
                "mask_ratio": self._ep_mask_ratio / ep_len,
                "lagrangian_lambda": float(getattr(self.model, "lambda_value", 0.0)),
                "constraint_violation": float(max(0.0, self._ep_cost / ep_len - float(getattr(self.model, "cost_limit", 0.0)))),
                "episode_seed": int(self._episode_seed),
                "eligible_size_mean": float(eligible_mean),
                "eligible_size_std": float(eligible_var**0.5),
                "trajectory_fingerprint": fingerprint,
                "episode_repeat_ratio": repeat_ratio,
                "recent_unique_trajectory_count": int(len(set(self._trajectory_fingerprints[-self.recent_window :]))),
            }
            self.rows.append(row)
            window = self.rows[-10:]
            row["rolling_reward_mean"] = self._window_mean("episode_reward", window)
            row["rolling_cost_mean"] = self._window_mean("episode_cost", window)
            row["rolling_unsafe_mean"] = self._window_mean("unsafe_rate", window)
            row["rolling_lambda_mean"] = self._window_mean("lagrangian_lambda", window)
            self._print_progress(row)
            self._reset_episode_tracking()
        return True

    def _print_progress(self, row: dict[str, Any]) -> None:
        """把训练关键指标实时打印到 stdout，便于 `tee` 保存控制台日志。

        这里输出最近 10 个 episode 的滚动均值，而不是单个 episode 的瞬时值，
        这样更适合论文实验期间观察训练是否稳定、lambda 是否失控以及 mask 是否过于激进。
        """

        window = self.rows[-10:]
        elapsed = time.monotonic() - self._start_time
        reward_mean = self._window_mean("episode_reward", window)
        cost_mean = self._window_mean("episode_cost", window)
        unsafe_mean = self._window_mean("unsafe_rate", window)
        mask_mean = self._window_mean("mask_ratio", window)
        lambda_mean = self._window_mean("lagrangian_lambda", window)
        eligible_mean = self._window_mean("eligible_size_mean", window)
        progress = self._total_timesteps if self._total_timesteps > 0 else "?"
        print(
            (
                f"[train] step={row['timesteps']}/{progress} "
                f"reward_mean={reward_mean:.3f} "
                f"cost_mean={cost_mean:.3f} "
                f"unsafe_rate={unsafe_mean:.3f} "
                f"lambda={float(row['lagrangian_lambda']):.4f} "
                f"lambda_mean={lambda_mean:.4f} "
                f"mask_ratio={mask_mean:.3f} "
                f"eligible_mean={eligible_mean:.2f} "
                f"repeat_ratio={float(row['episode_repeat_ratio']):.3f} "
                f"uniq{self.recent_window}={int(row['recent_unique_trajectory_count'])} "
                f"elapsed={elapsed:.1f}s"
            ),
            flush=True,
        )

    def _on_training_end(self) -> None:
        if not self.rows:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.rows).to_csv(self.log_path, index=False)
        if self.audit_path is not None:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            write_json(self.audit_path, self.audit_summary())
