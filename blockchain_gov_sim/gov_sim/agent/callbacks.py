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

    def __init__(
        self,
        log_path: str | Path,
        audit_path: str | Path | None = None,
        recent_window: int = 50,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.log_path = Path(log_path)
        self.audit_path = Path(audit_path) if audit_path is not None else None
        self.recent_window = int(recent_window)
        self.rows: list[dict[str, Any]] = []
        self.step_rows: list[dict[str, Any]] = []
        self._start_time = 0.0
        self._total_timesteps = 0
        self._ep_reward = 0.0
        self._ep_cost = 0.0
        self._ep_constraint_cost = 0.0
        self._ep_len = 0
        self._ep_unsafe = 0.0
        self._ep_tps = 0.0
        self._ep_latency = 0.0
        self._ep_mask_ratio = 0.0
        self._ep_eligible_sum = 0.0
        self._ep_eligible_sq_sum = 0.0
        self._ep_infeasible = 0.0
        self._ep_infeasible_high_template = 0.0
        self._ep_timeout = 0.0
        self._ep_policy_invalid = 0.0
        self._ep_structural_infeasible = 0.0
        self._ep_h_lcb_sum = 0.0
        self._episode_seed = 0
        self._episode_scenario_type = "unknown"
        self._episode_hasher = hashlib.sha256()
        self._trajectory_fingerprints: list[str] = []
        self._repeat_count = 0
        self._executed_high_template_counts: Counter[str] = Counter()
        self._executed_low_action_counts: Counter[str] = Counter()
        self._ep_high_template_counts: Counter[str] = Counter()
        self._ep_low_action_counts: Counter[str] = Counter()
        self._invalid_action_total = 0.0
        self._policy_invalid_total = 0.0
        self._structural_infeasible_total = 0.0
        self._infeasible_total = 0.0
        self._step_total = 0
        self._action_counts = {
            "m": Counter(),
            "b": Counter(),
            "tau": Counter(),
            "theta": Counter(),
        }
        self._action_combo_counts = Counter()
        self._scenario_counts = Counter()
        self._scenario_action_counts: dict[str, Counter[str]] = {}
        self._eligible_total = 0.0
        self._eligible_sq_total = 0.0
        self._eligible_count = 0
        self._ep_entropy_sum = 0.0
        self._ep_reward_components_sum = 0.0
        self._ep_cost_components_sum = 0.0
        self._ep_reward_term_sums: dict[str, float] = {}
        self._ep_cost_term_sums: dict[str, float] = {}

    def _reset_episode_tracking(self) -> None:
        self._ep_reward = 0.0
        self._ep_cost = 0.0
        self._ep_constraint_cost = 0.0
        self._ep_len = 0
        self._ep_unsafe = 0.0
        self._ep_tps = 0.0
        self._ep_latency = 0.0
        self._ep_mask_ratio = 0.0
        self._ep_eligible_sum = 0.0
        self._ep_eligible_sq_sum = 0.0
        self._ep_infeasible = 0.0
        self._ep_infeasible_high_template = 0.0
        self._ep_timeout = 0.0
        self._ep_policy_invalid = 0.0
        self._ep_structural_infeasible = 0.0
        self._ep_h_lcb_sum = 0.0
        self._episode_seed = 0
        self._episode_scenario_type = "unknown"
        self._episode_hasher = hashlib.sha256()
        self._ep_high_template_counts = Counter()
        self._ep_low_action_counts = Counter()
        self._ep_entropy_sum = 0.0
        self._ep_reward_components_sum = 0.0
        self._ep_cost_components_sum = 0.0
        self._ep_reward_term_sums = {}
        self._ep_cost_term_sums = {}

    def _action_signature(self, info: dict[str, Any]) -> str:
        return f"{int(info.get('m_e', 0))}|{int(info.get('b_e', 0))}|{int(info.get('tau_e', 0))}|{float(info.get('theta_e', 0.0)):.2f}"

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
        scenario_total = max(sum(self._scenario_counts.values()), 1)
        scenario_distribution = {scenario: float(count / scenario_total) for scenario, count in sorted(self._scenario_counts.items())}
        total_actions = max(sum(self._action_combo_counts.values()), 1)
        top_actions = self._action_combo_counts.most_common(5)
        per_scenario_action_usage: dict[str, dict[str, Any]] = {}
        for scenario, counter in sorted(self._scenario_action_counts.items()):
            total = max(sum(counter.values()), 1)
            top_actions = counter.most_common(5)
            dominant_ratio = float(top_actions[0][1] / total) if top_actions else 0.0
            per_scenario_action_usage[scenario] = {
                "dominant_action_ratio": dominant_ratio,
                "top_k_action_distribution": [
                    {"action": action, "count": int(count), "ratio": float(count / total)} for action, count in top_actions
                ],
            }
        summary = {
            "episodes": int(len(self.rows)),
            "episode_repeat_ratio": repeat_ratio,
            "unique_trajectory_count": int(len(set(self._trajectory_fingerprints))),
            "recent_unique_trajectory_count": int(len(set(self._trajectory_fingerprints[-self.recent_window :]))),
            "recent_window": int(self.recent_window),
            "eligible_size_mean": eligible_mean,
            "eligible_size_std": float(eligible_var**0.5),
            "h_lcb_mean": float(sum(float(row.get("h_lcb_mean", 0.0)) for row in self.rows) / max(len(self.rows), 1)),
            "action_frequency": action_frequencies,
            "dominant_action_ratio": float(top_actions[0][1] / total_actions) if top_actions else 0.0,
            "top_k_action_distribution": [
                {"action": action, "count": int(count), "ratio": float(count / total_actions)} for action, count in top_actions
            ],
            "scenario_type_distribution": scenario_distribution,
            "per_scenario_action_usage_summary": per_scenario_action_usage,
            "invalid_action_rate": float(self._invalid_action_total / max(self._step_total, 1)),
            "policy_invalid_rate": float(self._policy_invalid_total / max(self._step_total, 1)),
            "structural_infeasible_rate": float(self._structural_infeasible_total / max(self._step_total, 1)),
            "infeasible_rate": float(self._infeasible_total / max(self._step_total, 1)),
        }
        if self._executed_high_template_counts:
            total_high_templates = max(sum(self._executed_high_template_counts.values()), 1)
            summary["executed_high_template_distribution"] = [
                {"action": action, "count": int(count), "ratio": float(count / total_high_templates)}
                for action, count in self._executed_high_template_counts.most_common()
            ]
        if self._executed_low_action_counts:
            total_low_actions = max(sum(self._executed_low_action_counts.values()), 1)
            summary["executed_low_action_distribution"] = [
                {"action": action, "count": int(count), "ratio": float(count / total_low_actions)}
                for action, count in self._executed_low_action_counts.most_common()
            ]
        return summary

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
        
        # 尝试获取 entropy（如果可用）
        entropy = None
        try:
            if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'get_distribution'):
                obs_tensor = self.locals.get("obs_tensor")
                if obs_tensor is not None:
                    action_masks = None
                    if isinstance(obs_tensor, dict) and "action_mask" in obs_tensor:
                        action_masks = obs_tensor["action_mask"].detach().cpu().numpy().astype(bool)
                    was_training = bool(self.model.policy.training)
                    self.model.policy.train(False)
                    distribution = self.model.policy.get_distribution(obs_tensor, action_masks=action_masks)
                    if hasattr(distribution, 'distribution') and hasattr(distribution.distribution, 'entropy'):
                        entropy = float(distribution.distribution.entropy().mean().detach().cpu().numpy())
                    self.model.policy.train(was_training)
        except Exception:
            pass
        
        if rewards is not None:
            self._ep_reward += float(rewards[0])
        if infos:
            info = infos[0]
            reward_val = float(rewards[0]) if rewards is not None else 0.0
            cost_val = float(info.get("cost", 0.0))
            constraint_cost_val = float(info.get("high_chunk_normalized_cost", cost_val))
            self._ep_cost += cost_val
            self._ep_constraint_cost += constraint_cost_val
            self._ep_len += 1
            self._ep_unsafe += float(info.get("unsafe", 0.0))
            self._ep_tps += float(info.get("tps", 0.0))
            self._ep_latency += float(info.get("L_bar_e", 0.0))
            self._ep_mask_ratio += float(info.get("mask_ratio", 0.0))
            eligible = float(info.get("eligible_size", 0.0))
            self._ep_eligible_sum += eligible
            self._ep_eligible_sq_sum += eligible * eligible
            self._ep_infeasible += float(info.get("infeasible", 0.0))
            self._ep_infeasible_high_template += float(
                int(info.get("infeasible", 0) and info.get("infeasible_reason") == "no_legal_high_template")
            )
            self._ep_timeout += float(info.get("timeout_failure", 0.0))
            self._ep_policy_invalid += float(info.get("policy_invalid", 0.0))
            self._ep_structural_infeasible += float(info.get("structural_infeasible", 0.0))
            self._ep_h_lcb_sum += float(info.get("h_LCB_e", info.get("h_LCB", 0.0)))
            self._eligible_total += eligible
            self._eligible_sq_total += eligible * eligible
            self._eligible_count += 1
            self._episode_seed = int(info.get("episode_seed", self._episode_seed))
            self._episode_scenario_type = str(info.get("scenario_type", self._episode_scenario_type))
            self._fingerprint_step(info)
            executed_high_template = str(info.get("executed_high_template", ""))
            if executed_high_template:
                self._executed_high_template_counts[executed_high_template] += 1
                self._ep_high_template_counts[executed_high_template] += 1
            executed_low_action = str(info.get("executed_low_action", ""))
            if executed_low_action:
                self._executed_low_action_counts[executed_low_action] += 1
                self._ep_low_action_counts[executed_low_action] += 1
            self._invalid_action_total += float(info.get("invalid_action", 0.0))
            self._policy_invalid_total += float(info.get("policy_invalid", 0.0))
            self._structural_infeasible_total += float(info.get("structural_infeasible", 0.0))
            self._infeasible_total += float(info.get("infeasible", 0.0))
            self._step_total += 1
            action_signature = self._action_signature(info)
            self._action_combo_counts[action_signature] += 1
            self._scenario_action_counts.setdefault(self._episode_scenario_type, Counter())[action_signature] += 1
            for key, info_key in [("m", "m_e"), ("b", "b_e"), ("tau", "tau_e"), ("theta", "theta_e")]:
                self._action_counts[key][info.get(info_key)] += 1
            
            # 收集 entropy 和 reward/cost 分解
            if entropy is not None:
                self._ep_entropy_sum += entropy
            self._ep_reward_components_sum += reward_val
            self._ep_cost_components_sum += cost_val
            reward_terms = info.get("reward_terms", {})
            cost_terms = info.get("cost_terms", {})
            if isinstance(reward_terms, dict):
                for key, value in reward_terms.items():
                    self._ep_reward_term_sums[key] = self._ep_reward_term_sums.get(key, 0.0) + float(value)
            if isinstance(cost_terms, dict):
                for key, value in cost_terms.items():
                    self._ep_cost_term_sums[key] = self._ep_cost_term_sums.get(key, 0.0) + float(value)
            
            # 记录 step 级详细日志
            step_row = {
                "timesteps": int(self.num_timesteps),
                "episode_seed": int(self._episode_seed),
                "scenario_type": str(self._episode_scenario_type),
                "step_in_episode": int(self._ep_len),
                "reward": reward_val,
                "cost": cost_val,
                "high_chunk_discounted_cost": float(info.get("high_chunk_discounted_cost", cost_val)),
                "high_chunk_normalized_cost": constraint_cost_val,
                "high_chunk_gamma_sum": float(info.get("high_chunk_gamma_sum", 1.0)),
                "entropy": entropy if entropy is not None else 0.0,
                "lagrangian_lambda": float(getattr(self.model, "lambda_value", 0.0)),
                "m_e": int(info.get("m_e", 0)),
                "b_e": int(info.get("b_e", 0)),
                "tau_e": int(info.get("tau_e", 0)),
                "theta_e": float(info.get("theta_e", 0.0)),
                "executed_high_template": executed_high_template,
                "executed_low_action": executed_low_action,
                "tps": float(info.get("tps", 0.0)),
                "latency": float(info.get("L_bar_e", 0.0)),
                "unsafe": float(info.get("unsafe", 0.0)),
                "timeout_failure": float(info.get("timeout_failure", 0.0)),
                "policy_invalid": float(info.get("policy_invalid", 0.0)),
                "structural_infeasible": float(info.get("structural_infeasible", 0.0)),
                "mask_ratio": float(info.get("mask_ratio", 0.0)),
                "eligible_size": eligible,
                "h_LCB_e": float(info.get("h_LCB_e", info.get("h_LCB", 0.0))),
                "committee_honest_ratio": float(info.get("committee_honest_ratio", info.get("h_e", 0.0))),
                "S_e": float(info.get("S_e", 0.0)),
                "B_e": float(info.get("B_e", 0.0)),
                "L_queue_e": float(info.get("L_queue_e", 0.0)),
                "L_batch_e": float(info.get("L_batch_e", 0.0)),
                "L_cons_e": float(info.get("L_cons_e", 0.0)),
                "block_slack": float(info.get("block_slack", 0.0)),
            }
            if isinstance(reward_terms, dict):
                for key, value in reward_terms.items():
                    step_row[f"reward_{key}"] = float(value)
            if isinstance(cost_terms, dict):
                for key, value in cost_terms.items():
                    step_row[f"cost_{key}"] = float(value)
            self.step_rows.append(step_row)
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
            self._scenario_counts[self._episode_scenario_type] += 1
            
            # 计算 episode 级 template/action 分布
            import numpy as np
            ep_high_template_total = max(sum(self._ep_high_template_counts.values()), 1)
            ep_low_action_total = max(sum(self._ep_low_action_counts.values()), 1)
            top_high_template = self._ep_high_template_counts.most_common(1)[0][0] if self._ep_high_template_counts else ""
            top_low_action = self._ep_low_action_counts.most_common(1)[0][0] if self._ep_low_action_counts else ""
            high_template_entropy = -sum((c/ep_high_template_total) * np.log(c/ep_high_template_total + 1e-12)
                                         for c in self._ep_high_template_counts.values()) if self._ep_high_template_counts else 0.0
            low_action_entropy = -sum((c/ep_low_action_total) * np.log(c/ep_low_action_total + 1e-12)
                                      for c in self._ep_low_action_counts.values()) if self._ep_low_action_counts else 0.0
            
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
                "constraint_violation": float(
                    max(0.0, self._ep_constraint_cost / ep_len - float(getattr(self.model, "cost_limit", 0.0)))
                ),
                "episode_seed": int(self._episode_seed),
                "scenario_type": str(self._episode_scenario_type),
                "eligible_size_mean": float(eligible_mean),
                "eligible_size_std": float(eligible_var**0.5),
                "infeasible_rate": float(self._ep_infeasible / ep_len),
                "infeasible_high_template_rate": float(self._ep_infeasible_high_template / ep_len),
                "timeout_rate": float(self._ep_timeout / ep_len),
                "policy_invalid_rate_local": float(self._ep_policy_invalid / ep_len),
                "structural_infeasible_rate_local": float(self._ep_structural_infeasible / ep_len),
                "h_lcb_mean": float(self._ep_h_lcb_sum / ep_len),
                "trajectory_fingerprint": fingerprint,
                "episode_repeat_ratio": repeat_ratio,
                "recent_unique_trajectory_count": int(len(set(self._trajectory_fingerprints[-self.recent_window :]))),
                "invalid_action_rate": float(self._invalid_action_total / max(self._step_total, 1)),
                "policy_invalid_rate": float(self._policy_invalid_total / max(self._step_total, 1)),
                "structural_infeasible_rate": float(self._structural_infeasible_total / max(self._step_total, 1)),
                "mean_entropy": float(self._ep_entropy_sum / ep_len) if self._ep_entropy_sum > 0 else 0.0,
                "top_high_template": top_high_template,
                "high_template_entropy": float(high_template_entropy),
                "top_low_action": top_low_action,
                "low_action_entropy": float(low_action_entropy),
                "num_unique_high_templates": int(len(self._ep_high_template_counts)),
                "num_unique_low_actions": int(len(self._ep_low_action_counts)),
            }
            for key, value in sorted(self._ep_reward_term_sums.items()):
                row[f"reward_component_{key}"] = float(value / ep_len)
            for key, value in sorted(self._ep_cost_term_sums.items()):
                row[f"cost_component_{key}"] = float(value / ep_len)
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
                f"scenario={row.get('scenario_type', 'unknown')} "
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
        
        # 导出 step 级详细日志
        if self.step_rows:
            step_log_path = self.log_path.parent / f"{self.log_path.stem}_steps.csv"
            pd.DataFrame(self.step_rows).to_csv(step_log_path, index=False)
        
        if self.audit_path is not None:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            write_json(self.audit_path, self.audit_summary())
