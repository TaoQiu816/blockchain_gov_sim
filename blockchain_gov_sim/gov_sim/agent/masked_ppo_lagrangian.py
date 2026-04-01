"""Maskable PPO-Lagrangian 主算法实现。

实现思路：
1. 继续使用 `sb3-contrib` 的 MaskablePPO 策略网络、动作分布与 invalid-action mask；
2. 额外维护独立的 cost critic；
3. 额外维护拉格朗日乘子 `lambda >= 0`；
4. reward 与 cost 分开做 GAE；
5. actor 使用 `A_L = A_r - lambda * A_c` 更新。

这不是“普通 PPO + reward penalty”，而是显式约束优化版本。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator

import gymnasium as gym
import numpy as np
import torch
from torch.distributions import Distribution
from torch import nn
from torch.nn import functional as F

try:
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.type_aliases import MaybeCallback
    from stable_baselines3.common.utils import explained_variance, obs_as_tensor
    from stable_baselines3.common.vec_env import VecEnv
except ImportError as exc:  # pragma: no cover - 只有未安装 RL 依赖时触发
    raise ImportError(
        "导入 MaskablePPOLagrangian 失败：缺少 sb3-contrib / stable-baselines3。\n"
        "如果你正在复用 base_requirements.txt 对应的 conda 环境，请执行：\n"
        "  pip install -r base_env_delta_requirements.txt"
    ) from exc

from gov_sim.agent.policy_wrappers import CostValueNet
from gov_sim.utils.math_utils import RunningMeanStd

# 在极稀疏 action mask + 极小 softmax 概率下，PyTorch 对 categorical simplex 的严格校验
# 可能因浮点舍入而误报。这里关闭 validate_args，避免训练在高压 pilot/workpoint 下被数值噪声打断。
Distribution.set_default_validate_args(False)


@dataclass
class ConstrainedRolloutBatch:
    """一个 mini-batch 的 reward/cost 联合训练数据。"""

    observations: dict[str, torch.Tensor]
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    old_cost_values: torch.Tensor
    cost_advantages: torch.Tensor
    cost_returns: torch.Tensor
    action_masks: torch.Tensor


class ConstrainedDictRolloutBuffer:
    """同时缓存 reward 回报和 cost 回报的 rollout buffer。"""

    def __init__(
        self,
        buffer_size: int,
        observation_space: gym.spaces.Dict,
        action_space: gym.spaces.Discrete,
        device: torch.device,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.action_dim = int(action_space.n)
        self.reset()

    def reset(self) -> None:
        """清空缓存，准备下一轮 rollout。"""
        self.pos = 0
        self.full = False
        self.observations = {
            key: np.zeros((self.buffer_size, *space.shape), dtype=space.dtype)
            for key, space in self.observation_space.spaces.items()
        }
        self.actions = np.zeros((self.buffer_size, 1), dtype=np.int64)
        self.rewards = np.zeros(self.buffer_size, dtype=np.float32)
        self.costs = np.zeros(self.buffer_size, dtype=np.float32)
        self.raw_costs = np.zeros(self.buffer_size, dtype=np.float32)
        self.constraint_costs = np.zeros(self.buffer_size, dtype=np.float32)
        self.episode_starts = np.zeros(self.buffer_size, dtype=np.float32)
        self.values = np.zeros(self.buffer_size, dtype=np.float32)
        self.cost_values = np.zeros(self.buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(self.buffer_size, dtype=np.float32)
        self.advantages = np.zeros(self.buffer_size, dtype=np.float32)
        self.cost_advantages = np.zeros(self.buffer_size, dtype=np.float32)
        self.returns = np.zeros(self.buffer_size, dtype=np.float32)
        self.cost_returns = np.zeros(self.buffer_size, dtype=np.float32)
        self.action_masks = np.zeros((self.buffer_size, self.action_dim), dtype=np.float32)

    def add(
        self,
        obs: dict[str, np.ndarray],
        action: np.ndarray,
        reward: float,
        cost: float,
        raw_cost: float,
        constraint_cost: float,
        episode_start: bool,
        value: torch.Tensor,
        cost_value: torch.Tensor,
        log_prob: torch.Tensor,
        action_mask: np.ndarray,
    ) -> None:
        """写入一步 transition。

        注意这里同时缓存三种 cost 口径：
        - `costs`: 给 actor / cost critic / GAE 使用；
        - `raw_costs`: 记录环境返回的折扣 chunk cost，便于审计；
        - `constraint_costs`: 给 lambda update 使用，保持与 cost_limit 同尺度。
        """

        if self.pos >= self.buffer_size:
            raise RuntimeError("Rollout buffer overflow")
        for key in self.observations:
            self.observations[key][self.pos] = np.asarray(obs[key][0]).copy()
        self.actions[self.pos] = np.asarray(action).reshape(1)
        self.rewards[self.pos] = float(reward)
        self.costs[self.pos] = float(cost)
        self.raw_costs[self.pos] = float(raw_cost)
        self.constraint_costs[self.pos] = float(constraint_cost)
        self.episode_starts[self.pos] = float(episode_start)
        self.values[self.pos] = float(value.detach().cpu().numpy().reshape(-1)[0])
        self.cost_values[self.pos] = float(cost_value.detach().cpu().numpy().reshape(-1)[0])
        self.log_probs[self.pos] = float(log_prob.detach().cpu().numpy().reshape(-1)[0])
        self.action_masks[self.pos] = np.asarray(action_mask[0]).astype(np.float32)
        self.pos += 1
        self.full = self.full or self.pos == self.buffer_size

    def compute_returns_and_advantage(
        self,
        last_value: torch.Tensor,
        last_cost_value: torch.Tensor,
        done: np.ndarray,
    ) -> None:
        """分别对 reward 与 cost 做 GAE。"""
        last_gae_lam = 0.0
        last_cost_gae_lam = 0.0
        last_value_np = float(last_value.detach().cpu().numpy().reshape(-1)[0])
        last_cost_value_np = float(last_cost_value.detach().cpu().numpy().reshape(-1)[0])
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - float(done[0])
                next_value = last_value_np
                next_cost_value = last_cost_value_np
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_value = self.values[step + 1]
                next_cost_value = self.cost_values[step + 1]
            delta = self.rewards[step] + self.gamma * next_value * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
            cost_delta = self.costs[step] + self.gamma * next_cost_value * next_non_terminal - self.cost_values[step]
            last_cost_gae_lam = cost_delta + self.gamma * self.gae_lambda * next_non_terminal * last_cost_gae_lam
            self.cost_advantages[step] = last_cost_gae_lam
        self.returns = self.advantages + self.values
        self.cost_returns = self.cost_advantages + self.cost_values

    def get(self, batch_size: int) -> Generator[ConstrainedRolloutBatch, None, None]:
        """按随机顺序生成 mini-batch。"""
        indices = np.random.permutation(self.buffer_size)
        start = 0
        while start < self.buffer_size:
            batch_inds = indices[start : start + batch_size]
            observations = {
                key: torch.as_tensor(self.observations[key][batch_inds], device=self.device)
                for key in self.observations
            }
            yield ConstrainedRolloutBatch(
                observations=observations,
                actions=torch.as_tensor(self.actions[batch_inds], device=self.device).long().flatten(),
                old_values=torch.as_tensor(self.values[batch_inds], device=self.device),
                old_log_prob=torch.as_tensor(self.log_probs[batch_inds], device=self.device),
                advantages=torch.as_tensor(self.advantages[batch_inds], device=self.device),
                returns=torch.as_tensor(self.returns[batch_inds], device=self.device),
                old_cost_values=torch.as_tensor(self.cost_values[batch_inds], device=self.device),
                cost_advantages=torch.as_tensor(self.cost_advantages[batch_inds], device=self.device),
                cost_returns=torch.as_tensor(self.cost_returns[batch_inds], device=self.device),
                action_masks=torch.as_tensor(self.action_masks[batch_inds], device=self.device),
            )
            start += batch_size


class MaskablePPOLagrangian(MaskablePPO):
    """带成本约束的 Maskable PPO。

    关键差异：
    - reward critic 继续复用 SB3 PPO 的 value net；
    - cost critic 独立实现；
    - actor 使用拉格朗日优势而非纯 reward advantage；
    - `lambda_value` 按 rollout 平均原始 cost 与 `cost_limit` 的差值更新。
    """

    def __init__(
        self,
        *args: Any,
        cost_vf_coef: float = 0.5,
        lagrangian_lr: float = 0.02,
        cost_limit: float = 0.18,
        lambda_init: float = 0.1,
        lambda_max: float = 10.0,
        reward_normalization: bool = False,
        cost_normalization: bool = False,
        oracle_anchor_beta_init: float = 0.0,
        oracle_anchor_beta_final: float = 0.0,
        oracle_anchor_decay_fraction: float = 0.0,
        oracle_anchor_batch_size: int | None = None,
        **kwargs: Any,
    ) -> None:
        """初始化约束优化相关超参数。"""
        self.cost_vf_coef = cost_vf_coef
        self.lagrangian_lr = lagrangian_lr
        self.cost_limit = cost_limit
        self.lambda_value = lambda_init
        self.lambda_max = lambda_max
        self.reward_normalization = reward_normalization
        self.cost_normalization = cost_normalization
        self.oracle_anchor_beta_init = float(oracle_anchor_beta_init)
        self.oracle_anchor_beta_final = float(oracle_anchor_beta_final)
        self.oracle_anchor_decay_fraction = float(oracle_anchor_decay_fraction)
        self.oracle_anchor_batch_size = None if oracle_anchor_batch_size is None else int(oracle_anchor_batch_size)
        self._oracle_anchor_states: torch.Tensor | None = None
        self._oracle_anchor_masks: torch.Tensor | None = None
        self._oracle_anchor_targets: torch.Tensor | None = None
        self.reward_rms = RunningMeanStd()
        self.cost_rms = RunningMeanStd()
        super().__init__(*args, **kwargs)

    def set_oracle_anchor_schedule(
        self,
        *,
        beta_init: float,
        beta_final: float = 0.0,
        decay_fraction: float = 0.5,
        batch_size: int | None = None,
    ) -> None:
        self.oracle_anchor_beta_init = float(beta_init)
        self.oracle_anchor_beta_final = float(beta_final)
        self.oracle_anchor_decay_fraction = float(max(decay_fraction, 0.0))
        self.oracle_anchor_batch_size = None if batch_size is None else int(batch_size)

    def load_oracle_anchor_dataset(self, dataset_path: str | Path) -> None:
        data = np.load(Path(dataset_path), allow_pickle=False)
        if "states" not in data or "action_masks" not in data or "soft_targets" not in data:
            raise KeyError("Oracle anchor dataset must contain states, action_masks, and soft_targets.")
        self._oracle_anchor_states = torch.as_tensor(data["states"], dtype=torch.float32, device=self.device)
        self._oracle_anchor_masks = torch.as_tensor(data["action_masks"], dtype=torch.float32, device=self.device)
        self._oracle_anchor_targets = torch.as_tensor(data["soft_targets"], dtype=torch.float32, device=self.device)

    def clear_oracle_anchor(self) -> None:
        self._oracle_anchor_states = None
        self._oracle_anchor_masks = None
        self._oracle_anchor_targets = None
        self.oracle_anchor_beta_init = 0.0
        self.oracle_anchor_beta_final = 0.0
        self.oracle_anchor_decay_fraction = 0.0

    def _current_oracle_anchor_beta(self) -> float:
        if (
            self._oracle_anchor_states is None
            or self._oracle_anchor_masks is None
            or self._oracle_anchor_targets is None
            or self.oracle_anchor_beta_init <= 0.0
            or self.oracle_anchor_decay_fraction <= 0.0
        ):
            return 0.0
        progress = float(1.0 - self._current_progress_remaining)
        if progress >= self.oracle_anchor_decay_fraction:
            return max(0.0, float(self.oracle_anchor_beta_final))
        ratio = progress / max(self.oracle_anchor_decay_fraction, 1.0e-8)
        beta = self.oracle_anchor_beta_init + ratio * (self.oracle_anchor_beta_final - self.oracle_anchor_beta_init)
        return max(0.0, float(beta))

    def _oracle_anchor_kl(self, batch_size: int) -> torch.Tensor:
        if (
            self._oracle_anchor_states is None
            or self._oracle_anchor_masks is None
            or self._oracle_anchor_targets is None
            or self._oracle_anchor_states.shape[0] == 0
        ):
            return torch.zeros((), device=self.device)
        effective_batch_size = int(self.oracle_anchor_batch_size or batch_size)
        indices = torch.randint(
            low=0,
            high=int(self._oracle_anchor_states.shape[0]),
            size=(max(effective_batch_size, 1),),
            device=self.device,
        )
        states = self._oracle_anchor_states.index_select(0, indices)
        masks = self._oracle_anchor_masks.index_select(0, indices)
        targets = self._oracle_anchor_targets.index_select(0, indices)
        distribution = self.policy.get_distribution(
            {"state": states, "action_mask": masks},
            action_masks=masks.bool(),
        )
        probs = torch.clamp(distribution.distribution.probs, min=1.0e-8)
        log_probs = torch.log(probs)
        target_log_probs = torch.log(torch.clamp(targets, min=1.0e-8))
        return torch.sum(targets * (target_log_probs - log_probs), dim=1).mean()

    def _setup_model(self) -> None:
        """在 SB3 模型搭建完成后追加 cost critic 与自定义 buffer。"""
        super()._setup_model()
        if not isinstance(self.observation_space, gym.spaces.Dict):
            raise ValueError("MaskablePPOLagrangian requires Dict observation space")
        state_dim = int(self.observation_space.spaces["state"].shape[0])
        self.cost_critic = CostValueNet(state_dim).to(self.device)
        self.cost_optimizer = torch.optim.Adam(self.cost_critic.parameters(), lr=self.learning_rate)
        self.rollout_buffer = ConstrainedDictRolloutBuffer(
            buffer_size=self.n_steps,
            observation_space=self.observation_space,
            action_space=self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        """确保 cost critic 与其优化器可随模型一起保存/加载。"""
        state_dicts, tensors = super()._get_torch_save_params()
        state_dicts.extend(["cost_critic", "cost_optimizer"])
        return state_dicts, tensors

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: ConstrainedDictRolloutBuffer,
        n_rollout_steps: int,
        use_masking: bool = True,
    ) -> bool:
        """收集一轮 rollout。

        与标准 PPO 的区别：
        - 读取环境中的 `cost` 字段；
        - 读取 `action_mask` 并传给策略；
        - 同时缓存 reward value 和 cost value。
        """

        if env.num_envs != 1:
            raise ValueError("This implementation currently supports only one environment.")
        assert self._last_obs is not None
        self.policy.set_training_mode(False)
        rollout_buffer.reset()
        callback.on_rollout_start()
        n_steps = 0
        while n_steps < n_rollout_steps:
            with torch.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                action_masks = (
                    self._last_obs["action_mask"].astype(bool)
                    if use_masking
                    else np.ones_like(self._last_obs["action_mask"], dtype=bool)
                )
                actions, values, log_probs = self.policy(obs_tensor, action_masks=action_masks)
                # 独立的 cost critic 只看连续状态，不消费 action mask。
                cost_values = self.cost_critic(obs_tensor["state"]).flatten()
            actions_np = actions.cpu().numpy()
            new_obs, rewards, dones, infos = env.step(actions_np)
            costs = np.asarray([info.get("cost", 0.0) for info in infos], dtype=np.float32)
            constraint_costs = np.asarray(
                [info.get("high_chunk_normalized_cost", info.get("cost", 0.0)) for info in infos],
                dtype=np.float32,
            )
            self._update_info_buffer(infos, dones)
            reward_batch = rewards.astype(np.float32)
            cost_batch = costs.astype(np.float32)
            if self.reward_normalization:
                self.reward_rms.update(reward_batch)
                reward_batch = self.reward_rms.normalize(reward_batch)
            if self.cost_normalization:
                self.cost_rms.update(cost_batch)
                cost_batch = self.cost_rms.normalize(cost_batch)
            rollout_buffer.add(
                obs=self._last_obs,
                action=actions_np,
                reward=float(reward_batch[0]),
                cost=float(cost_batch[0]),
                raw_cost=float(costs[0]),
                constraint_cost=float(constraint_costs[0]),
                episode_start=bool(self._last_episode_starts[0]),
                value=values.flatten(),
                cost_value=cost_values.flatten(),
                log_prob=log_probs.flatten(),
                action_mask=action_masks,
            )
            self.num_timesteps += env.num_envs
            self._last_obs = new_obs
            self._last_episode_starts = dones
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False
            n_steps += 1
        with torch.no_grad():
            obs_tensor = obs_as_tensor(self._last_obs, self.device)
            last_values = self.policy.predict_values(obs_tensor)
            last_cost_values = self.cost_critic(obs_tensor["state"]).flatten()
        rollout_buffer.compute_returns_and_advantage(last_values, last_cost_values, dones)
        callback.update_locals(locals())
        callback.on_rollout_end()
        return True

    def train(self) -> None:
        """执行 PPO-Lagrangian 参数更新。

        关键点：
        - reward critic 使用 `returns` 回归；
        - cost critic 使用 `cost_returns` 回归；
        - actor 使用 `A_r - lambda * A_c`；
        - `lambda` 根据约束违背程度单调上升/缓慢回落，并截断到 `[0, lambda_max]`。
        """

        self.policy.set_training_mode(True)
        clip_range = self.clip_range(self._current_progress_remaining) if callable(self.clip_range) else self.clip_range
        entropy_losses: list[float] = []
        pg_losses: list[float] = []
        value_losses: list[float] = []
        cost_value_losses: list[float] = []
        clip_fractions: list[float] = []
        oracle_anchor_kls: list[float] = []
        oracle_anchor_betas: list[float] = []

        for _ in range(self.n_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations,
                    rollout_data.actions,
                    action_masks=rollout_data.action_masks.detach().cpu().numpy().astype(bool),
                )
                values = values.flatten()
                advantages = rollout_data.advantages
                cost_advantages = rollout_data.cost_advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1.0e-8)
                    cost_advantages = (cost_advantages - cost_advantages.mean()) / (cost_advantages.std() + 1.0e-8)
                # 真正的 Lagrangian actor 更新核心，不是把 cost 粗暴并入 reward。
                lagrangian_advantage = advantages - float(self.lambda_value) * cost_advantages
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = lagrangian_advantage * ratio
                policy_loss_2 = lagrangian_advantage * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
                value_loss = F.mse_loss(rollout_data.returns, values)
                entropy_loss = -torch.mean(entropy if entropy is not None else -log_prob)
                oracle_anchor_beta = self._current_oracle_anchor_beta()
                oracle_anchor_kl = torch.zeros((), device=self.device)
                if oracle_anchor_beta > 0.0:
                    oracle_anchor_kl = self._oracle_anchor_kl(batch_size=int(rollout_data.actions.shape[0]))

                self.policy.optimizer.zero_grad()
                total_policy_loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                    + float(oracle_anchor_beta) * oracle_anchor_kl
                )
                total_policy_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

                cost_values = self.cost_critic(rollout_data.observations["state"]).flatten()
                cost_value_loss = F.mse_loss(rollout_data.cost_returns, cost_values)
                self.cost_optimizer.zero_grad()
                (self.cost_vf_coef * cost_value_loss).backward()
                nn.utils.clip_grad_norm_(self.cost_critic.parameters(), self.max_grad_norm)
                self.cost_optimizer.step()

                approx_kl = torch.mean((rollout_data.old_log_prob - log_prob).detach()).cpu().numpy()
                clip_fraction = torch.mean((torch.abs(ratio - 1.0) > clip_range).float()).item()
                entropy_losses.append(float(entropy_loss.item()))
                pg_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                cost_value_losses.append(float(cost_value_loss.item()))
                clip_fractions.append(float(clip_fraction))
                oracle_anchor_kls.append(float(oracle_anchor_kl.item()))
                oracle_anchor_betas.append(float(oracle_anchor_beta))
                if self.target_kl is not None and approx_kl > 1.5 * self.target_kl:
                    break

        mean_raw_cost = float(np.mean(self.rollout_buffer.raw_costs))
        mean_constraint_cost = float(np.mean(self.rollout_buffer.constraint_costs))
        self.lambda_value = float(
            np.clip(
                self.lambda_value + self.lagrangian_lr * (mean_constraint_cost - self.cost_limit),
                0.0,
                self.lambda_max,
            )
        )
        explained_var = explained_variance(self.rollout_buffer.values, self.rollout_buffer.returns)
        cost_explained_var = explained_variance(self.rollout_buffer.cost_values, self.rollout_buffer.cost_returns)

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/cost_value_loss", np.mean(cost_value_losses))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/cost_explained_variance", cost_explained_var)
        self.logger.record("train/lagrangian_lambda", self.lambda_value)
        if oracle_anchor_betas:
            self.logger.record("train/oracle_anchor_beta", float(np.mean(oracle_anchor_betas)))
            self.logger.record("train/oracle_anchor_kl", float(np.mean(oracle_anchor_kls)))
        self.logger.record("train/mean_rollout_cost", mean_constraint_cost)
        self.logger.record("train/mean_rollout_discounted_cost", mean_raw_cost)
        self.logger.record("train/mean_rollout_normalized_cost", mean_constraint_cost)
        self.logger.record("train/constraint_violation", max(0.0, mean_constraint_cost - self.cost_limit))

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "MaskablePPOLagrangian",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> "MaskablePPOLagrangian":
        """保持 SB3 learn 接口不变，方便 runner 与脚本直接调用。"""
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
