"""实验编排公共工具。

该文件把环境、模型、baseline、输出目录和配置覆盖拼装起来，
为 train/eval/benchmark/ablation 四类 runner 提供统一底座。
"""

from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from gov_sim.agent.masked_ppo_lagrangian import MaskablePPOLagrangian
from gov_sim.agent.policy_wrappers import resolve_policy_kwargs
from gov_sim.baselines import BASELINE_REGISTRY, BaselinePolicy
from gov_sim.env.gov_env import BlockchainGovEnv
from gov_sim.modules.metrics_tracker import MetricsTracker
from gov_sim.utils.device import resolve_device
from gov_sim.utils.io import deep_update, ensure_dir


def _maybe_tensorboard_log(path: str | None) -> str | None:
    """只有本地安装了 tensorboard 时才启用其日志目录。"""
    if path is None:
        return None
    return path if importlib.util.find_spec("tensorboard") is not None else None


def make_env(config: dict[str, Any]) -> BlockchainGovEnv:
    """创建一个独立环境实例。"""
    return BlockchainGovEnv(config=deepcopy(config))


def make_vec_env(config: dict[str, Any]) -> DummyVecEnv:
    """创建单环境 VecEnv，方便直接复用 SB3 接口。"""
    return DummyVecEnv([lambda: Monitor(make_env(config))])


def build_model(
    config: dict[str, Any],
    env: DummyVecEnv,
    use_lagrangian: bool = True,
) -> MaskablePPOLagrangian | MaskablePPO | PPO:
    """根据配置创建 RL 控制器。"""
    agent_cfg = config["agent"]
    resolved_device = resolve_device(agent_cfg.get("device"))
    common_kwargs = dict(
        policy=agent_cfg["policy"],
        env=env,
        learning_rate=agent_cfg["learning_rate"],
        n_steps=agent_cfg["n_steps"],
        batch_size=agent_cfg["batch_size"],
        n_epochs=agent_cfg["n_epochs"],
        gamma=agent_cfg["gamma"],
        gae_lambda=agent_cfg["gae_lambda"],
        clip_range=agent_cfg["clip_range"],
        ent_coef=agent_cfg["ent_coef"],
        vf_coef=agent_cfg["vf_coef"],
        max_grad_norm=agent_cfg["max_grad_norm"],
        policy_kwargs=resolve_policy_kwargs(agent_cfg.get("policy_kwargs")),
        device=resolved_device,
        tensorboard_log=_maybe_tensorboard_log(agent_cfg.get("tensorboard_log")),
        verbose=0,
    )
    if use_lagrangian:
        return MaskablePPOLagrangian(
            **common_kwargs,
            cost_vf_coef=agent_cfg["cost_vf_coef"],
            lagrangian_lr=agent_cfg["lagrangian_lr"],
            cost_limit=agent_cfg["cost_limit"],
            lambda_init=agent_cfg["lambda_init"],
            lambda_max=agent_cfg["lambda_max"],
            reward_normalization=bool(config["env"]["reward_normalization"]),
            cost_normalization=bool(config["env"]["cost_normalization"]),
        )
    return MaskablePPO(**common_kwargs)


def build_plain_ppo(config: dict[str, Any], env: DummyVecEnv) -> PPO:
    """构建无约束普通 PPO，用于 ablation。"""
    agent_cfg = config["agent"]
    resolved_device = resolve_device(agent_cfg.get("device"))
    return PPO(
        policy=agent_cfg["policy"],
        env=env,
        learning_rate=agent_cfg["learning_rate"],
        n_steps=agent_cfg["n_steps"],
        batch_size=agent_cfg["batch_size"],
        n_epochs=agent_cfg["n_epochs"],
        gamma=agent_cfg["gamma"],
        gae_lambda=agent_cfg["gae_lambda"],
        clip_range=agent_cfg["clip_range"],
        ent_coef=agent_cfg["ent_coef"],
        vf_coef=agent_cfg["vf_coef"],
        max_grad_norm=agent_cfg["max_grad_norm"],
        policy_kwargs=resolve_policy_kwargs(agent_cfg.get("policy_kwargs")),
        device=resolved_device,
        tensorboard_log=_maybe_tensorboard_log(agent_cfg.get("tensorboard_log")),
        verbose=0,
    )


def instantiate_baseline(name: str, config: dict[str, Any]) -> BaselinePolicy:
    """实例化指定名称的 baseline。"""
    if name not in BASELINE_REGISTRY:
        raise ValueError(f"Unknown baseline: {name}")
    return BASELINE_REGISTRY[name](config)


def evaluate_controller(
    controller: Any,
    config: dict[str, Any],
    episodes: int,
    deterministic: bool,
    is_baseline: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """在原生环境上评估 RL 模型或 baseline。

    返回：
    - step 级 dataframe：便于后续画动作轨迹、队列恢复曲线；
    - summary：便于 benchmark / eval 汇总。
    """

    env = make_env(config)
    tracker = MetricsTracker()
    rows: list[dict[str, Any]] = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=int(config["seed"]) + episode)
        if is_baseline:
            controller.reset()
        done = False
        step_idx = 0
        while not done:
            if is_baseline:
                action = controller.select_action(env, obs)
            else:
                if isinstance(controller, PPO):
                    action, _ = controller.predict(obs, deterministic=deterministic)
                else:
                    action, _ = controller.predict(obs, deterministic=deterministic, action_masks=obs["action_mask"])
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            tracker.update(info)
            row = info.copy()
            row["episode"] = episode
            row["step"] = step_idx
            row["reward"] = reward
            rows.append(row)
            step_idx += 1
    dataframe = pd.DataFrame(rows)
    summary = tracker.summary()
    if not dataframe.empty:
        action_traj = dataframe[["m_e", "b_e", "tau_e", "theta_e"]].mean().to_dict()
        summary["action_trajectory_mean"] = {k: float(v) for k, v in action_traj.items()}
    return dataframe, summary


def prepare_output_dir(config: dict[str, Any], stage: str) -> Path:
    """准备某个实验阶段的输出目录。"""
    path = ensure_dir(Path(config["output_root"]) / stage / config["run_name"])
    return path


def apply_override(config: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """对基础配置做递归覆盖，常用于 benchmark / ablation。"""
    return deep_update(deepcopy(config), override)


def malicious_ratio_override(ratio: float) -> dict[str, Any]:
    """生成恶意节点渗透率覆盖配置。"""
    return {"env": {"malicious_ratio": ratio}}


def attack_override(on_off_period: int | None = None, zigzag_freq: float | None = None, collusion_group_size: int | None = None) -> dict[str, Any]:
    """生成攻击强度相关的覆盖配置。"""
    attack: dict[str, Any] = {}
    if on_off_period is not None:
        attack["on_off_period"] = on_off_period
    if zigzag_freq is not None:
        attack["zigzag_freq"] = zigzag_freq
    if collusion_group_size is not None:
        attack["collusion_group_size"] = collusion_group_size
    return {"scenario": {"attack": attack}}
