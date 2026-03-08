"""策略网络相关封装。

主环境观测是 `Dict(state, action_mask)`，但真正进入策略 trunk 的只应是 `state`。
因此这里提供自定义 features extractor，把 `action_mask` 留给 MaskablePPO 的分布层处理。
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class StateOnlyExtractor(BaseFeaturesExtractor):
    """仅提取连续状态向量，不把 action mask 作为网络输入特征。"""

    def __init__(self, observation_space: gym.spaces.Dict) -> None:
        state_space = observation_space.spaces["state"]
        if len(state_space.shape) != 1:
            raise ValueError("state observation must be a flat vector")
        self.state_dim = int(state_space.shape[0])
        super().__init__(observation_space, features_dim=self.state_dim)

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        return observations["state"]


class CostValueNet(nn.Module):
    """独立 cost critic。

    之所以不复用 reward critic 的最后一层，是因为 reward 与 cost 的目标不同，
    独立建模可以减少 value leakage。
    """

    def __init__(self, input_dim: int, hidden_dims: tuple[int, int] = (256, 256)) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1),
        )

    def forward(self, state_tensor: torch.Tensor) -> torch.Tensor:
        return self.net(state_tensor)


def resolve_policy_kwargs(policy_kwargs: dict[str, Any] | None) -> dict[str, Any]:
    """把 YAML 中易写的字符串配置转成 SB3 可接受的 policy kwargs。"""
    kwargs = dict(policy_kwargs or {})
    kwargs["features_extractor_class"] = StateOnlyExtractor
    activation_name = kwargs.get("activation_fn")
    if isinstance(activation_name, str):
        activation_map = {
            "ReLU": nn.ReLU,
            "Tanh": nn.Tanh,
            "ELU": nn.ELU,
        }
        if activation_name not in activation_map:
            raise ValueError(f"Unsupported activation_fn: {activation_name}")
        kwargs["activation_fn"] = activation_map[activation_name]
    return kwargs
