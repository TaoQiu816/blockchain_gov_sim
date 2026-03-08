"""Baseline 策略集合。

这些 baseline 都是在统一仿真框架下的“机制抽象版”：
- 不追求逐论文 1:1 工程复刻；
- 但必须保留各方法最核心的控制思想；
- 并且全部输出同一动作空间里的离散治理动作，保证公平比较。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from gov_sim.constants import B_CHOICES, M_CHOICES, TAU_CHOICES, THETA_CHOICES
from gov_sim.env.action_codec import ActionCodec, GovernanceAction


class BaselinePolicy(Protocol):
    """所有 baseline 的统一协议。"""

    name: str

    def reset(self) -> None:
        ...

    def select_action(self, env: Any, obs: dict[str, np.ndarray]) -> int:
        ...


@dataclass
class BaselineBase:
    """baseline 基类，提供公共工具函数。"""

    config: dict[str, Any]
    name: str
    codec: ActionCodec = ActionCodec()

    def reset(self) -> None:
        return None

    def _set_committee_method(self, env: Any, method: str) -> None:
        """允许 baseline 显式指定委员会机制。

        这样可以让 Top-K 类 baseline 与主方案的 soft sortition 真正区分开来，
        而不是只换个名字却还在用同一套委员会生成方式。
        """

        env.unwrapped.committee_override_method = method

    def _nearest_action(self, m: int, b: int, tau: int, theta: float) -> int:
        """把连续启发式控制量投影到离散动作集合。"""
        action = GovernanceAction(
            m=min(M_CHOICES, key=lambda x: abs(x - m)),
            b=min(B_CHOICES, key=lambda x: abs(x - b)),
            tau=min(TAU_CHOICES, key=lambda x: abs(x - tau)),
            theta=min(THETA_CHOICES, key=lambda x: abs(x - theta)),
        )
        return self.codec.encode(action)


def quantile_threshold(values: np.ndarray, q: float) -> float:
    """按分位数选取信誉门槛。"""
    if values.size == 0:
        return THETA_CHOICES[0]
    return float(np.quantile(values, q))


from gov_sim.baselines.ae_pbft_like import AEPBFTLikeBaseline
from gov_sim.baselines.dvrc_like import DVRCLikeBaseline
from gov_sim.baselines.heuristic_aimd import HeuristicAIMDBaseline
from gov_sim.baselines.joint_trust_consensus_like import JointTrustConsensusLikeBaseline
from gov_sim.baselines.multirep_topk_static import MultiRepTopKStaticBaseline
from gov_sim.baselines.single_rep_topk_static import SingleRepTopKStaticBaseline
from gov_sim.baselines.static_param import StaticParamBaseline
from gov_sim.baselines.two_layer_lpbft_like import TwoLayerLPBFTLikeBaseline

BASELINE_REGISTRY = {
    "Static-Param": StaticParamBaseline,
    "Heuristic-AIMD": HeuristicAIMDBaseline,
    "SingleRep-TopK-Static": SingleRepTopKStaticBaseline,
    "MultiRep-TopK-Static": MultiRepTopKStaticBaseline,
    "Joint-Trust-Consensus-like": JointTrustConsensusLikeBaseline,
    "AE-PBFT-like": AEPBFTLikeBaseline,
    "DVRC-like": DVRCLikeBaseline,
    "TwoLayer-LPBFT-like": TwoLayerLPBFTLikeBaseline,
}
