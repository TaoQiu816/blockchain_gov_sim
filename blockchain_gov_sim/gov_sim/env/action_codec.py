"""治理动作编解码器。

第四章动作空间固定为扁平离散索引 `0..399`，但语义动作是四元组：
`a_e = (m_e, b_e, tau_e, theta_e)`。

本文件负责在“论文动作”与“RL 可消费的离散索引”之间做双向映射。
"""

from __future__ import annotations

from dataclasses import dataclass

from gov_sim.constants import ACTION_DIM, B_CHOICES, M_CHOICES, TAU_CHOICES, THETA_CHOICES


@dataclass(frozen=True)
class GovernanceAction:
    """链侧治理动作语义表示。"""

    m: int
    b: int
    tau: int
    theta: float


class ActionCodec:
    """扁平索引 <-> 联合动作四元组 的双向编解码器。"""

    def __init__(self) -> None:
        self.m_values = M_CHOICES
        self.b_values = B_CHOICES
        self.tau_values = TAU_CHOICES
        self.theta_values = THETA_CHOICES

    def decode(self, action_idx: int) -> GovernanceAction:
        """把离散索引解码为 `(m,b,tau,theta)`。"""
        if not 0 <= action_idx < ACTION_DIM:
            raise ValueError(f"action_idx must be in [0, {ACTION_DIM}), got {action_idx}")
        idx = action_idx
        theta_idx = idx % len(self.theta_values)
        idx //= len(self.theta_values)
        tau_idx = idx % len(self.tau_values)
        idx //= len(self.tau_values)
        b_idx = idx % len(self.b_values)
        idx //= len(self.b_values)
        m_idx = idx
        return GovernanceAction(
            m=self.m_values[m_idx],
            b=self.b_values[b_idx],
            tau=self.tau_values[tau_idx],
            theta=self.theta_values[theta_idx],
        )

    def encode(self, action: GovernanceAction) -> int:
        """把合法动作四元组编码为离散动作索引。"""
        try:
            m_idx = self.m_values.index(action.m)
            b_idx = self.b_values.index(action.b)
            tau_idx = self.tau_values.index(action.tau)
            theta_idx = self.theta_values.index(action.theta)
        except ValueError as exc:
            raise ValueError(f"Illegal governance action: {action}") from exc
        idx = m_idx
        idx = idx * len(self.b_values) + b_idx
        idx = idx * len(self.tau_values) + tau_idx
        idx = idx * len(self.theta_values) + theta_idx
        return int(idx)
