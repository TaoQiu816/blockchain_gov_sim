"""治理动作编解码器。"""

from __future__ import annotations

from dataclasses import dataclass
import math

from gov_sim.constants import ACTION_DIM, B_CHOICES, M_MIN, RHO_M_CHOICES, TAU_CHOICES, THETA_CHOICES


@dataclass(frozen=True)
class GovernanceAction:
    """链侧治理动作语义表示。"""

    rho_m: float
    theta: float
    b: int
    tau: int


class ActionCodec:
    """扁平索引 <-> 联合动作四元组 的双向编解码器。"""

    def __init__(self) -> None:
        self.rho_m_values = RHO_M_CHOICES
        self.b_values = B_CHOICES
        self.tau_values = TAU_CHOICES
        self.theta_values = THETA_CHOICES

    def decode(self, action_idx: int) -> GovernanceAction:
        """把离散索引解码为 `(rho_m, theta, b, tau)`。"""
        if not 0 <= action_idx < ACTION_DIM:
            raise ValueError(f"action_idx must be in [0, {ACTION_DIM}), got {action_idx}")
        idx = action_idx
        theta_idx = idx % len(self.theta_values)
        idx //= len(self.theta_values)
        tau_idx = idx % len(self.tau_values)
        idx //= len(self.tau_values)
        b_idx = idx % len(self.b_values)
        idx //= len(self.b_values)
        rho_m_idx = idx
        return GovernanceAction(
            rho_m=self.rho_m_values[rho_m_idx],
            theta=self.theta_values[theta_idx],
            b=self.b_values[b_idx],
            tau=self.tau_values[tau_idx],
        )

    def encode(self, action: GovernanceAction) -> int:
        """把合法动作四元组编码为离散动作索引。"""
        try:
            rho_m_idx = self.rho_m_values.index(action.rho_m)
            theta_idx = self.theta_values.index(action.theta)
            b_idx = self.b_values.index(action.b)
            tau_idx = self.tau_values.index(action.tau)
        except ValueError as exc:
            raise ValueError(f"Illegal governance action: {action}") from exc
        idx = rho_m_idx
        idx = idx * len(self.b_values) + b_idx
        idx = idx * len(self.tau_values) + tau_idx
        idx = idx * len(self.theta_values) + theta_idx
        return int(idx)

    @staticmethod
    def mapped_committee_size(rho_m: float, eligible_size: int, m_min: int = M_MIN) -> int:
        """按冻结规格把比例型治理动作映射为委员会规模。"""

        if eligible_size < m_min:
            return 0
        proposed = int(math.ceil(float(rho_m) * int(eligible_size)))
        return max(int(m_min), min(int(eligible_size), proposed))

    def fallback_action(self, eligible_counts_by_theta: dict[float, int]) -> tuple[int, GovernanceAction]:
        """所有动作均非法时使用冻结版 deterministic fallback。"""

        theta = min(
            self.theta_values,
            key=lambda value: (-int(eligible_counts_by_theta.get(float(value), 0)), float(value)),
        )
        action = GovernanceAction(
            rho_m=min(self.rho_m_values),
            theta=float(theta),
            b=384,
            tau=80,
        )
        return self.encode(action), action
