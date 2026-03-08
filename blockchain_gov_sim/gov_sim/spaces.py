"""动作空间元信息。"""

from __future__ import annotations

from dataclasses import dataclass

from gov_sim.constants import ACTION_DIM, B_CHOICES, M_CHOICES, TAU_CHOICES, THETA_CHOICES


@dataclass(frozen=True)
class ActionSpec:
    """动作取值集合的结构化描述。"""

    m_values: tuple[int, ...] = M_CHOICES
    b_values: tuple[int, ...] = B_CHOICES
    tau_values: tuple[int, ...] = TAU_CHOICES
    theta_values: tuple[float, ...] = THETA_CHOICES

    @property
    def action_dim(self) -> int:
        """联合动作总数。"""
        return ACTION_DIM


DEFAULT_ACTION_SPEC = ActionSpec()
