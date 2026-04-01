"""两时标分层治理器的最终动作规格。"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from gov_sim.constants import B_CHOICES, M_CHOICES, TAU_CHOICES, THETA_CHOICES
from gov_sim.env.action_codec import ActionCodec, GovernanceAction

HIGH_LEVEL_M_VALUES: tuple[int, ...] = (5, 7, 9)
HIGH_LEVEL_THETA_VALUES: tuple[float, ...] = (0.45, 0.50, 0.55, 0.60)
LOW_LEVEL_B_VALUES: tuple[int, ...] = (256, 320, 384, 448, 512)
LOW_LEVEL_TAU_VALUES: tuple[int, ...] = (40, 60, 80, 100)

HIGH_LEVEL_TEMPLATES: tuple[tuple[int, float], ...] = tuple(
    (m, theta) for m, theta in product(HIGH_LEVEL_M_VALUES, HIGH_LEVEL_THETA_VALUES)
)
EXECUTABLE_HIGH_LEVEL_TEMPLATES: tuple[tuple[int, float], ...] = HIGH_LEVEL_TEMPLATES
LOW_LEVEL_ACTIONS: tuple[tuple[int, int], ...] = tuple(
    (b, tau) for b, tau in product(LOW_LEVEL_B_VALUES, LOW_LEVEL_TAU_VALUES)
)
HIGH_LEVEL_DIM = len(HIGH_LEVEL_TEMPLATES)
LOW_LEVEL_DIM = len(LOW_LEVEL_ACTIONS)
DEFAULT_HIGH_UPDATE_INTERVAL = 10

if tuple(HIGH_LEVEL_M_VALUES) != tuple(M_CHOICES):
    raise RuntimeError("Flat and hierarchical committee size domains diverged.")
if tuple(HIGH_LEVEL_THETA_VALUES) != tuple(THETA_CHOICES):
    raise RuntimeError("Flat and hierarchical trust threshold domains diverged.")
if LOW_LEVEL_B_VALUES != tuple(B_CHOICES):
    raise RuntimeError("Flat and hierarchical batch size domains diverged.")
if LOW_LEVEL_TAU_VALUES != tuple(TAU_CHOICES):
    raise RuntimeError("Flat and hierarchical timeout domains diverged.")


@dataclass(frozen=True)
class HighLevelAction:
    """高层模板动作。"""

    m: int
    theta: float


@dataclass(frozen=True)
class LowLevelAction:
    """低层执行动作。"""

    b: int
    tau: int


class HierarchicalActionCodec:
    """高层/低层动作与 flat 联合动作之间的映射。"""

    def __init__(self) -> None:
        self.flat_codec = ActionCodec()
        self.high_actions = tuple(HighLevelAction(m=m, theta=theta) for m, theta in HIGH_LEVEL_TEMPLATES)
        self.low_actions = tuple(LowLevelAction(b=b, tau=tau) for b, tau in LOW_LEVEL_ACTIONS)

    @property
    def high_dim(self) -> int:
        return HIGH_LEVEL_DIM

    @property
    def low_dim(self) -> int:
        return LOW_LEVEL_DIM

    def decode_high(self, action_idx: int) -> HighLevelAction:
        if not 0 <= int(action_idx) < self.high_dim:
            raise ValueError(f"high-level action_idx must be in [0, {self.high_dim}), got {action_idx}")
        return self.high_actions[int(action_idx)]

    def decode_low(self, action_idx: int) -> LowLevelAction:
        if not 0 <= int(action_idx) < self.low_dim:
            raise ValueError(f"low-level action_idx must be in [0, {self.low_dim}), got {action_idx}")
        return self.low_actions[int(action_idx)]

    def encode_high(self, action: HighLevelAction) -> int:
        return self.high_actions.index(action)

    def encode_low(self, action: LowLevelAction) -> int:
        return self.low_actions.index(action)

    def is_nominal_template(self, action: HighLevelAction) -> bool:
        """检查动作是否在高层模板列表中。"""
        return action in self.high_actions

    def to_governance_action(self, high_action: HighLevelAction, low_action: LowLevelAction) -> GovernanceAction:
        return GovernanceAction(m=high_action.m, b=low_action.b, tau=low_action.tau, theta=high_action.theta)

    def flat_index(self, high_action: HighLevelAction, low_action: LowLevelAction) -> int:
        return self.flat_codec.encode(self.to_governance_action(high_action=high_action, low_action=low_action))

    def split_governance_action(self, action: GovernanceAction) -> tuple[HighLevelAction, LowLevelAction]:
        return HighLevelAction(m=action.m, theta=action.theta), LowLevelAction(b=action.b, tau=action.tau)

    @staticmethod
    def high_action_repr(action: HighLevelAction | None) -> str:
        if action is None:
            return ""
        return f"{int(action.m)}|{float(action.theta):.2f}"

    @staticmethod
    def low_action_repr(action: LowLevelAction | None) -> str:
        if action is None:
            return ""
        return f"{int(action.b)}|{int(action.tau)}"
