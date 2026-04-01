"""动态动作掩码模块。"""

from __future__ import annotations

import numpy as np

from gov_sim.constants import ACTION_DIM, M_MIN
from gov_sim.env.action_codec import ActionCodec, GovernanceAction


def is_action_legal(
    action: GovernanceAction,
    eligible_size: int,
    m_min: int = M_MIN,
) -> bool:
    """资格节点数足以支撑映射后的委员会规模时动作合法。"""

    mapped_m = ActionCodec.mapped_committee_size(action.rho_m, int(eligible_size), m_min=int(m_min))
    return mapped_m >= int(m_min) and int(eligible_size) >= mapped_m


def build_action_mask(
    codec: ActionCodec,
    eligible_counts_by_theta: dict[float, int],
    m_min: int = M_MIN,
) -> np.ndarray:
    """枚举全部联合动作并生成动态 mask。"""

    mask = np.zeros(ACTION_DIM, dtype=np.int8)
    for action_idx in range(ACTION_DIM):
        action = codec.decode(action_idx)
        eligible_size = int(eligible_counts_by_theta.get(float(action.theta), 0))
        mask[action_idx] = int(is_action_legal(action=action, eligible_size=eligible_size, m_min=m_min))
    return mask


def resolve_action_with_fallback(
    codec: ActionCodec,
    requested_idx: int,
    mask: np.ndarray,
    eligible_counts_by_theta: dict[float, int],
) -> tuple[GovernanceAction, GovernanceAction, bool, int]:
    """解析动作，并在必要时执行 deterministic fallback。"""

    requested = codec.decode(int(requested_idx))
    if mask[int(requested_idx)]:
        return requested, requested, False, 0
    if int(np.sum(mask)) == 0:
        _, fallback = codec.fallback_action(eligible_counts_by_theta)
        return requested, fallback, True, 1
    fallback_idx = int(np.flatnonzero(mask)[0])
    return requested, codec.decode(fallback_idx), True, 0
