"""动作 mask 测试。"""

from __future__ import annotations

import numpy as np

from gov_sim.env.action_codec import ActionCodec, GovernanceAction
from gov_sim.env.action_mask import build_action_mask


def test_action_mask_blocks_oversized_committee() -> None:
    """当资格集不足时，大委员会动作必须被屏蔽。"""
    codec = ActionCodec()
    prev = GovernanceAction(m=11, b=256, tau=40, theta=0.6)
    trust = np.array([0.9, 0.85, 0.82, 0.78, 0.75], dtype=np.float32)
    uptime = np.ones(5, dtype=np.float32)
    online = np.ones(5, dtype=np.int8)
    mask = build_action_mask(
        codec=codec,
        prev_action=prev,
        trust_scores=trust,
        uptime=uptime,
        online=online,
        u_min=0.5,
        delta_m_max=10,
        delta_b_max=512,
        delta_tau_max=100,
        delta_theta_max=1.0,
        unsafe_guard=False,
        h_min=2 / 3,
    )
    illegal = codec.encode(GovernanceAction(m=15, b=256, tau=40, theta=0.4))
    assert mask[illegal] == 0


def test_action_mask_has_fallback_when_all_illegal() -> None:
    """即使所有动作都被屏蔽，也必须至少保留上一步动作。"""

    codec = ActionCodec()
    prev = GovernanceAction(m=11, b=256, tau=40, theta=0.8)
    trust = np.array([0.1, 0.1], dtype=np.float32)
    uptime = np.zeros(2, dtype=np.float32)
    online = np.zeros(2, dtype=np.int8)
    mask = build_action_mask(
        codec=codec,
        prev_action=prev,
        trust_scores=trust,
        uptime=uptime,
        online=online,
        u_min=0.9,
        delta_m_max=0,
        delta_b_max=0,
        delta_tau_max=0,
        delta_theta_max=0.0,
        unsafe_guard=True,
        h_min=2 / 3,
    )
    assert mask.sum() == 1
    assert mask[codec.encode(prev)] == 1
