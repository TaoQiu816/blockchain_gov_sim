"""动态动作掩码模块。

主方案要求：
- 动作为 400 维扁平离散动作；
- 非法动作必须在采样前就通过 action mask 屏蔽，而不是只靠 reward 罚项。

本文件负责对每个候选动作逐一判定合法性，并生成长度为 400 的 mask。
"""

from __future__ import annotations

import math

import numpy as np

from gov_sim.constants import ACTION_DIM
from gov_sim.env.action_codec import ActionCodec, GovernanceAction


def is_action_legal(
    action: GovernanceAction,
    prev_action: GovernanceAction,
    trust_scores: np.ndarray,
    uptime: np.ndarray,
    online: np.ndarray,
    u_min: float,
    delta_m_max: int,
    delta_b_max: int,
    delta_tau_max: int,
    delta_theta_max: float,
    unsafe_guard: bool,
    h_min: float,
) -> bool:
    """判断单个治理动作在当前状态下是否合法。

    非法条件包括：
    1. 资格集规模不足；
    2. 违反平滑约束；
    3. 在当前信誉分布下明显触发最低安全门槛。

    这里的“安全门槛”采用保守近似：
    对符合资格集的节点，将其信誉分数和视为“估计诚实质量”。
    如果该质量都无法支撑 `ceil(h_min * m)`，则提前屏蔽该动作。
    """

    eligible = (trust_scores >= action.theta) & (uptime >= u_min) & (online == 1)
    eligible_size = int(np.sum(eligible))
    if action.m > eligible_size:
        return False
    if abs(action.m - prev_action.m) > delta_m_max:
        return False
    if abs(action.b - prev_action.b) > delta_b_max:
        return False
    if abs(action.tau - prev_action.tau) > delta_tau_max:
        return False
    if abs(action.theta - prev_action.theta) > delta_theta_max + 1.0e-8:
        return False
    estimated_honest_mass = float(np.sum(trust_scores[eligible]))
    if unsafe_guard and estimated_honest_mass < math.ceil(h_min * action.m):
        return False
    return True


def build_action_mask(
    codec: ActionCodec,
    prev_action: GovernanceAction,
    trust_scores: np.ndarray,
    uptime: np.ndarray,
    online: np.ndarray,
    u_min: float,
    delta_m_max: int,
    delta_b_max: int,
    delta_tau_max: int,
    delta_theta_max: float,
    unsafe_guard: bool,
    h_min: float,
) -> np.ndarray:
    """枚举全部 400 个动作并生成动态 mask。"""

    mask = np.zeros(ACTION_DIM, dtype=np.int8)
    for action_idx in range(ACTION_DIM):
        action = codec.decode(action_idx)
        mask[action_idx] = int(
            is_action_legal(
                action=action,
                prev_action=prev_action,
                trust_scores=trust_scores,
                uptime=uptime,
                online=online,
                u_min=u_min,
                delta_m_max=delta_m_max,
                delta_b_max=delta_b_max,
                delta_tau_max=delta_tau_max,
                delta_theta_max=delta_theta_max,
                unsafe_guard=unsafe_guard,
                h_min=h_min,
            )
        )
    # 极端情况下如果所有动作都被屏蔽，至少保留上一步动作，避免策略完全无动作可选。
    if mask.sum() == 0:
        fallback = codec.encode(prev_action)
        mask[fallback] = 1
    return mask
