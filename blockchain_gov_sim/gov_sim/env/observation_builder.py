"""观测构造模块。

为避免把每个 RSU 的高维逐节点特征直接塞进观测，主环境仅输出统计摘要：
- 负载、队列、RTT、churn、资格规模；
- 总信誉分布分位数/直方图；
- 四维 `mu` 分布的均值与分位数；
- 上一步动作。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from gov_sim.constants import REPUTATION_DIMS
from gov_sim.env.action_codec import GovernanceAction
from gov_sim.modules.reputation_model import ReputationSnapshot
from gov_sim.utils.math_utils import histogram_summary, quantile_summary


def build_state_vector(
    summary: dict[str, float],
    snapshot: ReputationSnapshot,
    prev_action: GovernanceAction,
    bins: int,
) -> np.ndarray:
    """构造用于策略网络输入的连续状态向量。"""

    trust_scores = snapshot.final_scores
    state_parts: list[np.ndarray] = [
        np.asarray(
            [
                summary["A_e"],
                summary["Q_e"],
                summary["L_bar_e"],
                summary["RTT_e"],
                summary["chi_e"],
                summary["eligible_size"],
            ],
            dtype=np.float32,
        ),
        quantile_summary(trust_scores),
        histogram_summary(trust_scores, bins=bins),
    ]
    for dim in REPUTATION_DIMS:
        mu = snapshot.mu[dim]
        state_parts.append(np.asarray([float(np.mean(mu))], dtype=np.float32))
        state_parts.append(quantile_summary(mu, quantiles=(0.25, 0.5, 0.75)))
    state_parts.append(np.asarray([prev_action.m / 15.0, prev_action.b / 512.0, prev_action.tau / 80.0, prev_action.theta], dtype=np.float32))
    return np.concatenate(state_parts, axis=0).astype(np.float32)
