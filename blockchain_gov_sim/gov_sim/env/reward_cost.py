"""奖励与成本函数。

主方案明确要求：
- reward 与 cost 分开建模；
- cost 不能只被吸收到 reward 里；
- PPO-Lagrangian 使用 reward advantage 与 cost advantage 分别训练。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from gov_sim.env.action_codec import GovernanceAction


def compute_reward(
    config: dict[str, Any],
    service_capacity: float,
    latency: float,
    queue_next: float,
    action: GovernanceAction,
    prev_action: GovernanceAction,
) -> tuple[float, dict[str, float]]:
    """计算主任务 reward。

    对应论文：
    `r_e = λ1 log(1+S_e) - λ2 L_bar_e - λ3 Q_{e+1} - λ4 ||a_e-a_{e-1}||_1`
    """

    rw = config["reward_weights"]
    smooth = abs(action.m - prev_action.m) + abs(action.b - prev_action.b) / 128.0 + abs(action.tau - prev_action.tau) / 20.0 + abs(
        action.theta - prev_action.theta
    ) / 0.1
    terms = {
        "throughput": float(rw["throughput"]) * float(np.log1p(service_capacity)),
        "latency": -float(rw["latency"]) * float(latency),
        "queue": -float(rw["queue"]) * float(queue_next),
        "smooth": -float(rw["smooth"]) * float(smooth),
    }
    return float(sum(terms.values())), terms


def compute_cost(config: dict[str, Any], unsafe: int, honest_ratio: float) -> tuple[float, dict[str, float]]:
    """计算安全成本。

    对应论文：
    `c_e = U_e + ν (1 - h_e)`
    """

    cw = config["cost_weights"]
    terms = {
        "unsafe": float(unsafe),
        "honest_deficit": float(cw["honest_penalty"]) * float(1.0 - honest_ratio),
    }
    return float(sum(terms.values())), terms
