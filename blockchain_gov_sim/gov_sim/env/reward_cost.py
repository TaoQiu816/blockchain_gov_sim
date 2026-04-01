"""奖励与成本函数。"""

from __future__ import annotations

from typing import Any

import numpy as np

from gov_sim.env.action_codec import GovernanceAction


def compute_reward(
    config: dict[str, Any],
    served: float,
    latency: float,
    queue_next: float,
    batch_size: int,
    action: GovernanceAction,
    prev_action: GovernanceAction,
) -> tuple[float, dict[str, float]]:
    """计算冻结版主任务 reward。"""

    smooth = (
        abs(float(action.rho_m) - float(prev_action.rho_m))
        + abs(float(action.theta) - float(prev_action.theta))
        + abs(int(action.b) - int(prev_action.b))
        + abs(int(action.tau) - int(prev_action.tau))
    )
    block_slack = max(float(batch_size) - float(served), 0.0) / max(float(batch_size), 1.0)
    terms = {
        "throughput": 1.0 * float(np.log1p(served)),
        "latency": -0.01 * float(latency),
        "queue": -0.001 * float(queue_next),
        "smooth": -0.001 * float(smooth),
        "block_slack": -0.01 * float(block_slack),
    }
    return float(sum(terms.values())), terms


def compute_cost(config: dict[str, Any], unsafe: int, timeout_failure: int = 0, margin_cost: float = 0.0) -> tuple[float, dict[str, float]]:
    """计算冻结版安全成本。"""

    terms = {
        "unsafe": float(unsafe),
        "timeout_failure": 0.12 * float(timeout_failure),
        "margin_cost": 0.15 * float(margin_cost),
    }
    return float(sum(terms.values())), terms
