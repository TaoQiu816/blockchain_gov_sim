"""TwoLayer-LPBFT-like baseline。"""

from __future__ import annotations

from typing import Any

import numpy as np

from gov_sim.baselines import BaselineBase


class TwoLayerLPBFTLikeBaseline(BaselineBase):
    """两层轻量 PBFT 抽象版。

    抽象思路：
    - 外层用更高信誉门槛筛出“骨干节点”；
    - 内层只保留较小委员会，降低控制开销；
    - 因此它更偏向高门槛、小委员会、短等待窗口。
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config=config, name="TwoLayer-LPBFT-like")

    def select_action(self, env: Any, obs: dict[str, Any]) -> int:
        self._set_committee_method(env, "topk")
        state = env.unwrapped.get_governance_state()
        trust = state["snapshot"].final_scores
        uptime = state["scenario"].uptime
        spread = float(np.quantile(trust, 0.75) - np.quantile(trust, 0.25))
        backbone_quality = float(np.mean(0.6 * trust + 0.4 * uptime))
        theta = 0.8 if spread > 0.25 or backbone_quality < 0.72 else 0.7
        m = 7 if backbone_quality > 0.8 else 9
        b = 256 if state["queue_size"] < 180 else 384
        tau = 40 if state["scenario"].churn < 0.15 else 60
        return self._nearest_action(m, b, tau, theta)
