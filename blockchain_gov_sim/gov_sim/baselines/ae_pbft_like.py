"""AE-PBFT-like baseline。"""

from __future__ import annotations

from typing import Any

import numpy as np

from gov_sim.baselines import BaselineBase


class AEPBFTLikeBaseline(BaselineBase):
    """异常环境下偏保守的 PBFT 参数自适应。

    抽象保留的核心思想：
    - 环境越恶劣（高 RTT / 高 churn / 低信誉），越倾向扩大委员会和提高门槛；
    - 采用基于信誉的 Top-K 候选，强调“信用驱动的组建”。
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config=config, name="AE-PBFT-like")

    def select_action(self, env: Any, obs: dict[str, Any]) -> int:
        self._set_committee_method(env, "topk")
        state = env.unwrapped.get_governance_state()
        scenario = state["scenario"]
        snapshot = state["snapshot"]
        risk = scenario.churn + scenario.rtt / max(self.config["scenario"]["network"]["rtt_max"], 1.0) + (1.0 - float(np.mean(snapshot.final_scores)))
        theta = 0.8 if np.mean(snapshot.final_scores) < 0.6 else 0.7
        m = 15 if risk > 1.0 else 13
        b = 256 if risk > 1.0 else 384
        tau = 80 if risk > 1.0 else 60
        return self._nearest_action(m, b, tau, theta)
