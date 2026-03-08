"""联合信任-共识联动 baseline。"""

from __future__ import annotations

from typing import Any

import numpy as np

from gov_sim.baselines import BaselineBase, quantile_threshold


class JointTrustConsensusLikeBaseline(BaselineBase):
    """把整体信任风险和链侧拥塞同时映射为治理参数。"""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config=config, name="Joint-Trust-Consensus-like")

    def select_action(self, env: Any, obs: dict[str, Any]) -> int:
        self._set_committee_method(env, "soft_sortition")
        state = env.unwrapped.get_governance_state()
        queue = float(state["queue_size"])
        trust = state["snapshot"].final_scores
        risk = 1.0 - float(np.mean(trust))
        theta = quantile_threshold(trust, 0.6 + 0.2 * risk)
        m = 9 if risk < 0.25 else 13
        b = 384 if queue > 150 else 256
        tau = 60 if queue > 220 else 40
        return self._nearest_action(m, b, tau, theta)
