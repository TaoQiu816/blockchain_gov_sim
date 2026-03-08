"""DVRC-like baseline。"""

from __future__ import annotations

from typing import Any

import numpy as np

from gov_sim.baselines import BaselineBase, quantile_threshold


class DVRCLikeBaseline(BaselineBase):
    """按推荐污染和合谋风险动态调节信誉门槛与共识参数。"""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config=config, name="DVRC-like")

    def select_action(self, env: Any, obs: dict[str, Any]) -> int:
        self._set_committee_method(env, "soft_sortition")
        state = env.unwrapped.get_governance_state()
        snapshot = state["snapshot"]
        theta = quantile_threshold(snapshot.final_scores, 0.7)
        rec_penalty = float(np.mean(snapshot.rec_penalty))
        collusion = float(np.mean(snapshot.collusion_penalty))
        risk = rec_penalty + collusion
        m = 9 if risk < 0.35 else 13
        b = 384 if risk < 0.35 else 256
        tau = 40 if risk < 0.35 else 60
        theta += 0.1 * min(risk, 1.0)
        return self._nearest_action(m, b, tau, theta)
