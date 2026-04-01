"""Yadav-2025 baseline。"""

from __future__ import annotations

from typing import Any

import numpy as np

from gov_sim.baselines import BaselineBase, trust_entropy


class Yadav2025Baseline(BaselineBase):
    """基于信誉均值、熵、队列和 RTT 的 trust-based DPoS 启发式。"""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config=config, name="Yadav-2025")

    def select_action(self, env: Any, obs: np.ndarray, legal_mask: np.ndarray | None = None) -> int:
        del obs
        state = self._governance_state(env)
        snapshot = state["snapshot"]
        scenario = state["scenario"]
        queue = float(state["queue_size"])
        scores = np.asarray(snapshot.final_scores, dtype=np.float32)[np.asarray(snapshot.online_mask, dtype=np.int8) == 1]
        mean_trust = float(np.mean(scores)) if scores.size > 0 else 0.0
        entropy = float(trust_entropy(scores))
        high_queue = queue > 0.40 * float(env.queue_max)
        high_rtt = float(scenario.rtt) > 0.75 * float(env.rtt_max)

        if mean_trust < 0.55 or entropy > 0.80:
            theta = 0.60
            rho_m = 5.0 / 27.0
            b = 320
            tau = 100
        elif mean_trust < 0.65 or high_queue or high_rtt:
            theta = 0.55
            rho_m = 7.0 / 27.0
            b = 384
            tau = 80
        else:
            theta = 0.50
            rho_m = 9.0 / 27.0 if queue > 0.20 * float(env.queue_max) and not high_rtt else 7.0 / 27.0
            b = 384
            tau = 60 if not high_rtt else 80
        return self._legal_action(legal_mask=legal_mask, rho_m=rho_m, theta=theta, b=b, tau=tau)

