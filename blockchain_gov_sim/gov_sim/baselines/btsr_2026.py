"""BTSR-2026 baseline。"""

from __future__ import annotations

from typing import Any

import numpy as np

from gov_sim.baselines import BaselineBase


class BTSR2026Baseline(BaselineBase):
    """双层安全优先启发式，显式利用四源信誉与 TOF 结果。"""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config=config, name="BTSR-2026")

    def select_action(self, env: Any, obs: np.ndarray, legal_mask: np.ndarray | None = None) -> int:
        del obs
        state = self._governance_state(env)
        snapshot = state["snapshot"]
        queue = float(state["queue_size"])
        scores = np.asarray(snapshot.final_scores, dtype=np.float32)
        rsu_trust = np.asarray(snapshot.T_R, dtype=np.float32)
        online_mask = np.asarray(snapshot.online_mask, dtype=np.int8) == 1
        mean_trust = float(np.mean(scores[online_mask])) if np.any(online_mask) else 0.0
        mean_rsu_trust = float(np.mean(rsu_trust[online_mask])) if np.any(online_mask) else 0.0
        eligible_055 = int(snapshot.eligible_counts.get(0.55, 0))
        risk = 0.55 * (1.0 - mean_trust) + 0.30 * (1.0 - mean_rsu_trust) + 0.15 * float(eligible_055 < 5)

        if risk >= 0.40:
            theta = 0.60
            rho_m = 5.0 / 27.0
            b = 320
            tau = 100
        elif risk >= 0.26:
            theta = 0.55
            rho_m = 7.0 / 27.0
            b = 384
            tau = 80
        else:
            theta = 0.50
            rho_m = 7.0 / 27.0 if queue < 0.35 * float(env.queue_max) else 9.0 / 27.0
            b = 384
            tau = 80
        return self._legal_action(legal_mask=legal_mask, rho_m=rho_m, theta=theta, b=b, tau=tau)

