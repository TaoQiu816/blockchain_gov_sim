"""DVRC-2025 baseline。"""

from __future__ import annotations

from typing import Any

import numpy as np

from gov_sim.baselines import BaselineBase


class DVRC2025Baseline(BaselineBase):
    """动态信誉分数 / 动态共识阈值启发式。"""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config=config, name="DVRC-2025")

    def select_action(self, env: Any, obs: np.ndarray, legal_mask: np.ndarray | None = None) -> int:
        del obs
        state = self._governance_state(env)
        snapshot = state["snapshot"]
        final_scores = np.asarray(snapshot.final_scores, dtype=np.float32)
        global_trust = np.asarray(snapshot.T_G, dtype=np.float32)
        rsu_trust = np.asarray(snapshot.T_R, dtype=np.float32)
        online_scores = final_scores[np.asarray(snapshot.online_mask, dtype=np.int8) == 1]
        mean_trust = float(np.mean(online_scores)) if online_scores.size > 0 else 0.0
        std_trust = float(np.std(online_scores)) if online_scores.size > 0 else 0.0
        risk = (
            0.45 * (1.0 - mean_trust)
            + 0.20 * std_trust
            + 0.20 * float(np.mean(1.0 - global_trust))
            + 0.15 * float(np.mean(1.0 - rsu_trust))
        )
        if risk >= 0.42:
            theta = 0.60
            rho_m = 5.0 / 27.0
            b = 320
            tau = 100
        elif risk >= 0.28:
            theta = 0.55
            rho_m = 7.0 / 27.0
            b = 384
            tau = 80
        else:
            theta = 0.50
            rho_m = 9.0 / 27.0
            b = 448
            tau = 60
        return self._legal_action(legal_mask=legal_mask, rho_m=rho_m, theta=theta, b=b, tau=tau)

