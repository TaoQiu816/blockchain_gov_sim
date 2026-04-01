"""Zhao-2024 baseline。"""

from __future__ import annotations

from typing import Any

import numpy as np

from gov_sim.baselines import BaselineBase


class Zhao2024Baseline(BaselineBase):
    """基于 T_D + T_I 的轻量固定治理基线。"""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config=config, name="Zhao-2024")

    def select_action(self, env: Any, obs: np.ndarray, legal_mask: np.ndarray | None = None) -> int:
        del obs
        state = self._governance_state(env)
        snapshot = state["snapshot"]
        scenario = state["scenario"]
        combined = 0.5 * np.asarray(snapshot.T_D, dtype=np.float32) + 0.5 * np.asarray(snapshot.T_I, dtype=np.float32)
        online_scores = combined[np.asarray(snapshot.online_mask, dtype=np.int8) == 1]
        mean_score = float(np.mean(online_scores)) if online_scores.size > 0 else 0.0
        theta = 0.55 if mean_score < 0.50 else 0.50
        tau = 100 if float(scenario.rtt) > 0.8 * float(env.rtt_max) else 80
        return self._legal_action(
            legal_mask=legal_mask,
            rho_m=7.0 / 27.0,
            theta=theta,
            b=384,
            tau=tau,
        )

