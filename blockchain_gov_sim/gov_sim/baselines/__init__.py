"""统一 baseline 注册与公共工具。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from gov_sim.constants import B_CHOICES, RHO_M_CHOICES, TAU_CHOICES, THETA_CHOICES
from gov_sim.env.action_codec import ActionCodec, GovernanceAction


class BaselinePolicy(Protocol):
    """统一 baseline 协议。"""

    name: str

    def reset(self) -> None:
        ...

    def select_action(self, env: Any, obs: np.ndarray, legal_mask: np.ndarray | None = None) -> int:
        ...


@dataclass
class BaselineBase:
    """所有 baseline 的公共工具。"""

    config: dict[str, Any]
    name: str
    codec: ActionCodec = field(default_factory=ActionCodec)

    def reset(self) -> None:
        return None

    def _governance_state(self, env: Any) -> dict[str, Any]:
        return env.get_governance_state()

    def _nearest_legal_action(self, preferred: GovernanceAction, legal_mask: np.ndarray | None) -> int:
        preferred_idx = self.codec.encode(preferred)
        if legal_mask is None:
            return preferred_idx
        legal = np.flatnonzero(np.asarray(legal_mask, dtype=np.int8) == 1)
        if legal.size == 0:
            return preferred_idx
        if int(legal_mask[preferred_idx]) == 1:
            return preferred_idx
        best_idx = preferred_idx
        best_dist = float("inf")
        for action_idx in legal.tolist():
            candidate = self.codec.decode(int(action_idx))
            distance = (
                abs(float(candidate.rho_m) - float(preferred.rho_m))
                + abs(float(candidate.theta) - float(preferred.theta))
                + abs(int(candidate.b) - int(preferred.b))
                + abs(int(candidate.tau) - int(preferred.tau))
            )
            if distance < best_dist:
                best_dist = float(distance)
                best_idx = int(action_idx)
        return best_idx

    def _legal_action(
        self,
        legal_mask: np.ndarray | None,
        rho_m: float,
        theta: float,
        b: int,
        tau: int,
    ) -> int:
        preferred = GovernanceAction(
            rho_m=min(RHO_M_CHOICES, key=lambda value: abs(float(value) - float(rho_m))),
            theta=min(THETA_CHOICES, key=lambda value: abs(float(value) - float(theta))),
            b=min(B_CHOICES, key=lambda value: abs(int(value) - int(b))),
            tau=min(TAU_CHOICES, key=lambda value: abs(int(value) - int(tau))),
        )
        return self._nearest_legal_action(preferred, legal_mask)


def quantile_threshold(values: np.ndarray, q: float) -> float:
    """信誉分位阈值。"""

    if values.size == 0:
        return float(THETA_CHOICES[0])
    return float(np.quantile(values, q))


def trust_entropy(values: np.ndarray, bins: int = 5) -> float:
    """基于信誉分布的简单熵指标。"""

    if values.size == 0:
        return 0.0
    hist, _ = np.histogram(np.asarray(values, dtype=np.float32), bins=bins, range=(0.0, 1.0))
    probs = hist.astype(np.float64) / max(float(np.sum(hist)), 1.0)
    probs = probs[probs > 0.0]
    if probs.size == 0:
        return 0.0
    entropy = -float(np.sum(probs * np.log(probs)))
    return float(entropy / max(np.log(float(bins)), 1.0))


from gov_sim.baselines.btsr_2026 import BTSR2026Baseline
from gov_sim.baselines.dvrc_2025 import DVRC2025Baseline
from gov_sim.baselines.static_best_fixed import StaticBestFixedBaseline
from gov_sim.baselines.yadav_2025 import Yadav2025Baseline
from gov_sim.baselines.zhao_2024 import Zhao2024Baseline


BASELINE_REGISTRY = {
    "Static-Best-Fixed": StaticBestFixedBaseline,
    "BTSR-2026": BTSR2026Baseline,
    "Zhao-2024": Zhao2024Baseline,
    "Yadav-2025": Yadav2025Baseline,
    "DVRC-2025": DVRC2025Baseline,
}


def instantiate_baseline(name: str, config: dict[str, Any]) -> BaselinePolicy:
    if name not in BASELINE_REGISTRY:
        raise ValueError(f"Unknown baseline: {name}")
    return BASELINE_REGISTRY[name](config)

