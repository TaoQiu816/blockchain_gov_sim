"""解析式四源信誉模型。"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from gov_sim.constants import EPS, THETA_CHOICES, TR_MIN
from gov_sim.modules.evidence_generator import EvidenceBatch
from gov_sim.utils.math_utils import clip01


@dataclass
class ReputationSnapshot:
    """单周期四源信誉快照。"""

    alpha: np.ndarray
    beta: np.ndarray
    direct_trust: np.ndarray
    indirect_trust: np.ndarray
    global_trust: np.ndarray
    rsu_trust: np.ndarray
    final_scores: np.ndarray
    eligible_sets: dict[float, np.ndarray]
    eligible_counts: dict[float, int]
    mean_trust_online: float
    std_trust_online: float
    online_mask: np.ndarray

    @property
    def T_D(self) -> np.ndarray:
        return self.direct_trust

    @property
    def T_I(self) -> np.ndarray:
        return self.indirect_trust

    @property
    def T_G(self) -> np.ndarray:
        return self.global_trust

    @property
    def T_R(self) -> np.ndarray:
        return self.rsu_trust


class ReputationModel:
    """冻结版多源信誉模型。"""

    def __init__(self, config: dict[str, Any], num_rsus: int) -> None:
        self.config = config
        self.num_rsus = int(num_rsus)
        self.eps = 1.0e-8
        self.recent_window = 5
        self.xi = 0.7
        self.sigma = 0.3
        self.theta_p = 1.0
        self.theta_n = 2.0
        self.theta1 = 0.3
        self.verify_decay = 0.9
        self.k = 5
        self.delta_r = 0.10
        self.delta_plus = 0.02
        self.tr_init = 0.8
        self.tr_min = TR_MIN
        self.w_d = 0.35
        self.w_i = 0.20
        self.w_g = 0.25
        self.w_r = 0.20

        self.direct_positive_history: list[np.ndarray] = []
        self.direct_negative_history: list[np.ndarray] = []
        self.verification_history: list[np.ndarray] = []
        self.indirect_trust = np.full(self.num_rsus, 0.5, dtype=np.float32)
        self.global_trust = np.full(self.num_rsus, 0.5, dtype=np.float32)
        self.rsu_trust = np.full(self.num_rsus, self.tr_init, dtype=np.float32)
        self.last_snapshot: ReputationSnapshot | None = None
        self.reset()

    def reset(self) -> None:
        self.direct_positive_history = []
        self.direct_negative_history = []
        self.verification_history = []
        self.indirect_trust = np.full(self.num_rsus, 0.5, dtype=np.float32)
        self.global_trust = np.full(self.num_rsus, 0.5, dtype=np.float32)
        self.rsu_trust = np.full(self.num_rsus, self.tr_init, dtype=np.float32)
        initial_scores = np.full(self.num_rsus, 0.56, dtype=np.float32)
        empty_sets = {float(theta): np.array([], dtype=np.int64) for theta in THETA_CHOICES}
        empty_counts = {float(theta): 0 for theta in THETA_CHOICES}
        self.last_snapshot = ReputationSnapshot(
            alpha=np.ones(self.num_rsus, dtype=np.float32),
            beta=np.ones(self.num_rsus, dtype=np.float32),
            direct_trust=np.full(self.num_rsus, 0.5, dtype=np.float32),
            indirect_trust=self.indirect_trust.copy(),
            global_trust=self.global_trust.copy(),
            rsu_trust=self.rsu_trust.copy(),
            final_scores=initial_scores,
            eligible_sets=empty_sets,
            eligible_counts=empty_counts,
            mean_trust_online=float(np.mean(initial_scores)),
            std_trust_online=float(np.std(initial_scores)),
            online_mask=np.ones(self.num_rsus, dtype=np.int8),
        )

    def _direct_components(self, evidence: EvidenceBatch) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        positive = clip01(evidence.success_commit_ratio + evidence.endorse_match_ratio).astype(np.float32)
        negative = clip01(evidence.timeout_ratio + evidence.offline_ratio).astype(np.float32)
        self.direct_positive_history.append(positive.copy())
        self.direct_negative_history.append(negative.copy())

        recent_pos = np.sum(self.direct_positive_history[-self.recent_window :], axis=0, dtype=np.float32)
        recent_neg = np.sum(self.direct_negative_history[-self.recent_window :], axis=0, dtype=np.float32)
        if len(self.direct_positive_history) > self.recent_window:
            past_pos = np.sum(self.direct_positive_history[: -self.recent_window], axis=0, dtype=np.float32)
            past_neg = np.sum(self.direct_negative_history[: -self.recent_window], axis=0, dtype=np.float32)
        else:
            past_pos = np.zeros(self.num_rsus, dtype=np.float32)
            past_neg = np.zeros(self.num_rsus, dtype=np.float32)

        alpha = self.theta_p * (self.xi * recent_pos + self.sigma * past_pos)
        beta = self.theta_n * (self.xi * recent_neg + self.sigma * past_neg)
        direct_trust = alpha / (alpha + beta + self.eps)
        return alpha.astype(np.float32), beta.astype(np.float32), clip01(direct_trust).astype(np.float32)

    def _update_indirect_trust(self, evidence: EvidenceBatch) -> np.ndarray:
        self.indirect_trust = (1.0 - self.theta1) * self.indirect_trust + self.theta1 * evidence.detect_pass_rate
        self.indirect_trust = clip01(self.indirect_trust).astype(np.float32)
        return self.indirect_trust.copy()

    def _update_global_trust(self, evidence: EvidenceBatch) -> np.ndarray:
        self.verification_history.append(evidence.verification_pass.astype(np.float32).copy())
        depth = len(self.verification_history)
        weights = np.asarray([self.verify_decay ** k for k in range(depth - 1, -1, -1)], dtype=np.float32)
        stacked = np.stack(self.verification_history, axis=0).astype(np.float32)
        weighted = np.tensordot(weights, stacked, axes=(0, 0))
        self.global_trust = weighted / max(float(np.sum(weights)), self.eps)
        self.global_trust = clip01(self.global_trust).astype(np.float32)
        return self.global_trust.copy()

    def _update_rsu_trust(self, evidence: EvidenceBatch) -> np.ndarray:
        online_mask = (evidence.offline_ratio < 0.5).astype(bool)
        punished = np.asarray(evidence.lrd_score > 1.0, dtype=bool) & online_mask
        recovered = (~punished) & online_mask
        updated = self.rsu_trust.copy()
        updated[punished] = np.maximum(0.0, updated[punished] - self.delta_r)
        updated[recovered] = np.minimum(1.0, updated[recovered] + self.delta_plus)
        # Offline nodes remain neutral for this cycle: no punishment, no recovery.
        self.rsu_trust = updated.astype(np.float32)
        return self.rsu_trust.copy()

    def _eligible_sets(
        self,
        final_scores: np.ndarray,
        rsu_trust: np.ndarray,
        online_mask: np.ndarray,
    ) -> tuple[dict[float, np.ndarray], dict[float, int]]:
        eligible_sets: dict[float, np.ndarray] = {}
        eligible_counts: dict[float, int] = {}
        for theta in THETA_CHOICES:
            eligible = np.flatnonzero((final_scores >= float(theta)) & (online_mask == 1) & (rsu_trust >= self.tr_min)).astype(np.int64)
            eligible_sets[float(theta)] = eligible
            eligible_counts[float(theta)] = int(eligible.size)
        return eligible_sets, eligible_counts

    def update(self, context: np.ndarray, evidence: EvidenceBatch) -> ReputationSnapshot:
        del context
        online_mask = (evidence.offline_ratio < 0.5).astype(np.int8)
        alpha, beta, direct_trust = self._direct_components(evidence)
        indirect_trust = self._update_indirect_trust(evidence)
        global_trust = self._update_global_trust(evidence)
        rsu_trust = self._update_rsu_trust(evidence)

        final_scores = clip01(
            self.w_d * direct_trust
            + self.w_i * indirect_trust
            + self.w_g * global_trust
            + self.w_r * rsu_trust
        ).astype(np.float32)
        online_scores = final_scores[online_mask == 1]
        mean_trust_online = float(np.mean(online_scores)) if online_scores.size > 0 else 0.0
        std_trust_online = float(np.std(online_scores)) if online_scores.size > 0 else 0.0
        eligible_sets, eligible_counts = self._eligible_sets(final_scores, rsu_trust, online_mask)

        snapshot = ReputationSnapshot(
            alpha=alpha,
            beta=beta,
            direct_trust=direct_trust,
            indirect_trust=indirect_trust,
            global_trust=global_trust,
            rsu_trust=rsu_trust,
            final_scores=final_scores,
            eligible_sets=eligible_sets,
            eligible_counts=eligible_counts,
            mean_trust_online=mean_trust_online,
            std_trust_online=std_trust_online,
            online_mask=online_mask,
        )
        self.last_snapshot = snapshot
        return snapshot
