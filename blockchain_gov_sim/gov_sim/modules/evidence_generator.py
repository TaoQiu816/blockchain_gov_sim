"""四源信誉证据生成模块。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from gov_sim.constants import EPS
from gov_sim.utils.math_utils import clip01


@dataclass
class EvidenceBatch:
    """当前周期四源信誉更新所需证据。"""

    success_commit_ratio: np.ndarray
    endorse_match_ratio: np.ndarray
    timeout_ratio: np.ndarray
    offline_ratio: np.ndarray
    detect_pass_rate: np.ndarray
    verification_pass: np.ndarray
    lrd_score: np.ndarray
    pollute_rate: float


class EvidenceGenerator:
    """为 direct / indirect / global / RSU trust 生成冻结版证据。"""

    def __init__(self, config: dict[str, Any], seed: int) -> None:
        self.config = config
        self.attack_cfg = config["attack"]
        self.evidence_cfg = config.get("evidence", {})
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed + 17)
        self.success_noise_std = float(self.evidence_cfg.get("success_noise_std", 0.04))
        self.endorse_noise_std = float(self.evidence_cfg.get("endorse_noise_std", 0.04))
        self.timeout_noise_std = float(self.evidence_cfg.get("timeout_noise_std", 0.03))
        self.detect_noise_std = float(self.evidence_cfg.get("detect_noise_std", 0.05))
        self.success_malicious_penalty = float(self.evidence_cfg.get("success_malicious_penalty", 0.30))
        self.endorse_malicious_penalty = float(self.evidence_cfg.get("endorse_malicious_penalty", 0.22))
        self.detect_malicious_penalty = float(self.evidence_cfg.get("detect_malicious_penalty", 0.28))
        self.verify_malicious_penalty = float(self.evidence_cfg.get("verify_malicious_penalty", 0.30))
        self.timeout_malicious_boost = float(self.evidence_cfg.get("timeout_malicious_boost", 0.28))

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed + 17)

    def _adjust_for_malicious(self, values: np.ndarray, malicious: np.ndarray, penalty: float) -> np.ndarray:
        adjusted = values.copy()
        mal_mask = malicious.astype(bool)
        adjusted[mal_mask] = clip01(adjusted[mal_mask] - penalty)
        return adjusted.astype(np.float32)

    def _generate_ratios(
        self,
        base_probs: dict[str, np.ndarray],
        malicious: np.ndarray,
        online: np.ndarray,
        uptime: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        online_float = online.astype(np.float32)
        offline_ratio = (1.0 - online_float).astype(np.float32)

        success_base = clip01(0.65 * base_probs["svc"] + 0.25 * base_probs["stab"] + 0.10 * uptime)
        endorse_base = clip01(0.75 * base_probs["con"] + 0.25 * base_probs["rec"])
        timeout_base = clip01(1.0 - base_probs["con"] + 0.25 * offline_ratio)
        detect_base = clip01(0.55 * base_probs["rec"] + 0.45 * base_probs["stab"])
        verify_base = clip01(0.60 * base_probs["con"] + 0.40 * base_probs["stab"])

        success_base = self._adjust_for_malicious(success_base, malicious, penalty=self.success_malicious_penalty)
        endorse_base = self._adjust_for_malicious(endorse_base, malicious, penalty=self.endorse_malicious_penalty)
        detect_base = self._adjust_for_malicious(detect_base, malicious, penalty=self.detect_malicious_penalty)
        verify_base = self._adjust_for_malicious(verify_base, malicious, penalty=self.verify_malicious_penalty)

        timeout_base = timeout_base.copy()
        timeout_base[malicious.astype(bool)] = clip01(timeout_base[malicious.astype(bool)] + self.timeout_malicious_boost)

        success_commit_ratio = clip01(success_base * online_float + self.rng.normal(0.0, self.success_noise_std, size=online.shape))
        endorse_match_ratio = clip01(endorse_base * online_float + self.rng.normal(0.0, self.endorse_noise_std, size=online.shape))
        timeout_ratio = clip01(timeout_base + self.rng.normal(0.0, self.timeout_noise_std, size=online.shape))
        timeout_ratio = clip01(timeout_ratio * online_float + offline_ratio)
        detect_pass_rate = clip01(detect_base * online_float + self.rng.normal(0.0, self.detect_noise_std, size=online.shape))
        detect_pass_rate = clip01(detect_pass_rate * online_float)
        verify_prob = clip01(verify_base * online_float)
        verification_pass = (self.rng.random(size=online.shape[0]) < verify_prob).astype(np.int8)
        verification_pass = verification_pass * online.astype(np.int8)
        return (
            success_commit_ratio.astype(np.float32),
            endorse_match_ratio.astype(np.float32),
            timeout_ratio.astype(np.float32),
            offline_ratio.astype(np.float32),
            detect_pass_rate.astype(np.float32),
            verification_pass.astype(np.int8),
        )

    def _compute_lrd_score(
        self,
        success_commit_ratio: np.ndarray,
        endorse_match_ratio: np.ndarray,
        timeout_ratio: np.ndarray,
        offline_ratio: np.ndarray,
        online: np.ndarray,
        k: int = 5,
    ) -> np.ndarray:
        """使用本周期 direct-trust 代理向量计算基于密度的异常分数。"""

        direct_proxy = (success_commit_ratio + endorse_match_ratio) / (
            success_commit_ratio + endorse_match_ratio + timeout_ratio + offline_ratio + EPS
        )
        direct_proxy = direct_proxy.astype(np.float32)
        online_idx = np.flatnonzero(online == 1)
        lrd_score = np.ones_like(direct_proxy, dtype=np.float32)
        if online_idx.size == 0:
            return np.full_like(direct_proxy, 2.0, dtype=np.float32)

        effective_k = max(1, min(int(k), max(online_idx.size - 1, 1)))
        densities = np.zeros(online_idx.size, dtype=np.float32)

        for pos, rsu_idx in enumerate(online_idx):
            distances = np.abs(direct_proxy[online_idx] - direct_proxy[rsu_idx]).astype(np.float32)
            order = np.argsort(distances, kind="stable")
            neighbors = order[1 : effective_k + 1] if online_idx.size > 1 else order[:1]
            reach_dist = np.maximum(distances[neighbors], float(np.max(distances[neighbors])) if neighbors.size > 0 else 0.0)
            densities[pos] = 1.0 / max(float(np.mean(reach_dist)) + EPS, EPS)

        for pos, rsu_idx in enumerate(online_idx):
            distances = np.abs(direct_proxy[online_idx] - direct_proxy[rsu_idx]).astype(np.float32)
            order = np.argsort(distances, kind="stable")
            neighbors = order[1 : effective_k + 1] if online_idx.size > 1 else order[:1]
            neighbor_density = float(np.mean(densities[neighbors])) if neighbors.size > 0 else float(densities[pos])
            lrd_score[rsu_idx] = float(np.clip(neighbor_density / max(float(densities[pos]), EPS), 0.0, 10.0))

        # Offline nodes are excluded from this cycle's outlier punishment path.
        lrd_score[online == 0] = 1.0
        return lrd_score.astype(np.float32)

    def generate(
        self,
        epoch: int,
        base_probs: dict[str, np.ndarray],
        malicious: np.ndarray,
        online: np.ndarray,
        uptime: np.ndarray,
    ) -> EvidenceBatch:
        del epoch
        (
            success_commit_ratio,
            endorse_match_ratio,
            timeout_ratio,
            offline_ratio,
            detect_pass_rate,
            verification_pass,
        ) = self._generate_ratios(
            base_probs=base_probs,
            malicious=malicious,
            online=online,
            uptime=uptime,
        )
        lrd_score = self._compute_lrd_score(
            success_commit_ratio=success_commit_ratio,
            endorse_match_ratio=endorse_match_ratio,
            timeout_ratio=timeout_ratio,
            offline_ratio=offline_ratio,
            online=online,
            k=5,
        )
        pollute_rate = float(np.mean(malicious.astype(np.float32) * online.astype(np.float32)))
        return EvidenceBatch(
            success_commit_ratio=success_commit_ratio,
            endorse_match_ratio=endorse_match_ratio,
            timeout_ratio=timeout_ratio,
            offline_ratio=offline_ratio,
            detect_pass_rate=detect_pass_rate,
            verification_pass=verification_pass,
            lrd_score=lrd_score,
            pollute_rate=pollute_rate,
        )
