"""证据生成模块。

该文件把“外部场景 + 节点角色（诚实/恶意）”映射成第四章信誉模型需要的输入证据：

- 四维软成功度对应的折扣伪计数增量 `Δs / Δf`
- 推荐矩阵
- 预测质量 `yhat`
- 污染率 `pollute_rate`

攻击模式不追求逐论文逐机制复刻，而是服务于统一数值仿真框架下的攻击抽象：
on-off、zigzag、bad-mouthing、ballot-stuffing、collusion、cross-dimensional camouflage。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from gov_sim.constants import REPUTATION_DIMS
from gov_sim.utils.math_utils import clip01


@dataclass
class EvidenceBatch:
    """信誉更新所需的完整证据批次。"""

    delta_s: dict[str, np.ndarray]
    delta_f: dict[str, np.ndarray]
    recommendation_matrix: np.ndarray
    predicted_quality: np.ndarray
    uptime: np.ndarray
    pollute_rate: float


class EvidenceGenerator:
    """把场景状态转成四维信誉证据。"""

    def __init__(self, config: dict[str, Any], seed: int) -> None:
        self.config = config
        self.attack_cfg = config["attack"]
        self.evidence_per_node = int(config["evidence_per_node"])
        self.seed = seed
        self.rng = np.random.default_rng(seed + 17)

    def reset(self, seed: int | None = None) -> None:
        """重置内部随机源。"""

        if seed is not None:
            self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed + 17)

    def _malicious_prob_adjustment(
        self,
        epoch: int,
        base_probs: dict[str, np.ndarray],
        malicious: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """将诚实基础成功概率调整为恶意行为概率。

        设计思路：
        - `svc/con/rec/stab` 先做整体降级，表示恶意节点平均更不可靠；
        - on-off / zigzag 主要作用于 `svc`，对应服务表现时好时坏；
        - cross-dimensional camouflage 刻意提高 `con/rec`、压低 `svc`，
          用来制造“看起来会投票/会推荐，但真实服务质量差”的跨维伪装。
        """

        adjusted = {dim: values.copy() for dim, values in base_probs.items()}
        mal_idx = malicious.astype(bool)
        if not np.any(mal_idx):
            return adjusted

        svc = adjusted["svc"]
        con = adjusted["con"]
        rec = adjusted["rec"]
        stab = adjusted["stab"]

        svc[mal_idx] *= 0.55
        con[mal_idx] *= 0.65
        rec[mal_idx] *= 0.75
        stab[mal_idx] *= 0.82

        if bool(self.attack_cfg.get("on_off", True)):
            period = max(int(self.attack_cfg.get("on_off_period", 8)), 2)
            phase = (epoch // max(period // 2, 1)) % 2
            svc[mal_idx] *= 1.20 if phase == 0 else 0.35
        if bool(self.attack_cfg.get("zigzag", True)):
            freq = float(self.attack_cfg.get("zigzag_freq", 0.18))
            zigzag = 0.5 + 0.5 * np.sign(np.sin(2.0 * np.pi * freq * max(epoch, 1)))
            svc[mal_idx] *= 0.55 + 0.65 * zigzag
        if bool(self.attack_cfg.get("cross_dim_camouflage", True)):
            con[mal_idx] = clip01(con[mal_idx] + 0.15)
            rec[mal_idx] = clip01(rec[mal_idx] + 0.12)
            svc[mal_idx] = clip01(svc[mal_idx] - 0.15)
        return {dim: clip01(values) for dim, values in adjusted.items()}

    def _generate_recommendations(
        self,
        adjusted_probs: dict[str, np.ndarray],
        malicious: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """生成推荐矩阵并估计推荐污染率。

        - bad-mouthing: 恶意节点贬低诚实节点；
        - ballot-stuffing: 恶意节点抬高恶意同伙；
        - collusion: 一小群恶意节点形成高度相似的推荐向量。
        """

        num_rsus = malicious.size
        svc_truth = adjusted_probs["svc"]
        recommendation = np.tile(svc_truth[None, :], (num_rsus, 1))
        recommendation += self.rng.normal(0.0, 0.05, size=recommendation.shape)
        np.fill_diagonal(recommendation, svc_truth)

        malicious_idx = np.where(malicious == 1)[0]
        honest_idx = np.where(malicious == 0)[0]
        if malicious_idx.size > 0:
            group_size = min(int(self.attack_cfg.get("collusion_group_size", 3)), malicious_idx.size)
            colluders = malicious_idx[:group_size]
            if bool(self.attack_cfg.get("bad_mouthing", True)) and honest_idx.size > 0:
                recommendation[np.ix_(malicious_idx, honest_idx)] -= 0.35
            if bool(self.attack_cfg.get("ballot_stuffing", True)) and malicious_idx.size > 0:
                recommendation[np.ix_(malicious_idx, malicious_idx)] += 0.25
            if bool(self.attack_cfg.get("collusion", True)) and colluders.size > 1:
                recommendation[np.ix_(colluders, colluders)] = 0.95

        recommendation = clip01(recommendation)
        honest_truth = np.tile(svc_truth[None, :], (num_rsus, 1))
        pollute_rate = float(np.mean(np.abs(recommendation - honest_truth) > 0.25))
        return recommendation.astype(np.float32), pollute_rate

    def generate(
        self,
        epoch: int,
        base_probs: dict[str, np.ndarray],
        malicious: np.ndarray,
        online: np.ndarray,
        uptime: np.ndarray,
    ) -> EvidenceBatch:
        """生成单周期证据。

        每条证据都以软成功度 `y in [0,1]` 和权重 `w in (0,1]` 的形式进入
        `Δs = Σ w*y`、`Δf = Σ w*(1-y)`，与论文折扣伪计数模型严格一致。
        """

        num_rsus = malicious.size
        adjusted_probs = self._malicious_prob_adjustment(epoch=epoch, base_probs=base_probs, malicious=malicious)
        delta_s: dict[str, np.ndarray] = {}
        delta_f: dict[str, np.ndarray] = {}
        for dim in REPUTATION_DIMS:
            prob = adjusted_probs[dim] * online + (1.0 - online) * 0.05
            delta_s_dim = np.zeros(num_rsus, dtype=np.float32)
            delta_f_dim = np.zeros(num_rsus, dtype=np.float32)
            for _ in range(self.evidence_per_node):
                weight = self.rng.uniform(0.55, 1.0, size=num_rsus).astype(np.float32)
                y = clip01(prob + self.rng.normal(0.0, 0.06, size=num_rsus))
                delta_s_dim += weight * y
                delta_f_dim += weight * (1.0 - y)
            delta_s[dim] = delta_s_dim
            delta_f[dim] = delta_f_dim
        recommendation_matrix, pollute_rate = self._generate_recommendations(adjusted_probs=adjusted_probs, malicious=malicious)
        return EvidenceBatch(
            delta_s=delta_s,
            delta_f=delta_f,
            recommendation_matrix=recommendation_matrix,
            predicted_quality=adjusted_probs["svc"].astype(np.float32),
            uptime=uptime.astype(np.float32),
            pollute_rate=pollute_rate,
        )
