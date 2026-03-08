"""委员会采样模块。

主方案要求委员会采用“软抽签 + 无放回加权采样”，而不是简单 Top-K。
这里实现可复现、稳定的无放回加权采样，并对外提供带 seed 的封装类。
"""

from __future__ import annotations

import numpy as np


def weighted_sample_without_replacement(
    candidates: np.ndarray,
    weights: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Efraimidis-Spirakis 风格的无放回加权采样。

    关键点：
    - `k > |candidates|` 直接报错，防止静默截断；
    - 使用对数形式 `log(U)/w` 构造 key，避免 `U^(1/w)` 在小概率下数值不稳定；
    - 每个候选只对应一个 key，因此天然无放回。
    """

    if k < 0:
        raise ValueError("k must be non-negative")
    if k > candidates.size:
        raise ValueError("k cannot exceed number of candidates")
    if candidates.size == 0 or k == 0:
        return np.array([], dtype=np.int64)
    stable_weights = np.clip(weights.astype(np.float64), 1.0e-12, None)
    uniforms = np.clip(rng.random(candidates.size), 1.0e-12, 1.0 - 1.0e-12)
    keys = np.log(uniforms) / stable_weights
    selected = np.argsort(keys)[-k:]
    selected = selected[np.argsort(keys[selected])[::-1]]
    return candidates[selected].astype(np.int64)


class CommitteeSampler:
    """带内部随机源的委员会采样器。"""

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed + 29)

    def reset(self, seed: int | None = None) -> None:
        """重置采样器随机源。"""

        if seed is not None:
            self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed + 29)

    def sample(self, candidates: np.ndarray, weights: np.ndarray, committee_size: int) -> np.ndarray:
        return weighted_sample_without_replacement(candidates=candidates, weights=weights, k=committee_size, rng=self.rng)
