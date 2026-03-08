"""数值计算辅助函数。"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def clip01(x: np.ndarray | float) -> np.ndarray | float:
    """裁剪到 [0, 1]。"""
    return np.clip(x, 0.0, 1.0)


def positive_part(x: np.ndarray | float) -> np.ndarray | float:
    """正部函数 `[x]_+`。"""
    return np.maximum(x, 0.0)


def safe_div(numerator: np.ndarray | float, denominator: np.ndarray | float, eps: float = 1.0e-8) -> np.ndarray | float:
    """带 `eps` 的安全除法，避免除零。"""
    return np.asarray(numerator) / (np.asarray(denominator) + eps)


def quantile_summary(values: np.ndarray, quantiles: Iterable[float] = (0.1, 0.25, 0.5, 0.75, 0.9)) -> np.ndarray:
    """提取分位数摘要，用于压缩高维节点分布。"""
    if values.size == 0:
        return np.zeros(len(tuple(quantiles)), dtype=np.float32)
    return np.quantile(values, list(quantiles)).astype(np.float32)


def histogram_summary(values: np.ndarray, bins: int = 5) -> np.ndarray:
    """提取 [0,1] 区间上的直方图摘要。"""
    if values.size == 0:
        return np.zeros(bins, dtype=np.float32)
    hist, _ = np.histogram(values, bins=bins, range=(0.0, 1.0), density=True)
    return hist.astype(np.float32)


class RunningMeanStd:
    """在线均值/方差统计，用于 reward/cost 标准化。"""

    def __init__(self, epsilon: float = 1.0e-4) -> None:
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, values: np.ndarray) -> None:
        """合并一批新样本。"""
        batch_mean = float(np.mean(values))
        batch_var = float(np.var(values))
        batch_count = float(values.size)
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total
        self.mean = new_mean
        self.var = max(m2 / total, 1.0e-8)
        self.count = total

    def normalize(self, values: np.ndarray) -> np.ndarray:
        """用当前均值/方差标准化输入。"""
        return (values - self.mean) / np.sqrt(self.var + 1.0e-8)
