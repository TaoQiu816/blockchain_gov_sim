"""多维信誉模型。

该文件实现第四章核心建模：

1. 四维信誉 `svc/con/rec/stab`；
2. 折扣伪计数更新 `alpha/beta`；
3. 后验均值 `mu`、方差 `sigma^2`、有效证据量 `n_eff`；
4. 冷启动门控 `kappa`；
5. 上下文感知维度权重；
6. 跨维一致性、on-off/zigzag、推荐偏差、合谋、稳定性惩罚。

最终输出 `ReputationSnapshot`，供环境构造资格集、action mask、baseline 与评估模块使用。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from gov_sim.constants import EPS, REPUTATION_DIMS
from gov_sim.modules.evidence_generator import EvidenceBatch
from gov_sim.utils.math_utils import clip01, positive_part, safe_div


@dataclass
class ReputationSnapshot:
    """单周期信誉快照。

    该对象刻意保留中间量，而不仅仅是最终分数，
    便于论文绘图、审计、消融和二次开发。
    """

    alpha: dict[str, np.ndarray]
    beta: dict[str, np.ndarray]
    mu: dict[str, np.ndarray]
    var: dict[str, np.ndarray]
    neff: dict[str, np.ndarray]
    kappa: dict[str, np.ndarray]
    dim_weights: dict[str, float]
    base_scores: np.ndarray
    final_scores: np.ndarray
    penalties: dict[str, np.ndarray]
    rec_penalty: np.ndarray
    collusion_penalty: np.ndarray


class ReputationModel:
    """上下文感知多维信誉模型。"""

    def __init__(self, config: dict[str, Any], num_rsus: int) -> None:
        self.config = config
        self.num_rsus = num_rsus
        self.prior_alpha = float(config["prior_alpha"])
        self.prior_beta = float(config["prior_beta"])
        self.rho_cfg = config["rho"]
        self.n0 = float(config["n0"])
        self.eps = float(config["eps"])
        self.context_eta = {dim: np.asarray(config["context_eta"][dim], dtype=np.float32) for dim in REPUTATION_DIMS}
        self.penalty_cfg = config["penalty"]
        self.use_context_gate = bool(config.get("use_context_gate", True))
        self.use_penalties = bool(config.get("use_penalties", True))
        self.context_count = 1.0
        self.context_mean = np.zeros(6, dtype=np.float32)
        self.context_m2 = np.ones(6, dtype=np.float32)
        self.alpha: dict[str, np.ndarray] = {}
        self.beta: dict[str, np.ndarray] = {}
        self.prev_mu_svc = np.full(num_rsus, 0.5, dtype=np.float32)
        self.vbar = np.zeros(num_rsus, dtype=np.float32)
        self.last_snapshot: ReputationSnapshot | None = None
        self.reset()

    def reset(self) -> None:
        """重置伪计数、上下文统计和历史波动状态。"""
        self.alpha = {dim: np.full(self.num_rsus, self.prior_alpha, dtype=np.float32) for dim in REPUTATION_DIMS}
        self.beta = {dim: np.full(self.num_rsus, self.prior_beta, dtype=np.float32) for dim in REPUTATION_DIMS}
        self.prev_mu_svc = np.full(self.num_rsus, self.prior_alpha / (self.prior_alpha + self.prior_beta), dtype=np.float32)
        self.vbar = np.zeros(self.num_rsus, dtype=np.float32)
        self.last_snapshot = None
        self.context_count = 1.0
        self.context_mean = np.zeros(6, dtype=np.float32)
        self.context_m2 = np.ones(6, dtype=np.float32)

    def _normalize_context(self, context: np.ndarray) -> np.ndarray:
        """在线标准化上下文向量。

        论文要求对 `c_e` 做标准化后再参与 `exp(eta_d^T c_bar_e)`。
        这里使用在线均值/方差，避免必须预先扫描全数据集。
        """

        self.context_count += 1.0
        delta = context - self.context_mean
        self.context_mean += delta / self.context_count
        delta2 = context - self.context_mean
        self.context_m2 += delta * delta2
        var = self.context_m2 / max(self.context_count - 1.0, 1.0)
        return (context - self.context_mean) / np.sqrt(var + self.eps)

    def _calc_rec_penalty(self, recommendation_matrix: np.ndarray, yhat: np.ndarray) -> np.ndarray:
        """推荐偏差惩罚。

        对应：
        `p_rec_i(e) = mean_j | r_{i->j}(e) - yhat_j(e) |`
        """
        return np.mean(np.abs(recommendation_matrix - yhat[None, :]), axis=1).astype(np.float32)

    def _calc_collusion_penalty(self, recommendation_matrix: np.ndarray) -> np.ndarray:
        """合谋惩罚。

        论文允许 Pearson 或 cosine 相似度。这里采用 cosine，
        原因是数值稳定、无需额外处理近常数向量的方差退化问题。
        """
        normalized = recommendation_matrix - recommendation_matrix.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(normalized, axis=1, keepdims=True) + self.eps
        cosine = (normalized @ normalized.T) / (norms @ norms.T)
        np.fill_diagonal(cosine, 0.0)
        rho0 = float(self.penalty_cfg["rho0"])
        return np.mean(positive_part(cosine - rho0), axis=1).astype(np.float32)

    def update(self, context: np.ndarray, evidence: EvidenceBatch) -> ReputationSnapshot:
        """用单周期证据更新全部节点信誉。

        关键公式与论文对齐：

        - `alpha_{j,d}(e) = rho_d * alpha_{j,d}(e-1) + Δs_{j,d}(e)`
        - `beta_{j,d}(e) = rho_d * beta_{j,d}(e-1) + Δf_{j,d}(e)`
        - `mu = alpha / (alpha + beta)`
        - `var = alpha*beta / ((alpha+beta)^2 (alpha+beta+1))`
        - `kappa = 1 - exp(-n_eff / N0)`
        - `w_tilde_d = rbar_d * exp(eta_d^T cbar_e)`
        - `T_base = Σ_d w_d * mu_{j,d}`
        - `T_j = clip_01(T_base - Σ penalty)`
        """

        mu: dict[str, np.ndarray] = {}
        var: dict[str, np.ndarray] = {}
        neff: dict[str, np.ndarray] = {}
        kappa: dict[str, np.ndarray] = {}
        reliability_median: dict[str, float] = {}

        for dim in REPUTATION_DIMS:
            rho = float(self.rho_cfg[dim])
            # 折扣伪计数：旧证据被 rho_d 衰减，新证据以软成功/失败量注入。
            self.alpha[dim] = rho * self.alpha[dim] + evidence.delta_s[dim]
            self.beta[dim] = rho * self.beta[dim] + evidence.delta_f[dim]
            total = self.alpha[dim] + self.beta[dim]
            mu[dim] = safe_div(self.alpha[dim], total, eps=self.eps).astype(np.float32)
            var[dim] = (self.alpha[dim] * self.beta[dim] / ((total**2) * (total + 1.0) + self.eps)).astype(np.float32)
            neff[dim] = total.astype(np.float32)
            kappa[dim] = (1.0 - np.exp(-total / max(self.n0, self.eps))).astype(np.float32)
            # `r_{j,d} = kappa / (sigma^2 + eps)`：
            # 冷启动门控和不确定性一起决定该维度在本周期是否“值得被信任”。
            reliability = kappa[dim] / (var[dim] + self.eps)
            reliability_median[dim] = float(np.median(reliability))

        normalized_context = self._normalize_context(context)
        raw_weights = []
        for dim in REPUTATION_DIMS:
            # 上下文门控不是固定权重，而是随 `A_e/Q_e/L_bar_e/RTT_e/churn/|E_e|` 动态变化。
            ctx_term = float(np.exp(np.dot(self.context_eta[dim], normalized_context))) if self.use_context_gate else 1.0
            raw_weights.append(reliability_median[dim] * ctx_term)
        raw_weights_np = np.asarray(raw_weights, dtype=np.float32)
        raw_weights_np = raw_weights_np / np.clip(raw_weights_np.sum(), self.eps, None)
        dim_weights = {dim: float(raw_weights_np[idx]) for idx, dim in enumerate(REPUTATION_DIMS)}

        base_scores = np.zeros(self.num_rsus, dtype=np.float32)
        for dim in REPUTATION_DIMS:
            base_scores += dim_weights[dim] * mu[dim]

        # 跨维一致性惩罚：防止“共识/推荐看起来很好，但服务维度很差”的伪装。
        delta_sc = positive_part(mu["con"] - mu["svc"] - float(self.penalty_cfg["delta_cs"])) + positive_part(
            mu["rec"] - mu["svc"] - float(self.penalty_cfg["delta_rs"])
        )
        # on-off / zigzag 惩罚：用服务维度的波动 EWMA 识别间歇式攻击。
        v = np.abs(mu["svc"] - self.prev_mu_svc)
        eta = float(self.penalty_cfg["onoff_eta"])
        self.vbar = eta * self.vbar + (1.0 - eta) * v
        p_onoff = positive_part(self.vbar - float(self.penalty_cfg["v0"])) * (mu["svc"] > float(self.penalty_cfg["mu0"]))
        p_rec = self._calc_rec_penalty(evidence.recommendation_matrix, evidence.predicted_quality)
        p_col = self._calc_collusion_penalty(evidence.recommendation_matrix)
        p_stab = positive_part(float(self.penalty_cfg["u0"]) - evidence.uptime)

        penalties = {
            "cross_dim": delta_sc.astype(np.float32),
            "onoff": p_onoff.astype(np.float32),
            "rec": p_rec.astype(np.float32),
            "collusion": p_col.astype(np.float32),
            "stab": p_stab.astype(np.float32),
        }

        if self.use_penalties:
            final_scores = base_scores.copy()
            final_scores -= float(self.penalty_cfg["gamma_sc"]) * delta_sc
            final_scores -= float(self.penalty_cfg["gamma_on"]) * p_onoff
            final_scores -= float(self.penalty_cfg["gamma_rec"]) * p_rec
            final_scores -= float(self.penalty_cfg["gamma_col"]) * p_col
            final_scores -= float(self.penalty_cfg["gamma_stab"]) * p_stab
        else:
            p_rec = np.zeros_like(base_scores)
            p_col = np.zeros_like(base_scores)
            penalties = {key: np.zeros_like(base_scores) for key in penalties}
            final_scores = base_scores.copy()
        final_scores = clip01(final_scores).astype(np.float32)
        self.prev_mu_svc = mu["svc"].copy()

        snapshot = ReputationSnapshot(
            alpha={dim: values.copy() for dim, values in self.alpha.items()},
            beta={dim: values.copy() for dim, values in self.beta.items()},
            mu=mu,
            var=var,
            neff=neff,
            kappa=kappa,
            dim_weights=dim_weights,
            base_scores=base_scores.astype(np.float32),
            final_scores=final_scores,
            penalties=penalties,
            rec_penalty=p_rec.astype(np.float32),
            collusion_penalty=p_col.astype(np.float32),
        )
        self.last_snapshot = snapshot
        return snapshot
