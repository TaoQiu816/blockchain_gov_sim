"""链侧性能模型。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ChainStepResult:
    """链侧一步执行结果。"""

    queue_next: float
    service_capacity: float
    arrival_hat: float
    wait_to_fill: float
    effective_batch: float
    mu_eff: float
    queue_latency: float
    batch_latency: float
    consensus_latency: float
    total_latency: float
    committee_mean_trust: float
    unsafe: int
    timeout_failure: int
    success: int
    margin_cost: float
    block_slack: float
    tps: float
    h_lcb: float


class ChainModel:
    """保留已审计性能结构的链侧模型。"""

    def __init__(self, config: dict[str, Any], h_min: float) -> None:
        self.cfg = config
        self.h_min = float(h_min)
        self.arrival_hat = float(config.get("initial_arrival_hat", 1.0))

    def reset(self) -> None:
        self.arrival_hat = float(self.cfg.get("initial_arrival_hat", 1.0))

    def step(
        self,
        queue_size: float,
        arrivals: int,
        eligible_size: int,
        committee: np.ndarray,
        committee_size: int,
        batch_size: int,
        tau_ms: int,
        rtt: float,
        churn: float,
        committee_trust_scores: np.ndarray,
        malicious: np.ndarray,
    ) -> ChainStepResult:
        eps = 1.0e-8
        delta_t_ms = float(self.cfg.get("delta_t_ms", 100.0))
        self.arrival_hat = float(self.cfg["lambda_ewma"]) * self.arrival_hat + (1.0 - float(self.cfg["lambda_ewma"])) * float(arrivals)
        x_total = float(queue_size) + float(arrivals)
        effective_batch = min(x_total, float(batch_size))
        pair_term = float(committee_size * max(committee_size - 1, 0) / 2.0)
        structural_feasible = int(int(eligible_size) >= int(committee_size) and int(committee_size) > 0)

        mu_eff = (
            float(self.cfg["mu0"])
            * np.exp(-float(self.cfg["a_r"]) * float(rtt) - float(self.cfg["a_chi"]) * float(churn))
            * np.exp(-float(self.cfg["a_m"]) * pair_term)
            * (1.0 / (1.0 + float(self.cfg["a_b"]) * effective_batch / max(float(self.cfg["B_ref"]), 1.0)))
            * structural_feasible
        )
        served = min(effective_batch, mu_eff)
        queue_next = max(0.0, x_total - served)
        arrival_rate_per_ms = self.arrival_hat / max(delta_t_ms, eps)
        wait_to_fill = min(float(tau_ms), max(0.0, (float(batch_size) - x_total) / max(arrival_rate_per_ms, eps))) if x_total < float(batch_size) else 0.0
        queue_latency = queue_next / max(mu_eff, eps) if structural_feasible else 0.0
        batch_latency = wait_to_fill / 2.0
        consensus_latency = float(rtt) * (
            float(self.cfg["c0"]) + float(self.cfg["c1"]) * float(committee_size) + float(self.cfg["c2"]) * pair_term
        ) + float(self.cfg["c3"]) * effective_batch / max(float(self.cfg["B_ref"]), 1.0)
        total_latency = queue_latency + batch_latency + consensus_latency

        committee_mean_trust = float(np.mean(committee_trust_scores)) if committee_trust_scores.size > 0 else 0.0
        h_warn = float(self.cfg.get("h_warn", self.h_min + 0.10))
        unsafe = int(structural_feasible and committee_mean_trust < self.h_min)
        timeout_failure = int(structural_feasible and consensus_latency > float(tau_ms))
        margin_cost = (
            float(np.clip((h_warn - committee_mean_trust) / max(h_warn - self.h_min, eps), 0.0, 1.0))
            if committee_size > 0
            else 1.0
        )
        success = int(structural_feasible and not unsafe and not timeout_failure)
        block_slack = max(float(batch_size) - served, 0.0) / max(float(batch_size), 1.0)
        tps = float(served) / max(total_latency / 1000.0, eps) if total_latency > 0.0 else 0.0
        h_lcb = committee_mean_trust
        return ChainStepResult(
            queue_next=float(queue_next),
            service_capacity=float(served),
            arrival_hat=float(self.arrival_hat),
            wait_to_fill=float(wait_to_fill),
            effective_batch=float(effective_batch),
            mu_eff=float(mu_eff),
            queue_latency=float(queue_latency),
            batch_latency=float(batch_latency),
            consensus_latency=float(consensus_latency),
            total_latency=float(total_latency),
            committee_mean_trust=float(committee_mean_trust),
            unsafe=int(unsafe),
            timeout_failure=int(timeout_failure),
            success=int(success),
            margin_cost=float(margin_cost),
            block_slack=float(block_slack),
            tps=float(tps),
            h_lcb=float(h_lcb),
        )
