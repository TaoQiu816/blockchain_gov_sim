"""链侧性能与安全模型。

该文件把治理动作 `(m, b, tau, theta)` 与当前场景状态映射为：

- 排队/批处理/共识时延
- 安全破缺与超时失败
- 有效服务量与队列更新
- TPS 与总确认时延

它对应第四章的“联盟链链侧治理”主体，不涉及 DAG 卸载本身。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ChainStepResult:
    """链侧一步执行结果。

    为了便于审计，环境会将这里的大部分字段原样写入 `info`。
    """

    queue_next: float
    service_capacity: float
    queue_latency: float
    batch_latency: float
    consensus_latency: float
    total_latency: float
    honest_ratio: float
    unsafe: int
    success: int
    leader_unstable: int
    timeout_failure: int
    tps: float


class ChainModel:
    """联盟链单周期性能/安全近似模型。"""

    def __init__(self, config: dict[str, Any], h_min: float) -> None:
        self.cfg = config
        self.h_min = h_min
        self.lambda_hat = 1.0

    def reset(self) -> None:
        self.lambda_hat = 1.0

    def step(
        self,
        queue_size: float,
        arrivals: int,
        committee: np.ndarray,
        committee_size: int,
        batch_size: int,
        tau_ms: int,
        rtt: float,
        churn: float,
        uptime: np.ndarray,
        malicious: np.ndarray,
    ) -> ChainStepResult:
        """计算一次治理动作下的链侧执行结果。

        公式对应关系：
        - `S_tilde = min(Q_e + A_e, b_e)`
        - `L_queue = c_q * Q_e / (S_tilde + eps)`
        - `L_batch = min(tau_e, [b_e-(Q_e+A_e)]_+ / (lambda_hat + eps))`
        - `L_cons` 同时受 RTT、churn、committee size、leader instability 影响
        - `Z_e = 1[h_e >= h_min] * 1[L_cons <= tau_view]`
        - `S_e = Z_e * S_tilde`
        - `Q_{e+1} = max(0, Q_e + A_e - S_e)`

        注意：
        共识成功不是只看诚实比例，而是“诚实比例 + 超时”双条件。
        """

        self.lambda_hat = float(self.cfg["lambda_ewma"]) * self.lambda_hat + (1.0 - float(self.cfg["lambda_ewma"])) * float(arrivals)
        staged = min(queue_size + arrivals, batch_size)
        queue_latency = float(self.cfg["cq"]) * queue_size / max(staged, 1.0)
        batch_slack = max(batch_size - (queue_size + arrivals), 0.0)
        batch_latency = min(float(tau_ms), batch_slack / max(self.lambda_hat, 1.0))
        leader_unstable = 0
        if committee.size > 0:
            leader_unstable = int(float(np.min(uptime[committee])) < float(self.cfg["leader_instability_threshold"]))
        consensus_latency = (
            float(self.cfg["c0"])
            + float(self.cfg["c1"]) * rtt
            + float(self.cfg["c2"]) * committee_size * max(committee_size - 1, 0) * float(self.cfg["l_ctrl"]) / max(float(self.cfg["b_ctrl"]), 1.0)
            + float(self.cfg["c3"]) * churn * committee_size
            + float(self.cfg["c4"]) * leader_unstable
        )
        honest_ratio = 1.0
        if committee_size > 0:
            honest_ratio = float(np.mean(1 - malicious[committee]))
        unsafe = int(honest_ratio < self.h_min)
        tau_view = max(float(self.cfg["min_view_timeout"]), float(tau_ms) * float(self.cfg["tau_view_factor"]))
        success = int((honest_ratio >= self.h_min) and (consensus_latency <= tau_view))
        timeout_failure = int((honest_ratio >= self.h_min) and (consensus_latency > tau_view))
        service_capacity = float(success * staged)
        queue_next = max(0.0, queue_size + arrivals - service_capacity)
        total_latency = queue_latency + batch_latency + consensus_latency + (1 - success) * float(self.cfg["l_pen"])
        tps = service_capacity / max(total_latency / 1000.0, 1.0e-6)
        return ChainStepResult(
            queue_next=queue_next,
            service_capacity=service_capacity,
            queue_latency=queue_latency,
            batch_latency=batch_latency,
            consensus_latency=consensus_latency,
            total_latency=total_latency,
            honest_ratio=honest_ratio,
            unsafe=unsafe,
            success=success,
            leader_unstable=leader_unstable,
            timeout_failure=timeout_failure,
            tps=tps,
        )
