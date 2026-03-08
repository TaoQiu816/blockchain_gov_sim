"""链侧模型测试。"""

from __future__ import annotations

import numpy as np

from gov_sim.modules.chain_model import ChainModel
from gov_sim.utils.io import load_config


def test_chain_model_non_negative_outputs() -> None:
    """正常情况下链侧输出应保持非负。"""
    config = load_config("configs/default.yaml")
    chain = ChainModel(config["chain"], h_min=float(config["env"]["h_min"]))
    result = chain.step(
        queue_size=100,
        arrivals=50,
        committee=np.array([0, 1, 2, 3, 4, 5, 6]),
        committee_size=7,
        batch_size=128,
        tau_ms=40,
        rtt=20.0,
        churn=0.1,
        uptime=np.ones(10, dtype=np.float32),
        malicious=np.zeros(10, dtype=np.int8),
    )
    assert result.queue_next >= 0.0
    assert result.total_latency >= 0.0
    assert result.service_capacity >= 0.0


def test_chain_model_timeout_blocks_success() -> None:
    """共识成功不能只看诚实比例，超时也必须导致失败。"""

    config = load_config("configs/default.yaml")
    chain = ChainModel(config["chain"], h_min=float(config["env"]["h_min"]))
    result = chain.step(
        queue_size=0,
        arrivals=32,
        committee=np.array([0, 1, 2, 3, 4, 5, 6]),
        committee_size=7,
        batch_size=128,
        tau_ms=20,
        rtt=500.0,
        churn=0.8,
        uptime=np.ones(10, dtype=np.float32),
        malicious=np.zeros(10, dtype=np.int8),
    )
    assert result.honest_ratio >= float(config["env"]["h_min"])
    assert result.timeout_failure == 1
    assert result.success == 0
