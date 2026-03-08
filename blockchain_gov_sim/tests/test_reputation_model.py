"""信誉模型测试。"""

from __future__ import annotations

import numpy as np

from gov_sim.modules.evidence_generator import EvidenceBatch
from gov_sim.modules.reputation_model import ReputationModel
from gov_sim.utils.io import load_config


def test_reputation_model_outputs_valid_scores() -> None:
    """信誉输出必须位于 [0,1]，且维度权重应归一化。"""
    config = load_config("configs/default.yaml")
    model = ReputationModel(config["reputation"], num_rsus=6)
    evidence = EvidenceBatch(
        delta_s={dim: np.full(6, 2.0, dtype=np.float32) for dim in ("svc", "con", "rec", "stab")},
        delta_f={dim: np.full(6, 0.5, dtype=np.float32) for dim in ("svc", "con", "rec", "stab")},
        recommendation_matrix=np.full((6, 6), 0.8, dtype=np.float32),
        predicted_quality=np.full(6, 0.8, dtype=np.float32),
        uptime=np.full(6, 0.9, dtype=np.float32),
        pollute_rate=0.0,
    )
    snapshot = model.update(np.array([100, 10, 20, 15, 0.1, 6], dtype=np.float32), evidence)
    assert np.all(snapshot.final_scores >= 0.0)
    assert np.all(snapshot.final_scores <= 1.0)
    assert abs(sum(snapshot.dim_weights.values()) - 1.0) < 1e-5


def test_reputation_penalties_reduce_camouflaged_nodes() -> None:
    """跨维伪装和稳定性惩罚应真实压低最终信誉。"""

    config = load_config("configs/default.yaml")
    model = ReputationModel(config["reputation"], num_rsus=2)
    evidence = EvidenceBatch(
        delta_s={
            "svc": np.array([0.4, 2.2], dtype=np.float32),
            "con": np.array([2.3, 2.2], dtype=np.float32),
            "rec": np.array([2.2, 2.1], dtype=np.float32),
            "stab": np.array([0.5, 2.0], dtype=np.float32),
        },
        delta_f={
            "svc": np.array([2.1, 0.4], dtype=np.float32),
            "con": np.array([0.2, 0.3], dtype=np.float32),
            "rec": np.array([0.3, 0.4], dtype=np.float32),
            "stab": np.array([1.8, 0.3], dtype=np.float32),
        },
        recommendation_matrix=np.array([[0.2, 0.2], [0.8, 0.8]], dtype=np.float32),
        predicted_quality=np.array([0.2, 0.8], dtype=np.float32),
        uptime=np.array([0.2, 0.95], dtype=np.float32),
        pollute_rate=0.0,
    )
    snapshot = model.update(np.array([120, 50, 30, 20, 0.2, 5], dtype=np.float32), evidence)
    assert snapshot.penalties["cross_dim"][0] > 0.0
    assert snapshot.penalties["stab"][0] > 0.0
    assert snapshot.final_scores[0] < snapshot.base_scores[0]
