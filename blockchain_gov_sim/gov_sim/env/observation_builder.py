"""观测构造模块。"""

from __future__ import annotations

import numpy as np

from gov_sim.constants import EPS


STATE_VECTOR_DIM = 13


def state_vector_dim() -> int:
    """返回冻结版状态向量长度。"""

    return STATE_VECTOR_DIM


def build_state_vector(
    A_e: float,
    Q_e: float,
    RTT_e: float,
    delta_RTT_e: float,
    churn_ratio_e: float,
    online_ratio_e: float,
    mean_trust_e: float,
    std_trust_e: float,
    n_045: int,
    n_050: int,
    n_055: int,
    n_060: int,
    previous_action_idx: int,
    A_max: float,
    Q_max: float,
    RTT_max: float,
    N_RSU: int,
) -> np.ndarray:
    """构造冻结版 13 维状态向量。"""

    return np.asarray(
        [
            float(A_e) / max(float(A_max), EPS),
            float(Q_e) / max(float(Q_max), EPS),
            float(RTT_e) / max(float(RTT_max), EPS),
            float(delta_RTT_e) / max(float(RTT_max), EPS),
            float(churn_ratio_e),
            float(online_ratio_e),
            float(mean_trust_e),
            float(std_trust_e),
            float(n_045) / max(float(N_RSU), 1.0),
            float(n_050) / max(float(N_RSU), 1.0),
            float(n_055) / max(float(N_RSU), 1.0),
            float(n_060) / max(float(N_RSU), 1.0),
            float(previous_action_idx) / 239.0,
        ],
        dtype=np.float32,
    )
