"""分层控制器的高层/低层状态构造。"""

from __future__ import annotations

import numpy as np

from gov_sim.env.observation_builder import SUMMARY_FEATURE_KEYS
from gov_sim.env.gov_env import BlockchainGovEnv
from gov_sim.hierarchical.spec import HIGH_LEVEL_M_VALUES, HIGH_LEVEL_THETA_VALUES, HighLevelAction, HierarchicalActionCodec

SUMMARY_INDEX = {key: idx for idx, key in enumerate(SUMMARY_FEATURE_KEYS)}
LOW_STATE_DIM = 11
HIGH_STATE_KEYS: tuple[str, ...] = (
    "eligible_size",
    "committee_honest_lcb",
    "committee_honest_ratio",
    "final_score_mean",
    "final_score_std",
    "final_score_p25",
    "final_score_p50",
    "final_score_p75",
    "con_score_mean",
    "con_score_std",
    "con_score_p25",
    "con_score_p50",
    "con_score_p75",
    "rec_score_mean",
    "rec_score_std",
    "rec_score_p25",
    "rec_score_p50",
    "rec_score_p75",
    "unsafe_ema",
    "margin_cost_ema",
    "rtt",
    "rtt_ema",
    "delta_rtt",
    "timeout_ema",
    "l_cons_ema",
    "l_batch_ema",
    "eligible_per_current_m",
    "legal_high_template_count",
    "legal_m_5_count",
    "legal_m_7_count",
    "legal_m_9_count",
    "legal_theta_045_count",
    "legal_theta_050_count",
    "legal_theta_055_count",
    "legal_theta_060_count",
    "structural_infeasible_ema",
    "churn_ema",
    "online_ratio",
    "queue",
    "delta_queue",
    "arrival",
    "arrival_ema",
    "served",
    "queue_over_served_ratio",
)
HIGH_STATE_DIM = len(HIGH_STATE_KEYS)
_HIGH_CODEC = HierarchicalActionCodec()


def _summary_value(env: BlockchainGovEnv, key: str) -> float:
    if env.current_obs is None:
        raise RuntimeError("Environment observation is not initialized.")
    return float(env.current_obs["state"][SUMMARY_INDEX[key]])


def _distribution_summary(values: np.ndarray) -> list[float]:
    if values.size == 0:
        return [0.0] * 5
    return [
        float(np.mean(values)),
        float(np.std(values)),
        float(np.quantile(values, 0.25)),
        float(np.quantile(values, 0.50)),
        float(np.quantile(values, 0.75)),
    ]


def _legal_template_breakdown(env: BlockchainGovEnv) -> tuple[int, dict[int, int], dict[float, int]]:
    if env.current_mask is None:
        raise RuntimeError("Environment action mask is not initialized.")
    by_m = {int(m): 0 for m in HIGH_LEVEL_M_VALUES}
    by_theta = {float(theta): 0 for theta in HIGH_LEVEL_THETA_VALUES}
    total = 0
    for high_action in _HIGH_CODEC.high_actions:
        is_legal = False
        for low_action in _HIGH_CODEC.low_actions:
            flat_idx = _HIGH_CODEC.flat_index(high_action=high_action, low_action=low_action)
            if int(env.current_mask[flat_idx]) == 1:
                is_legal = True
                break
        if is_legal:
            total += 1
            by_m[int(high_action.m)] += 1
            by_theta[float(high_action.theta)] += 1
    return total, by_m, by_theta


def _high_state_values(env: BlockchainGovEnv) -> list[float]:
    if env.current_snapshot is None or env.current_scenario is None:
        raise RuntimeError("Environment state is not initialized.")
    final_summary = _distribution_summary(env.current_snapshot.final_scores)
    con_summary = _distribution_summary(env.current_snapshot.score_dim["con"])
    rec_summary = _distribution_summary(env.current_snapshot.score_dim["rec"])
    legal_total, legal_by_m, legal_by_theta = _legal_template_breakdown(env)
    eligible_size = _summary_value(env, "eligible_size")
    current_m = max(float(env.prev_action.m), 1.0)
    queue_now = _summary_value(env, "Q_e")
    served_prev = float(env.prev_served)
    return [
        eligible_size,
        float(env.prev_honest_lcb),
        float(env.prev_committee_honest_ratio),
        *final_summary,
        *con_summary,
        *rec_summary,
        float(env.unsafe_ema),
        float(env.margin_cost_ema),
        _summary_value(env, "RTT_e"),
        _summary_value(env, "RTT_ema"),
        _summary_value(env, "delta_RTT"),
        float(env.timeout_ema),
        float(env.consensus_latency_ema),
        float(env.batch_latency_ema),
        float(eligible_size / current_m),
        float(legal_total),
        float(legal_by_m[5]),
        float(legal_by_m[7]),
        float(legal_by_m[9]),
        float(legal_by_theta[0.45]),
        float(legal_by_theta[0.50]),
        float(legal_by_theta[0.55]),
        float(legal_by_theta[0.60]),
        float(env.structural_infeasible_ema),
        _summary_value(env, "chi_ema"),
        float(np.mean(env.current_scenario.online)),
        queue_now,
        _summary_value(env, "delta_Q"),
        _summary_value(env, "A_e"),
        _summary_value(env, "lambda_hat"),
        served_prev,
        float(queue_now / max(served_prev, 1.0e-8)),
    ]


def build_high_level_state(env: BlockchainGovEnv) -> np.ndarray:
    """高层状态：安全结构摘要。"""

    return np.asarray(_high_state_values(env), dtype=np.float32)


def build_low_level_state(env: BlockchainGovEnv, high_action: HighLevelAction) -> np.ndarray:
    """低层状态：性能执行摘要。"""

    return np.asarray(
        [
            _summary_value(env, "A_e"),
            _summary_value(env, "Q_e"),
            _summary_value(env, "delta_Q"),
            _summary_value(env, "RTT_e"),
            _summary_value(env, "delta_RTT"),
            _summary_value(env, "chi_e"),
            _summary_value(env, "delta_chi"),
            float(env.prev_action.b),
            float(env.prev_action.tau),
            float(high_action.m),
            float(high_action.theta),
        ],
        dtype=np.float32,
    )
