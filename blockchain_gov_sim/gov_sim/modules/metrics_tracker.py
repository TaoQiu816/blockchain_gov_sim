"""评估指标聚合模块。

该文件负责把 step-level `info` 聚合为论文实验直接可用的 summary 指标，
例如 unsafe rate、pollute rate、F1/AUC、queue peak、recovery time 等。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from collections import Counter
import warnings

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def _split_invalid_counts(row: dict[str, Any]) -> tuple[int, int]:
    structural_infeasible = int(bool(row.get("structural_infeasible", 0)))
    policy_invalid = int(bool(row.get("policy_invalid", 0))) if not structural_infeasible else 0
    return policy_invalid, structural_infeasible


@dataclass
class MetricsTracker:
    """把逐步审计指标累积成 episode / experiment summary。"""

    rows: list[dict[str, Any]] = field(default_factory=list)

    def reset(self) -> None:
        self.rows.clear()

    def update(self, info: dict[str, Any]) -> None:
        self.rows.append(info.copy())

    def summary(self) -> dict[str, Any]:
        """汇总当前缓存的 step 级记录。"""

        if not self.rows:
            return {}
        total_steps = max(len(self.rows), 1)
        invalid_splits = [_split_invalid_counts(row) for row in self.rows]
        committee_rows = [row for row in self.rows if int(row.get("committee_size", 0)) > 0]
        committee_steps = max(len(committee_rows), 1)
        metrics = {
            "unsafe_rate_all_steps": float(np.mean([row["unsafe"] for row in self.rows])),
            "pollute_rate": float(np.mean([row["pollute_rate"] for row in self.rows])),
            "tps_all_steps": float(np.mean([row["tps"] for row in self.rows])),
            "mean_latency": float(np.mean([row["L_bar_e"] for row in self.rows])),
            "consensus_failure_rate": float(np.mean([1 - row.get("Z_e", int(not row.get("timeout", 0))) for row in self.rows])),
            "timeout_rate_all_steps": float(np.mean([row.get("timeout", row.get("timeout_failure", 0)) for row in self.rows])),
            "queue_peak": float(np.max([row["Q_e"] for row in self.rows])),
            "eligible_size_mean_all_steps": float(np.mean([row["eligible_size"] for row in self.rows])),
            "committee_mean_trust_mean_committee_steps": float(
                np.mean([row.get("committee_mean_trust", row.get("h_LCB_e", row.get("h_LCB", 0.0))) for row in committee_rows])
            )
            if committee_rows
            else 0.0,
            "unsafe_rate_committee_steps": float(np.mean([row["unsafe"] for row in committee_rows])) if committee_rows else 0.0,
            "timeout_rate_committee_steps": float(np.mean([row.get("timeout", row.get("timeout_failure", 0)) for row in committee_rows]))
            if committee_rows
            else 0.0,
            "committee_honest_ratio_mean": float(np.mean([row.get("committee_honest_ratio", row.get("h_e", 0.0)) for row in self.rows])),
            "effective_batch_mean": float(np.mean([row.get("B_e", 0.0) for row in self.rows])),
            "served_mean": float(np.mean([row.get("S_e", 0.0) for row in self.rows])),
            "L_queue_mean": float(np.mean([row.get("L_queue_e", 0.0) for row in self.rows])),
            "L_batch_mean": float(np.mean([row.get("L_batch_e", 0.0) for row in self.rows])),
            "L_cons_mean": float(np.mean([row.get("L_cons_e", 0.0) for row in self.rows])),
            "block_slack_mean": float(np.mean([row.get("block_slack", 0.0) for row in self.rows])),
            "invalid_action_rate": float(np.mean([row.get("invalid_action", 0) for row in self.rows])),
            "policy_invalid_rate": float(np.mean([split[0] for split in invalid_splits])),
            "structural_infeasible_rate_all_steps": float(np.mean([split[1] for split in invalid_splits])),
            "infeasible_rate": float(np.mean([row.get("infeasible", 0) for row in self.rows])),
            "used_backstop_template_rate": float(np.mean([row.get("used_backstop_template", 0) for row in self.rows])),
            "infeasible_high_template_rate": float(
                np.mean(
                    [
                        int(row.get("infeasible", 0) and row.get("infeasible_reason") == "no_legal_high_template")
                        for row in self.rows
                    ]
                )
            ),
            "committee_step_count": int(len(committee_rows)),
            "all_step_count": int(total_steps),
        }
        metrics["committee_mean_trust_mean_all_steps"] = float(
            np.mean([row.get("committee_mean_trust", row.get("h_LCB_e", row.get("h_LCB", 0.0))) for row in self.rows])
        )
        metrics["eligible_size_mean_committee_steps"] = float(np.mean([row["eligible_size"] for row in committee_rows])) if committee_rows else 0.0
        metrics["h_lcb_mean"] = metrics["committee_mean_trust_mean_committee_steps"]
        # 为论文表格保留一组更接近章节记法的别名字段。
        metrics["unsafe_rate"] = metrics["unsafe_rate_all_steps"]
        metrics["timeout_failure_rate"] = metrics["timeout_rate_all_steps"]
        metrics["timeout_rate"] = metrics["timeout_rate_all_steps"]
        metrics["structural_infeasible_rate"] = metrics["structural_infeasible_rate_all_steps"]
        metrics["eligible_size_mean"] = metrics["eligible_size_mean_all_steps"]
        metrics["committee_mean_trust_mean"] = metrics["committee_mean_trust_mean_committee_steps"]
        metrics["qualified_node_count_mean"] = metrics["eligible_size_mean_all_steps"]
        metrics["tps"] = metrics["tps_all_steps"]
        metrics["TPS"] = metrics["tps_all_steps"]
        metrics["R_unsafe"] = metrics["unsafe_rate_all_steps"]
        metrics["R_pollute"] = metrics["pollute_rate"]
        metrics["unsafe_rate_committee_steps"] = float(metrics["unsafe_rate_committee_steps"])
        metrics["timeout_rate_committee_steps"] = float(metrics["timeout_rate_committee_steps"])
        if "recovery_time" not in metrics:
            queue_series = np.asarray([row["Q_e"] for row in self.rows], dtype=np.float32)
            peak_idx = int(np.argmax(queue_series))
            threshold = 0.2 * float(np.max(queue_series))
            recovery_idx = next((idx for idx in range(peak_idx, len(queue_series)) if queue_series[idx] <= threshold), len(queue_series) - 1)
            metrics["recovery_time"] = int(recovery_idx - peak_idx)
        if all("malicious_pred" in row for row in self.rows):
            y_true = np.concatenate([np.asarray(row["malicious_true"], dtype=np.int8) for row in self.rows])
            y_pred = np.concatenate([np.asarray(row["malicious_pred"], dtype=np.int8) for row in self.rows])
            y_score = np.concatenate([1.0 - np.asarray(row["trust_scores"], dtype=np.float32) for row in self.rows])
            metrics["malicious_detection_f1"] = float(f1_score(y_true, y_pred, zero_division=0))
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    auc = float(roc_auc_score(y_true, y_score))
                metrics["malicious_detection_auc"] = 0.5 if np.isnan(auc) else auc
            except ValueError:
                metrics["malicious_detection_auc"] = 0.5
            negatives = max(int(np.sum(y_true == 0)), 1)
            metrics["false_positive_rate"] = float(np.sum((y_pred == 1) & (y_true == 0)) / negatives)
        actions = np.asarray([[row.get("mapped_m_e", row["m_e"]), row["b_e"], row["tau_e"], row["theta_e"]] for row in self.rows], dtype=np.float32)
        action_signatures = [
            f"{int(row.get('mapped_m_e', row['m_e']))}|{int(row['b_e'])}|{int(row['tau_e'])}|{float(row['theta_e']):.2f}"
            for row in self.rows
        ]
        action_counter = Counter(action_signatures)
        top_actions = action_counter.most_common(5)
        total_actions = max(sum(action_counter.values()), 1)
        metrics["dominant_action_ratio"] = float(top_actions[0][1] / total_actions) if top_actions else 0.0
        metrics["top_k_action_distribution"] = [
            {"action": action, "count": int(count), "ratio": float(count / total_actions)} for action, count in top_actions
        ]
        if len(actions) > 1:
            metrics["oscillation_index"] = float(np.mean(np.abs(np.diff(actions, axis=0))))
        else:
            metrics["oscillation_index"] = 0.0
        if all("executed_high_template" in row for row in self.rows):
            executed_templates = [str(row.get("executed_high_template", "")) for row in self.rows if str(row.get("executed_high_template", ""))]
            template_counter = Counter(executed_templates)
            template_total = max(sum(template_counter.values()), 1)
            metrics["executed_high_template_distribution"] = [
                {"action": action, "count": int(count), "ratio": float(count / template_total)}
                for action, count in template_counter.most_common(5)
            ]
        if all("executed_low_action" in row for row in self.rows):
            executed_lows = [str(row.get("executed_low_action", "")) for row in self.rows if str(row.get("executed_low_action", ""))]
            low_counter = Counter(executed_lows)
            low_total = max(sum(low_counter.values()), 1)
            metrics["executed_low_action_distribution"] = [
                {"action": action, "count": int(count), "ratio": float(count / low_total)}
                for action, count in low_counter.most_common(5)
            ]
        if all("scenario_type" in row for row in self.rows):
            scenario_counter = Counter(str(row["scenario_type"]) for row in self.rows)
            scenario_total = max(sum(scenario_counter.values()), 1)
            metrics["scenario_type_distribution"] = {
                scenario: float(count / scenario_total) for scenario, count in sorted(scenario_counter.items())
            }
            per_scenario: dict[str, dict[str, Any]] = {}
            for scenario in sorted(scenario_counter):
                scenario_actions = Counter(
                    f"{int(row.get('mapped_m_e', row['m_e']))}|{int(row['b_e'])}|{int(row['tau_e'])}|{float(row['theta_e']):.2f}"
                    for row in self.rows
                    if str(row["scenario_type"]) == scenario
                )
                scenario_total_actions = max(sum(scenario_actions.values()), 1)
                scenario_top = scenario_actions.most_common(5)
                per_scenario[scenario] = {
                    "dominant_action_ratio": float(scenario_top[0][1] / scenario_total_actions) if scenario_top else 0.0,
                    "top_k_action_distribution": [
                        {"action": action, "count": int(count), "ratio": float(count / scenario_total_actions)}
                        for action, count in scenario_top
                    ],
                }
            metrics["per_scenario_action_usage_summary"] = per_scenario
        metrics["eligible_size_distribution"] = [int(v) for v in np.asarray([row["eligible_size"] for row in self.rows], dtype=int)]
        return metrics
