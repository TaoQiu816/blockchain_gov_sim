"""评估指标聚合模块。

该文件负责把 step-level `info` 聚合为论文实验直接可用的 summary 指标，
例如 unsafe rate、pollute rate、F1/AUC、queue peak、recovery time 等。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import warnings

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


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
        metrics = {
            "unsafe_rate": float(np.mean([row["unsafe"] for row in self.rows])),
            "pollute_rate": float(np.mean([row["pollute_rate"] for row in self.rows])),
            "tps": float(np.mean([row["tps"] for row in self.rows])),
            "mean_latency": float(np.mean([row["L_bar_e"] for row in self.rows])),
            "consensus_failure_rate": float(np.mean([1 - row["Z_e"] for row in self.rows])),
            "timeout_failure_rate": float(np.mean([row["timeout_failure"] for row in self.rows])),
            "queue_peak": float(np.max([row["Q_e"] for row in self.rows])),
            "eligible_size_mean": float(np.mean([row["eligible_size"] for row in self.rows])),
        }
        # 为论文表格保留一组更接近章节记法的别名字段。
        metrics["R_unsafe"] = metrics["unsafe_rate"]
        metrics["R_pollute"] = metrics["pollute_rate"]
        metrics["TPS"] = metrics["tps"]
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
        actions = np.asarray([[row["m_e"], row["b_e"], row["tau_e"], row["theta_e"]] for row in self.rows], dtype=np.float32)
        if len(actions) > 1:
            metrics["oscillation_index"] = float(np.mean(np.abs(np.diff(actions, axis=0))))
        else:
            metrics["oscillation_index"] = 0.0
        metrics["eligible_size_distribution"] = [int(v) for v in np.asarray([row["eligible_size"] for row in self.rows], dtype=int)]
        return metrics
