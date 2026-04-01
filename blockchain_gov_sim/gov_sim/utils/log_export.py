"""训练/评估日志统一导出工具。

该模块负责将训练和评估过程中的各类日志（episode 级、step 级、audit）
统一导出为结构化的 CSV/JSON 格式，便于后续分析和可视化。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from gov_sim.utils.io import write_json


def export_training_summary(
    stage_dir: Path,
    episode_log_path: Path | None = None,
    step_log_path: Path | None = None,
    audit_path: Path | None = None,
) -> dict[str, Any]:
    """导出训练阶段的综合摘要。
    
    Args:
        stage_dir: 训练阶段输出目录
        episode_log_path: episode 级日志路径
        step_log_path: step 级日志路径
        audit_path: audit JSON 路径
    
    Returns:
        综合摘要字典
    """
    summary: dict[str, Any] = {
        "stage_dir": str(stage_dir),
        "artifacts": {},
    }
    
    # 读取 episode 级日志
    if episode_log_path and episode_log_path.exists():
        df_episode = pd.read_csv(episode_log_path)
        summary["artifacts"]["episode_log"] = str(episode_log_path)
        summary["episode_count"] = len(df_episode)
        
        # 提取关键统计
        if not df_episode.empty:
            summary["episode_stats"] = {
                "mean_reward": float(df_episode["episode_reward"].mean()),
                "mean_cost": float(df_episode["episode_cost"].mean()),
                "mean_unsafe_rate": float(df_episode["unsafe_rate"].mean()),
                "mean_tps": float(df_episode["tps"].mean()),
                "mean_latency": float(df_episode["latency"].mean()),
                "final_lambda": float(df_episode["lagrangian_lambda"].iloc[-1]),
            }
            for column in ("timeout_rate", "policy_invalid_rate", "structural_infeasible_rate", "committee_mean_trust_mean", "eligible_size_mean"):
                if column in df_episode.columns:
                    summary["episode_stats"][f"mean_{column}"] = float(df_episode[column].mean())
            reward_cols = [column for column in df_episode.columns if column.startswith("reward_component_")]
            cost_cols = [column for column in df_episode.columns if column.startswith("cost_component_")]
            if reward_cols:
                summary["episode_stats"]["reward_decomposition"] = {
                    column.replace("reward_component_", ""): float(df_episode[column].mean()) for column in reward_cols
                }
            if cost_cols:
                summary["episode_stats"]["cost_decomposition"] = {
                    column.replace("cost_component_", ""): float(df_episode[column].mean()) for column in cost_cols
                }
            
            # 添加 template/action 分布（如果存在）
            if "top_high_template" in df_episode.columns:
                template_counts = df_episode["top_high_template"].value_counts()
                summary["episode_stats"]["high_template_distribution"] = [
                    {"template": str(k), "count": int(v)} 
                    for k, v in template_counts.head(5).items()
                ]
            
            if "top_low_action" in df_episode.columns:
                action_counts = df_episode["top_low_action"].value_counts()
                summary["episode_stats"]["low_action_distribution"] = [
                    {"action": str(k), "count": int(v)} 
                    for k, v in action_counts.head(5).items()
                ]
    
    # 读取 step 级日志
    if step_log_path and step_log_path.exists():
        df_step = pd.read_csv(step_log_path)
        summary["artifacts"]["step_log"] = str(step_log_path)
        summary["step_count"] = len(df_step)
        
        if not df_step.empty:
            summary["step_stats"] = {
                "mean_reward": float(df_step["reward"].mean()),
                "mean_cost": float(df_step["cost"].mean()),
                "mean_entropy": float(df_step["entropy"].mean()) if "entropy" in df_step.columns else 0.0,
            }
            for column in ("unsafe", "timeout_failure", "policy_invalid", "structural_infeasible", "eligible_size", "committee_mean_trust"):
                if column in df_step.columns:
                    summary["step_stats"][f"mean_{column}"] = float(df_step[column].mean())
            
            # Lambda 轨迹统计
            if "lagrangian_lambda" in df_step.columns:
                lambda_series = df_step["lagrangian_lambda"]
                summary["step_stats"]["lambda_trajectory"] = {
                    "min": float(lambda_series.min()),
                    "max": float(lambda_series.max()),
                    "mean": float(lambda_series.mean()),
                    "std": float(lambda_series.std()),
                    "final": float(lambda_series.iloc[-1]),
                }
            
            # Template/Action 使用频率
            if "executed_high_template" in df_step.columns:
                template_counts = df_step["executed_high_template"].value_counts()
                total = template_counts.sum()
                summary["step_stats"]["high_template_usage"] = [
                    {"template": str(k), "count": int(v), "ratio": float(v/total)} 
                    for k, v in template_counts.items() if k
                ]
            
            if "executed_low_action" in df_step.columns:
                action_counts = df_step["executed_low_action"].value_counts()
                total = action_counts.sum()
                summary["step_stats"]["low_action_usage"] = [
                    {"action": str(k), "count": int(v), "ratio": float(v/total)} 
                    for k, v in action_counts.items() if k
                ]
            for column in ("m_e", "theta_e", "b_e", "tau_e"):
                if column in df_step.columns:
                    counts = df_step[column].value_counts()
                    total = max(counts.sum(), 1)
                    summary["step_stats"][f"{column}_distribution"] = [
                        {"value": float(k), "count": int(v), "ratio": float(v / total)} for k, v in counts.items()
                    ]
            reward_cols = [column for column in df_step.columns if column.startswith("reward_")]
            cost_cols = [column for column in df_step.columns if column.startswith("cost_")]
            if reward_cols:
                summary["step_stats"]["reward_decomposition"] = {
                    column.replace("reward_", ""): float(df_step[column].mean())
                    for column in reward_cols
                    if column not in {"reward", "reward_component"}
                }
            if cost_cols:
                summary["step_stats"]["cost_decomposition"] = {
                    column.replace("cost_", ""): float(df_step[column].mean())
                    for column in cost_cols
                    if column not in {"cost", "cost_component"}
                }
    
    # 读取 audit JSON
    if audit_path and audit_path.exists():
        import json
        with open(audit_path, "r") as f:
            audit_data = json.load(f)
        summary["artifacts"]["audit"] = str(audit_path)
        summary["audit"] = audit_data
    
    # 导出综合摘要
    summary_path = stage_dir / "training_export_summary.json"
    write_json(summary_path, summary)
    summary["artifacts"]["export_summary"] = str(summary_path)
    
    return summary


def export_evaluation_summary(
    eval_dir: Path,
    episode_csv_paths: list[Path],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """导出评估阶段的综合摘要。
    
    Args:
        eval_dir: 评估输出目录
        episode_csv_paths: episode 级 CSV 文件路径列表
        metadata: 额外的元数据
    
    Returns:
        综合摘要字典
    """
    summary: dict[str, Any] = {
        "eval_dir": str(eval_dir),
        "metadata": metadata or {},
        "artifacts": {},
        "scenarios": {},
    }
    
    for csv_path in episode_csv_paths:
        if not csv_path.exists():
            continue
        
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        
        # 从文件名提取 scenario 和 controller
        parts = csv_path.stem.split("_")
        if len(parts) >= 2:
            scenario = "_".join(parts[:-1])
            controller = parts[-1]
        else:
            scenario = csv_path.stem
            controller = "unknown"
        
        if scenario not in summary["scenarios"]:
            summary["scenarios"][scenario] = {}
        
        # 计算关键指标
        metrics = {
            "episodes": len(df),
            "mean_tps": float(df["tps"].mean()) if "tps" in df.columns else 0.0,
            "mean_latency": float(df["L_bar_e"].mean()) if "L_bar_e" in df.columns else 0.0,
            "unsafe_rate": float(df["unsafe"].mean()) if "unsafe" in df.columns else 0.0,
            "infeasible_rate": float(df["infeasible"].mean()) if "infeasible" in df.columns else 0.0,
        }
        
        # Template/Action 分布
        if "executed_high_template" in df.columns:
            template_counts = df["executed_high_template"].value_counts()
            total = max(template_counts.sum(), 1)
            metrics["high_template_distribution"] = [
                {"template": str(k), "count": int(v), "ratio": float(v/total)} 
                for k, v in template_counts.head(5).items() if k
            ]
        
        if "executed_low_action" in df.columns:
            action_counts = df["executed_low_action"].value_counts()
            total = max(action_counts.sum(), 1)
            metrics["low_action_distribution"] = [
                {"action": str(k), "count": int(v), "ratio": float(v/total)} 
                for k, v in action_counts.head(5).items() if k
            ]
        
        summary["scenarios"][scenario][controller] = metrics
        summary["artifacts"][f"{scenario}_{controller}"] = str(csv_path)
    
    # 导出综合摘要
    summary_path = eval_dir / "evaluation_export_summary.json"
    write_json(summary_path, summary)
    summary["artifacts"]["export_summary"] = str(summary_path)
    
    return summary


def generate_training_artifacts_with_export(stage_dir: Path) -> None:
    """生成训练产物并导出综合摘要。
    
    Args:
        stage_dir: 训练阶段输出目录
    """
    episode_log = stage_dir / "train_log.csv"
    step_log = stage_dir / "train_log_steps.csv"
    audit = stage_dir / "train_audit.json"
    
    export_training_summary(
        stage_dir=stage_dir,
        episode_log_path=episode_log if episode_log.exists() else None,
        step_log_path=step_log if step_log.exists() else None,
        audit_path=audit if audit.exists() else None,
    )
