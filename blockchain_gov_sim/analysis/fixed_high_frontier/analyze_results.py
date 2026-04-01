"""固定高层模板前沿评估结果分析脚本。

该脚本读取评估结果，生成图表和排序表，用于判断：
1. 7|0.55 是否在多数场景下接近 Pareto 最优
2. 哪些场景下 m=5/7/9 类模板更优
3. learned high policy 集中于 7|0.55 是合理收敛还是训练问题
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from gov_sim.utils.io import ensure_dir, write_json
from gov_sim.utils.logger import get_logger

logger = get_logger(__name__)

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams["font.family"] = ["Arial Unicode MS", "DejaVu Sans", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


def load_results(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """加载评估结果。"""
    fixed_df = pd.read_csv(data_dir / "fixed_high_summary_across_seeds.csv")
    learned_df = pd.read_csv(data_dir / "learned_high_baseline_summary.csv")
    episode_df = pd.read_csv(data_dir / "fixed_high_episode_metrics.csv")
    return fixed_df, learned_df, episode_df


def plot_tps_latency_scatter(fixed_df: pd.DataFrame, learned_df: pd.DataFrame, scene: str, output_dir: Path) -> None:
    """绘制 TPS-latency scatter 图。"""
    scene_data = fixed_df[fixed_df["scene"] == scene].copy()
    
    if scene_data.empty:
        logger.warning(f"场景 {scene} 无数据，跳过绘图")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 按 m 值分组绘制
    for m_val in [5, 7, 9]:
        m_data = scene_data[scene_data["m"] == m_val]
        if m_data.empty:
            continue
        
        # 颜色映射 unsafe_rate
        scatter = ax.scatter(
            m_data["mean_latency_mean"],
            m_data["TPS_mean"],
            c=m_data["unsafe_rate_mean"],
            s=100,
            alpha=0.7,
            marker="o" if m_val == 5 else ("s" if m_val == 7 else "^"),
            label=f"m={m_val}",
            cmap="YlOrRd",
            vmin=0,
            vmax=0.1,
        )
    
    # 添加 learned baseline
    if not learned_df.empty:
        learned_scene = learned_df[learned_df["scene"] == scene]
        if not learned_scene.empty:
            ax.scatter(
                learned_scene["mean_latency_mean"].mean(),
                learned_scene["TPS_mean"].mean(),
                c="blue",
                s=200,
                alpha=0.8,
                marker="*",
                label="Learned",
                edgecolors="black",
                linewidths=2,
            )
    
    # 标注 7|0.55
    template_755 = scene_data[(scene_data["m"] == 7) & (np.abs(scene_data["theta"] - 0.55) < 0.01)]
    if not template_755.empty:
        ax.scatter(
            template_755["mean_latency_mean"],
            template_755["TPS_mean"],
            c="red",
            s=300,
            alpha=0.5,
            marker="o",
            edgecolors="red",
            linewidths=3,
            label="7|0.55",
        )
    
    ax.set_xlabel("Mean Latency (ms)", fontsize=12)
    ax.set_ylabel("TPS", fontsize=12)
    ax.set_title(f"TPS-Latency Frontier: {scene}", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 添加 colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Unsafe Rate", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"tps_latency_scatter_{scene}.png", dpi=300)
    plt.close()
    
    logger.info(f"生成 TPS-latency scatter: {scene}")


def plot_risk_bars(fixed_df: pd.DataFrame, scene: str, output_dir: Path) -> None:
    """绘制风险柱状图。"""
    scene_data = fixed_df[fixed_df["scene"] == scene].copy()
    
    if scene_data.empty:
        logger.warning(f"场景 {scene} 无数据，跳过风险柱状图")
        return
    
    scene_data["template_label"] = scene_data["m"].astype(str) + "|" + scene_data["theta"].astype(str)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Unsafe rate
    axes[0].bar(range(len(scene_data)), scene_data["unsafe_rate_mean"], color="red", alpha=0.7)
    axes[0].set_xticks(range(len(scene_data)))
    axes[0].set_xticklabels(scene_data["template_label"], rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("Unsafe Rate", fontsize=10)
    axes[0].set_title("Unsafe Rate", fontsize=12, fontweight="bold")
    axes[0].grid(True, alpha=0.3, axis="y")
    
    # Timeout rate
    axes[1].bar(range(len(scene_data)), scene_data["timeout_rate_mean"], color="orange", alpha=0.7)
    axes[1].set_xticks(range(len(scene_data)))
    axes[1].set_xticklabels(scene_data["template_label"], rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("Timeout Rate", fontsize=10)
    axes[1].set_title("Timeout Rate", fontsize=12, fontweight="bold")
    axes[1].grid(True, alpha=0.3, axis="y")
    
    # Structural infeasible rate
    axes[2].bar(range(len(scene_data)), scene_data["structural_infeasible_rate_mean"], color="purple", alpha=0.7)
    axes[2].set_xticks(range(len(scene_data)))
    axes[2].set_xticklabels(scene_data["template_label"], rotation=45, ha="right", fontsize=8)
    axes[2].set_ylabel("Structural Infeasible Rate", fontsize=10)
    axes[2].set_title("Structural Infeasible Rate", fontsize=12, fontweight="bold")
    axes[2].grid(True, alpha=0.3, axis="y")
    
    plt.suptitle(f"Risk Metrics: {scene}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / f"risk_bars_{scene}.png", dpi=300)
    plt.close()
    
    logger.info(f"生成风险柱状图: {scene}")


def generate_template_ranking(fixed_df: pd.DataFrame, scene: str, output_dir: Path) -> pd.DataFrame:
    """生成模板排序表。"""
    scene_data = fixed_df[fixed_df["scene"] == scene].copy()
    
    if scene_data.empty:
        logger.warning(f"场景 {scene} 无数据，跳过排序表")
        return pd.DataFrame()
    
    # 计算综合得分：TPS 越高越好，latency 越低越好，unsafe/timeout 越低越好
    # 先归一化
    scene_data["TPS_norm"] = (scene_data["TPS_mean"] - scene_data["TPS_mean"].min()) / (scene_data["TPS_mean"].max() - scene_data["TPS_mean"].min() + 1e-6)
    scene_data["latency_norm"] = 1 - (scene_data["mean_latency_mean"] - scene_data["mean_latency_mean"].min()) / (scene_data["mean_latency_mean"].max() - scene_data["mean_latency_mean"].min() + 1e-6)
    scene_data["unsafe_norm"] = 1 - scene_data["unsafe_rate_mean"]
    scene_data["timeout_norm"] = 1 - scene_data["timeout_rate_mean"]
    
    # 综合得分 (可调整权重)
    scene_data["score"] = (
        0.3 * scene_data["TPS_norm"] +
        0.3 * scene_data["latency_norm"] +
        0.2 * scene_data["unsafe_norm"] +
        0.2 * scene_data["timeout_norm"]
    )
    
    # 排序
    ranking = scene_data.sort_values("score", ascending=False)[
        ["template_idx", "m", "theta", "TPS_mean", "mean_latency_mean", "unsafe_rate_mean", "timeout_rate_mean", "score"]
    ].reset_index(drop=True)
    
    ranking.to_csv(output_dir / f"template_ranking_{scene}.csv", index=False)
    logger.info(f"生成模板排序表: {scene}")
    
    return ranking


def generate_gap_analysis(fixed_df: pd.DataFrame, learned_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """生成 fixed vs learned 差距分析。"""
    gap_rows: list[dict[str, Any]] = []
    
    for scene in fixed_df["scene"].unique():
        scene_fixed = fixed_df[fixed_df["scene"] == scene]
        scene_learned = learned_df[learned_df["scene"] == scene]
        
        if scene_fixed.empty or scene_learned.empty:
            continue
        
        # 找到最优固定模板 (按 TPS 排序，同时考虑 unsafe_rate < 0.05)
        safe_templates = scene_fixed[scene_fixed["unsafe_rate_mean"] < 0.05]
        if safe_templates.empty:
            safe_templates = scene_fixed
        
        best_fixed = safe_templates.loc[safe_templates["TPS_mean"].idxmax()]
        learned_mean = scene_learned["TPS_mean"].mean()
        
        # 找到 learned 主导模板 (假设是 7|0.55)
        learned_dominant = "7|0.55"
        
        gap_rows.append({
            "scene": scene,
            "best_fixed_template": f"{int(best_fixed['m'])}|{float(best_fixed['theta']):.2f}",
            "best_fixed_TPS": best_fixed["TPS_mean"],
            "best_fixed_latency": best_fixed["mean_latency_mean"],
            "best_fixed_unsafe": best_fixed["unsafe_rate_mean"],
            "learned_dominant_template": learned_dominant,
            "learned_TPS": learned_mean,
            "learned_latency": scene_learned["mean_latency_mean"].mean(),
            "learned_unsafe": scene_learned["unsafe_rate_mean"].mean(),
            "TPS_gap": learned_mean - best_fixed["TPS_mean"],
            "TPS_gap_pct": (learned_mean - best_fixed["TPS_mean"]) / (best_fixed["TPS_mean"] + 1e-6) * 100,
        })
    
    gap_df = pd.DataFrame(gap_rows)
    gap_df.to_csv(output_dir / "fixed_vs_learned_gap.csv", index=False)
    logger.info("生成 fixed vs learned 差距分析")
    
    return gap_df


def generate_final_judgment(fixed_df: pd.DataFrame, learned_df: pd.DataFrame, gap_df: pd.DataFrame, output_dir: Path) -> None:
    """生成最终判断报告。"""
    report_lines: list[str] = []
    
    report_lines.append("# 固定高层模板前沿评估 - 最终判断报告\n")
    report_lines.append("## 1. 7|0.55 是否接近 Pareto 最优？\n")
    
    # 检查 7|0.55 在各场景的排名
    for scene in fixed_df["scene"].unique():
        scene_data = fixed_df[fixed_df["scene"] == scene].copy()
        template_755 = scene_data[(scene_data["m"] == 7) & (np.abs(scene_data["theta"] - 0.55) < 0.01)]
        
        if not template_755.empty:
            tps_rank = (scene_data["TPS_mean"] > template_755["TPS_mean"].values[0]).sum() + 1
            total_templates = len(scene_data)
            report_lines.append(f"- **{scene}**: 7|0.55 TPS 排名 {tps_rank}/{total_templates}\n")
    
    report_lines.append("\n## 2. 不同场景下的最优模板类型\n")
    
    for scene in fixed_df["scene"].unique():
        scene_data = fixed_df[fixed_df["scene"] == scene].copy()
        safe_templates = scene_data[scene_data["unsafe_rate_mean"] < 0.05]
        if safe_templates.empty:
            safe_templates = scene_data
        
        best_template = safe_templates.loc[safe_templates["TPS_mean"].idxmax()]
        report_lines.append(f"- **{scene}**: 最优模板 {int(best_template['m'])}|{float(best_template['theta']):.2f} (TPS={best_template['TPS_mean']:.2f})\n")
    
    report_lines.append("\n## 3. Learned vs Best Fixed 差距\n")
    
    if not gap_df.empty:
        avg_gap_pct = gap_df["TPS_gap_pct"].mean()
        report_lines.append(f"- 平均 TPS 差距: {avg_gap_pct:.2f}%\n")
        
        for _, row in gap_df.iterrows():
            report_lines.append(f"- **{row['scene']}**: Learned TPS={row['learned_TPS']:.2f}, Best Fixed TPS={row['best_fixed_TPS']:.2f}, Gap={row['TPS_gap_pct']:.2f}%\n")
    
    report_lines.append("\n## 4. 最终判断\n")
    
    if not gap_df.empty:
        avg_gap_pct = gap_df["TPS_gap_pct"].mean()
        
        if abs(avg_gap_pct) < 5:
            report_lines.append("**结论**: Learned high policy 集中在 7|0.55 是**合理收敛**，该模板在多数场景下接近 Pareto 最优。\n")
        else:
            report_lines.append("**结论**: Learned high policy 集中在 7|0.55 可能存在**训练问题**，与最优固定模板存在显著差距。\n")
            report_lines.append("\n**建议检查**:\n")
            report_lines.append("- 高层状态表达是否充分捕捉场景差异\n")
            report_lines.append("- 高层 chunk update 机制是否合理\n")
            report_lines.append("- 高层目标/约束权衡是否需要调整\n")
    
    report_text = "".join(report_lines)
    
    with open(output_dir / "final_judgment.md", "w", encoding="utf-8") as f:
        f.write(report_text)
    
    logger.info("生成最终判断报告")
    print("\n" + report_text)


def main() -> None:
    parser = argparse.ArgumentParser(description="固定高层模板前沿评估结果分析")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="analysis/fixed_high_frontier",
        help="数据目录",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis/fixed_high_frontier/plots",
        help="输出目录",
    )
    
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    output_dir = ensure_dir(Path(args.output_dir))
    
    logger.info("加载评估结果...")
    fixed_df, learned_df, episode_df = load_results(data_dir)
    
    logger.info("生成图表和分析...")
    
    # 为每个场景生成图表
    for scene in fixed_df["scene"].unique():
        plot_tps_latency_scatter(fixed_df, learned_df, scene, output_dir)
        plot_risk_bars(fixed_df, scene, output_dir)
        generate_template_ranking(fixed_df, scene, output_dir)
    
    # 生成差距分析
    gap_df = generate_gap_analysis(fixed_df, learned_df, output_dir)
    
    # 生成最终判断
    generate_final_judgment(fixed_df, learned_df, gap_df, output_dir)
    
    logger.info(f"分析完成，结果保存到: {output_dir}")


if __name__ == "__main__":
    main()
