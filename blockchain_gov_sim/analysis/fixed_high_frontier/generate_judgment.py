"""生成固定高层模板前沿评估的判断报告。"""

from __future__ import annotations

import pandas as pd
from pathlib import Path


def analyze_template_performance(data_path: Path) -> None:
    """分析固定模板性能并生成判断报告。"""
    
    # 读取数据
    df = pd.read_csv(data_path / "fixed_high_summary_across_seeds.csv")
    
    # 为每个场景找出最优模板
    report_lines = []
    report_lines.append("# 固定高层模板前沿评估判断报告\n")
    report_lines.append("## 评估概���\n")
    report_lines.append("- **评估模式**: minimal (5 episodes per scene)")
    report_lines.append("- **Checkpoints**: 5 个训练 seeds (seed42-46)")
    report_lines.append("- **场景**: load_shock, high_rtt_burst, churn_burst, malicious_burst")
    report_lines.append("- **模板数量**: 12 个 (m=5/7/9, theta=0.45/0.50/0.55/0.60)\n")
    
    report_lines.append("## 各场景最优模板分析\n")
    
    for scene in df["scene"].unique():
        scene_df = df[df["scene"] == scene].copy()
        scene_df = scene_df.sort_values("TPS_mean", ascending=False)
        
        report_lines.append(f"### {scene}\n")
        report_lines.append("**TPS 排名 (Top 5):**\n")
        
        for i, row in scene_df.head(5).iterrows():
            m, theta = int(row["m"]), row["theta"]
            tps = row["TPS_mean"]
            latency = row["mean_latency_mean"]
            unsafe = row["unsafe_rate_mean"]
            
            marker = " ⭐" if (m == 7 and theta == 0.55) else ""
            report_lines.append(
                f"{int(row['template_idx'])+1}. **m={m}, θ={theta}**{marker}: "
                f"TPS={tps:.1f}, Latency={latency:.2f}ms, Unsafe={unsafe:.4f}"
            )
        
        # 找到 7|0.55 的排名
        template_755 = scene_df[
            (scene_df["m"] == 7) & (scene_df["theta"] == 0.55)
        ]
        if not template_755.empty:
            rank = (scene_df["TPS_mean"] > template_755["TPS_mean"].values[0]).sum() + 1
            tps_755 = template_755["TPS_mean"].values[0]
            best_tps = scene_df["TPS_mean"].max()
            gap_pct = (best_tps - tps_755) / best_tps * 100
            
            report_lines.append(f"\n**7|0.55 分析:**")
            report_lines.append(f"- 排名: {rank}/12")
            report_lines.append(f"- TPS: {tps_755:.1f}")
            report_lines.append(f"- 与最优差距: {gap_pct:.2f}%")
        
        report_lines.append("")
    
    # 总体判断
    report_lines.append("## 总体判断\n")
    
    # 统计 7|0.55 在各场景的排名
    ranks_755 = []
    for scene in df["scene"].unique():
        scene_df = df[df["scene"] == scene].copy()
        scene_df = scene_df.sort_values("TPS_mean", ascending=False)
        template_755 = scene_df[
            (scene_df["m"] == 7) & (scene_df["theta"] == 0.55)
        ]
        if not template_755.empty:
            rank = (scene_df["TPS_mean"] > template_755["TPS_mean"].values[0]).sum() + 1
            ranks_755.append(rank)
    
    avg_rank = sum(ranks_755) / len(ranks_755) if ranks_755 else 0
    
    report_lines.append(f"### 7|0.55 模板表现总结\n")
    report_lines.append(f"- **平均排名**: {avg_rank:.1f}/12")
    report_lines.append(f"- **各场景排名**: {ranks_755}")
    
    if avg_rank <= 3:
        report_lines.append("\n**结论**: ✅ 7|0.55 在多数场景下接近 Pareto 最优，learned high policy 收敛到该模板是**合理的**。")
    elif avg_rank <= 6:
        report_lines.append("\n**结论**: ⚠️ 7|0.55 表现中等，learned high policy 可能存在**轻微次优**，但仍在可接受范围内。")
    else:
        report_lines.append("\n**结论**: ❌ 7|0.55 表现较差，learned high policy 收敛到该模板可能存在**训练问题**。")
    
    # 场景特异性分析
    report_lines.append("\n### 场景特异性分析\n")
    
    for scene in df["scene"].unique():
        scene_df = df[df["scene"] == scene].copy()
        best_template = scene_df.loc[scene_df["TPS_mean"].idxmax()]
        m_best = int(best_template["m"])
        theta_best = best_template["theta"]
        
        report_lines.append(f"- **{scene}**: 最优模板为 m={m_best}, θ={theta_best}")
    
    # 保存报告
    output_path = data_path / "JUDGMENT_REPORT.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    
    print(f"判断报告已保存到: {output_path}")
    
    # 打印到控制台
    print("\n" + "\n".join(report_lines))


if __name__ == "__main__":
    data_dir = Path("analysis/fixed_high_frontier")
    analyze_template_performance(data_dir)
