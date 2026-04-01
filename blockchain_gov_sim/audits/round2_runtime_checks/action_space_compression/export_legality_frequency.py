"""导出高层动作 legality 频率核验

目标：基于已有评估轨迹或环境重放，导出四个 hard 场景下所有高层候选组合的 legality ratio

统计对象：
- m ∈ {5,7,9}
- theta ∈ {0.45,0.50,0.55,0.60}
共 12 个高层组合

场景：
- load_shock
- high_rtt_burst
- churn_burst
- malicious_burst
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np
import pandas as pd
from collections import defaultdict
from copy import deepcopy

from gov_sim.env.gov_env import BlockchainGovEnv
from gov_sim.env.action_codec import GovernanceAction
from gov_sim.env.action_mask import is_action_legal
from gov_sim.utils.io import read_yaml


def check_high_template_legality(env: BlockchainGovEnv, m: int, theta: float, 
                                  b_values: list[int], tau_values: list[int]) -> bool:
    """检查给定(m, theta)模板是否至少存在一个合法的(b, tau)组合"""
    if env.current_snapshot is None or env.current_scenario is None:
        return False
    
    for b in b_values:
        for tau in tau_values:
            action = GovernanceAction(m=int(m), b=int(b), tau=int(tau), theta=float(theta))
            if is_action_legal(
                action=action,
                prev_action=env.prev_action,
                trust_scores=env.current_snapshot.final_scores,
                uptime=env.current_scenario.uptime,
                online=env.current_scenario.online,
                u_min=float(env.env_cfg["u_min"]),
                delta_m_max=int(env.env_cfg["delta_m_max"]),
                delta_b_max=int(env.env_cfg["delta_b_max"]),
                delta_tau_max=int(env.env_cfg["delta_tau_max"]),
                delta_theta_max=float(env.env_cfg["delta_theta_max"]),
                h_min=float(env.env_cfg["h_min"]),
                unsafe_guard_slack=float(env.env_cfg.get("unsafe_guard_slack", 1.0)),
            ):
                return True
    return False


def run_legality_audit(config_path: str, scenario_name: str, episodes: int = 10, seed: int = 42):
    """运行legality审计"""
    config = read_yaml(config_path)
    
    # 设置场景 - 使用training_mix配置
    config["scenario"]["training_mix"]["enabled"] = True
    config["scenario"]["training_mix"]["enabled_in_train"] = True
    # 只保留当前场景
    profile = config["scenario"]["training_mix"]["profiles"][scenario_name].copy()
    profile["weight"] = 1.0
    config["scenario"]["training_mix"]["profiles"] = {scenario_name: profile}
    config["seed"] = seed
    
    env = BlockchainGovEnv(config)
    
    # 要统计的高层模板
    m_values = [5, 7, 9]
    theta_values = [0.45, 0.50, 0.55, 0.60]
    
    # 从codec获取b和tau的值
    b_values = [32, 64, 96, 128, 160]
    tau_values = [5, 10, 15, 20]
    
    # 统计数据
    legality_counts = defaultdict(int)
    total_steps = 0
    
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        truncated = False
        
        while not (done or truncated):
            # 统计当前步各模板的legality
            for m in m_values:
                for theta in theta_values:
                    key = f"m={m},theta={theta:.2f}"
                    if check_high_template_legality(env, m, theta, b_values, tau_values):
                        legality_counts[key] += 1
            
            total_steps += 1
            
            # 执行一个合法动作继续
            mask = env.current_mask
            if mask.sum() > 0:
                legal_actions = np.where(mask > 0)[0]
                action = int(np.random.choice(legal_actions))
            else:
                action = 0
            
            obs, reward, done, truncated, info = env.step(action)
    
    # 计算legality频率
    results = []
    for m in m_values:
        for theta in theta_values:
            key = f"m={m},theta={theta:.2f}"
            count = legality_counts[key]
            ratio = count / max(total_steps, 1)
            results.append({
                "m": m,
                "theta": theta,
                "legality_count": count,
                "total_checked_count": total_steps,
                "legality_ratio": ratio
            })
    
    return pd.DataFrame(results), total_steps


def generate_report(output_dir: Path, scenario_dfs: dict[str, pd.DataFrame], overall_df: pd.DataFrame):
    """生成 markdown 报告"""
    
    report_lines = [
        "# 高层动作 Legality 频率核验报告",
        "",
        "## 1. 核验目标",
        "",
        "统计四个 hard 场景下所有高层候选组合的 legality ratio：",
        "- m ∈ {5, 7, 9}",
        "- theta ∈ {0.45, 0.50, 0.55, 0.60}",
        "- 共 12 个高层组合",
        "",
        "## 2. 场景结果",
        ""
    ]
    
    scenarios = ["load_shock", "high_rtt_burst", "churn_burst", "malicious_burst"]
    
    for scenario in scenarios:
        df = scenario_dfs[scenario]
        report_lines.append(f"### 2.{scenarios.index(scenario)+1} {scenario}")
        report_lines.append("")
        report_lines.append("| m | theta | legality_count | total_checked | legality_ratio |")
        report_lines.append("|---|-------|----------------|---------------|----------------|")
        
        for _, row in df.iterrows():
            report_lines.append(
                f"| {row['m']} | {row['theta']:.2f} | {row['legality_count']} | "
                f"{row['total_checked_count']} | {row['legality_ratio']:.4f} |"
            )
        report_lines.append("")
    
    # 总体统计
    report_lines.extend([
        "## 3. 总体统计",
        "",
        "### 3.1 各组合在四个场景的平均 legality_ratio",
        "",
        "| m | theta | mean_ratio | std_ratio | min_ratio | max_ratio |",
        "|---|-------|------------|-----------|-----------|-----------|"
    ])
    
    for _, row in overall_df.iterrows():
        report_lines.append(
            f"| {row['m']} | {row['theta']:.2f} | {row['mean_ratio']:.4f} | "
            f"{row['std_ratio']:.4f} | {row['min_ratio']:.4f} | {row['max_ratio']:.4f} |"
        )
    
    report_lines.append("")
    
    # m=5 分析
    m5_data = overall_df[overall_df['m'] == 5]
    m7_data = overall_df[overall_df['m'] == 7]
    m9_data = overall_df[overall_df['m'] == 9]
    
    m5_mean = m5_data['mean_ratio'].mean()
    m7_mean = m7_data['mean_ratio'].mean()
    m9_mean = m9_data['mean_ratio'].mean()
    
    report_lines.extend([
        "## 4. 关键发现：m=5 分析",
        "",
        "### 4.1 各 m 值的平均 legality_ratio",
        "",
        f"- m=5: {m5_mean:.4f}",
        f"- m=7: {m7_mean:.4f}",
        f"- m=9: {m9_mean:.4f}",
        "",
        "### 4.2 m=5 在四个场景下的表现",
        ""
    ])
    
    for scenario in scenarios:
        df = scenario_dfs[scenario]
        m5_scenario = df[df['m'] == 5]
        avg_ratio = m5_scenario['legality_ratio'].mean()
        report_lines.append(f"- {scenario}: {avg_ratio:.4f}")
    
    report_lines.append("")
    
    # 判断
    diff_m5_m7 = abs(m5_mean - m7_mean)
    diff_m5_m9 = abs(m5_mean - m9_mean)
    
    report_lines.extend([
        "### 4.3 m=5 是否显著低于 m=7/9",
        "",
        f"- m=5 vs m=7 差异: {diff_m5_m7:.4f}",
        f"- m=5 vs m=9 差异: {diff_m5_m9:.4f}",
        ""
    ])
    
    # 结论判断
    if m5_mean < m7_mean - 0.05 and m5_mean < m9_mean - 0.05:
        conclusion = "不支持"
        reason = f"m=5 的 legality_ratio ({m5_mean:.4f}) 显著低于 m=7 ({m7_mean:.4f}) 和 m=9 ({m9_mean:.4f})，差异超过 5%"
    elif abs(m5_mean - m7_mean) < 0.02 and abs(m5_mean - m9_mean) < 0.02:
        conclusion = "支持"
        reason = f"m=5 的 legality_ratio ({m5_mean:.4f}) 与 m=7 ({m7_mean:.4f}) 和 m=9 ({m9_mean:.4f}) 接近，差异小于 2%"
    else:
        conclusion = "存疑"
        reason = f"m=5 的 legality_ratio ({m5_mean:.4f}) 与 m=7/9 存在差异，但不够显著"
    
    report_lines.extend([
        "## 5. 最终判断",
        "",
        f"**当前 'm∈{{7,9}}+backstop(5)' 设计的数据支撑：{conclusion}**",
        "",
        f"**理由：** {reason}",
        "",
        "### 5.1 设计合理性分析",
        ""
    ])
    
    if conclusion == "支持":
        report_lines.extend([
            "- m=5 作为 backstop 是合理的，因为其 legality 与 m=7/9 相当",
            "- 在极端情况下，m=5 可以作为可靠的后备选项",
            "- 当前设计有充分的数据支撑"
        ])
    elif conclusion == "不支持":
        report_lines.extend([
            "- m=5 的 legality 显著低于 m=7/9，作为 backstop 可能不够可靠",
            "- 在极端情况下，m=5 可能频繁不可用",
            "- 建议重新评估 backstop 策略或考虑其他 m 值"
        ])
    else:
        report_lines.extend([
            "- m=5 的 legality 与 m=7/9 存在差异，但不够显著",
            "- 需要更多数据或更长时间的评估来确认",
            "- 当前设计可以保留，但需要持续监控"
        ])
    
    report_lines.append("")
    
    # 写入文件
    report_path = output_dir / "report_legality_frequency.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    return conclusion


def main():
    # 使用绝对路径
    project_root = Path(__file__).parent.parent.parent.parent
    config_path = project_root / "configs/default.yaml"
    scenarios = ["load_shock", "high_rtt_burst", "churn_burst", "malicious_burst"]
    episodes = 12
    seed = 42
    
    output_dir = project_root / "audits/round2_runtime_checks/action_space_compression"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    scenario_dfs = {}
    all_results = []
    
    for scenario in scenarios:
        print(f"\n=== 审计场景: {scenario} ===")
        df, total_steps = run_legality_audit(config_path, scenario, episodes, seed)
        df["scenario"] = scenario
        scenario_dfs[scenario] = df
        all_results.append(df)
        
        # 保存单场景结果
        output_path = output_dir / f"legality_by_{scenario}.csv"
        df.to_csv(output_path, index=False)
        print(f"完成 {scenario}, 总步数: {total_steps}")
        print(df[["m", "theta", "legality_ratio"]].to_string(index=False))
    
    # 合并所有场景
    combined = pd.concat(all_results, ignore_index=True)
    
    # 计算总体统计
    overall = combined.groupby(["m", "theta"]).agg({
        "legality_ratio": ["mean", "std", "min", "max"]
    }).reset_index()
    overall.columns = ["m", "theta", "mean_ratio", "std_ratio", "min_ratio", "max_ratio"]
    overall.to_csv(output_dir / "overall_legality_summary.csv", index=False)
    
    print("\n=== 总体统计 ===")
    print(overall.to_string(index=False))
    
    # 生成报告
    conclusion = generate_report(output_dir, scenario_dfs, overall)
    
    print(f"\n=== 最终判断 ===")
    print(f"当前 'm∈{{7,9}}+backstop(5)' 设计的数据支撑：{conclusion}")
    print(f"\n结果已保存到: {output_dir}")
    print(f"- legality_by_load_shock.csv")
    print(f"- legality_by_high_rtt_burst.csv")
    print(f"- legality_by_churn_burst.csv")
    print(f"- legality_by_malicious_burst.csv")
    print(f"- overall_legality_summary.csv")
    print(f"- report_legality_frequency.md")


if __name__ == "__main__":
    main()
