"""核验1A：统计高层动作空间legality频率

目标：在hard/default配置下，统计m=5,7,9 × theta=0.45,0.50,0.55,0.60各组合的合法率
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


def check_high_template_legality(env: BlockchainGovEnv, m: int, theta: float) -> bool:
    """检查给定(m, theta)模板是否至少存在一个合法的(b, tau)组合"""
    if env.current_snapshot is None or env.current_scenario is None:
        return False
    
    for b in env.codec.b_values:
        for tau in env.codec.tau_values:
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
                    if check_high_template_legality(env, m, theta):
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
            freq = legality_counts[key] / max(total_steps, 1)
            results.append({
                "m": m,
                "theta": theta,
                "legal_count": legality_counts[key],
                "total_steps": total_steps,
                "legality_frequency": freq
            })
    
    return pd.DataFrame(results), total_steps


def main():
    # 使用绝对路径
    project_root = Path(__file__).parent.parent.parent.parent
    config_path = project_root / "configs/default.yaml"
    scenarios = ["load_shock", "high_rtt_burst", "churn_burst", "malicious_burst"]
    episodes = 12
    seed = 42
    
    output_dir = project_root / "audits/round2_runtime_checks/action_space_compression"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    for scenario in scenarios:
        print(f"\n=== 审计场景: {scenario} ===")
        df, total_steps = run_legality_audit(config_path, scenario, episodes, seed)
        df["scenario"] = scenario
        all_results.append(df)
        
        # 保存单场景结果
        df.to_csv(output_dir / f"legality_{scenario}.csv", index=False)
        print(f"完成 {scenario}, 总步数: {total_steps}")
        print(df[["m", "theta", "legality_frequency"]].to_string(index=False))
    
    # 合并所有场景
    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(output_dir / "legality_all_scenarios.csv", index=False)
    
    # 计算总体统计
    summary = combined.groupby(["m", "theta"]).agg({
        "legality_frequency": ["mean", "std", "min", "max"]
    }).reset_index()
    summary.columns = ["m", "theta", "mean_freq", "std_freq", "min_freq", "max_freq"]
    summary.to_csv(output_dir / "legality_summary.csv", index=False)
    
    print("\n=== 总体统计 ===")
    print(summary.to_string(index=False))
    
    print(f"\n结果已保存到: {output_dir}")


if __name__ == "__main__":
    main()
