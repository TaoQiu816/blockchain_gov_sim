#!/usr/bin/env python3
"""
补充场景验证脚本 - 在恶意burst和高RTT burst场景下验证耦合关系

验证：
1. theta → unsafe_rate (在 malicious_burst 场景)
2. tau → timeout_rate (在 high_rtt_burst 场景)
"""

import sys
import os
import numpy as np
import pandas as pd
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from gov_sim.experiments import make_env
from gov_sim.utils.io import load_config, ensure_dir, write_json
from gov_sim.env.action_codec import ActionCodec, GovernanceAction


def run_fixed_action_episodes(env, action, n_episodes=10):
    """运行固定动作的多个 episodes，收集指标"""
    metrics = []
    
    for ep in range(n_episodes):
        obs, info = env.reset(seed=30000 + ep)
        done = False
        truncated = False
        ep_metrics = {
            'unsafe': [],
            'timeout_failure': [],
            'eligible_size': [],
            'committee_size': [],
            'latency': [],
            'queue_size': [],
            'honest_ratio': [],
            'tps': [],
        }
        
        while not (done or truncated):
            obs, reward, done, truncated, info = env.step(action)
            
            ep_metrics['unsafe'].append(info.get('unsafe', 0))
            ep_metrics['timeout_failure'].append(info.get('timeout_failure', 0))
            ep_metrics['eligible_size'].append(info.get('eligible_size', 0))
            ep_metrics['committee_size'].append(info.get('executed_committee_size', 0))
            ep_metrics['latency'].append(info.get('L_bar_e', 0.0))
            ep_metrics['queue_size'].append(info.get('Q_e', 0))
            ep_metrics['honest_ratio'].append(info.get('h_e', 0.0))
            ep_metrics['tps'].append(info.get('tps', 0.0))
        
        metrics.append({
            'unsafe_rate': np.mean([1 if u > 0 else 0 for u in ep_metrics['unsafe']]),
            'timeout_rate': np.mean([1 if t > 0 else 0 for t in ep_metrics['timeout_failure']]),
            'eligible_size': np.mean(ep_metrics['eligible_size']),
            'committee_size': np.mean(ep_metrics['committee_size']),
            'latency': np.mean(ep_metrics['latency']),
            'queue_size': np.mean(ep_metrics['queue_size']),
            'honest_ratio': np.mean(ep_metrics['honest_ratio']),
            'tps': np.mean(ep_metrics['tps']),
        })
    
    df = pd.DataFrame(metrics)
    return df.mean().to_dict()


def build_scene_config(base_config, scene):
    """构建场景配置"""
    config = base_config.copy()
    profile = config["scenario"]["training_mix"]["profiles"][scene].copy()
    profile["weight"] = 1.0
    config["scenario"]["training_mix"] = {
        "enabled": True,
        "enabled_in_train": False,
        "profiles": {scene: profile},
    }
    return config


def check_theta_unsafe_coupling(config):
    """验证 theta → unsafe_rate 在 malicious_burst 场景"""
    print("\n" + "="*80)
    print("补充验证: theta → unsafe_rate (malicious_burst 场景)")
    print("="*80)
    
    scene_config = build_scene_config(config, 'malicious_burst')
    env = make_env(scene_config)
    codec = ActionCodec()
    
    theta_values = [0.45, 0.50, 0.55, 0.60]
    base_m, base_b, base_tau = 7, 384, 60
    
    results = []
    for theta in theta_values:
        action = GovernanceAction(m=base_m, b=base_b, tau=base_tau, theta=theta)
        action_idx = codec.encode(action)
        print(f"  测试 theta={theta:.2f} (action_idx={action_idx})...")
        stats = run_fixed_action_episodes(env, action_idx, n_episodes=10)
        results.append({
            'theta': theta,
            'unsafe_rate': stats['unsafe_rate'],
            'timeout_rate': stats['timeout_rate'],
            'committee_size': stats['committee_size'],
            'honest_ratio': stats['honest_ratio'],
            'latency': stats['latency'],
        })
        print(f"    unsafe_rate: {stats['unsafe_rate']:.4f}")
        print(f"    honest_ratio: {stats['honest_ratio']:.4f}")
        print(f"    committee_size: {stats['committee_size']:.2f}")
    
    env.close()
    return results


def check_tau_timeout_coupling(config):
    """验证 tau → timeout_rate 在 high_rtt_burst 场景"""
    print("\n" + "="*80)
    print("补充验证: tau → timeout_rate (high_rtt_burst 场景)")
    print("="*80)
    
    scene_config = build_scene_config(config, 'high_rtt_burst')
    env = make_env(scene_config)
    codec = ActionCodec()
    
    tau_values = [40, 60, 80, 100]
    base_m, base_b, base_theta = 7, 384, 0.50
    
    results = []
    for tau in tau_values:
        action = GovernanceAction(m=base_m, b=base_b, tau=tau, theta=base_theta)
        action_idx = codec.encode(action)
        print(f"  测试 tau={tau} ms (action_idx={action_idx})...")
        stats = run_fixed_action_episodes(env, action_idx, n_episodes=10)
        results.append({
            'tau': tau,
            'timeout_rate': stats['timeout_rate'],
            'unsafe_rate': stats['unsafe_rate'],
            'latency': stats['latency'],
            'committee_size': stats['committee_size'],
            'tps': stats['tps'],
        })
        print(f"    timeout_rate: {stats['timeout_rate']:.4f}")
        print(f"    latency: {stats['latency']:.2f} ms")
        print(f"    tps: {stats['tps']:.0f}")
    
    env.close()
    return results


def main():
    config = load_config("configs/default.yaml", "configs/train_hierarchical_formal_final.yaml")
    
    output_dir = ensure_dir(Path("audits/action_mechanism_coupling"))
    
    # 1. theta → unsafe_rate 在 malicious_burst
    theta_results = check_theta_unsafe_coupling(config)
    
    # 2. tau → timeout_rate 在 high_rtt_burst
    tau_results = check_tau_timeout_coupling(config)
    
    # 生成报告
    report = []
    report.append("# 补充场景耦合验证报告\n")
    
    report.append("## 1. theta → unsafe_rate (malicious_burst 场景)\n\n")
    theta_df = pd.DataFrame(theta_results)
    report.append(theta_df.to_string(index=False))
    report.append("\n\n")
    
    # 检查相关性
    if theta_df['unsafe_rate'].std() > 0:
        unsafe_corr = theta_df[['theta', 'unsafe_rate']].corr().iloc[0, 1]
        report.append(f"theta 与 unsafe_rate 相关性: {unsafe_corr:.3f} (预期 <0)\n")
        report.append(f"结论: {'✅ theta↑ 时 unsafe_rate↓' if unsafe_corr < 0 else '❌ 未观察到预期趋势'}\n")
    else:
        report.append("⚠️ 所有场景 unsafe_rate 相同，无法验证\n")
    
    report.append("\n## 2. tau → timeout_rate (high_rtt_burst 场景)\n\n")
    tau_df = pd.DataFrame(tau_results)
    report.append(tau_df.to_string(index=False))
    report.append("\n\n")
    
    # 检查相关性
    if tau_df['timeout_rate'].std() > 0:
        timeout_corr = tau_df[['tau', 'timeout_rate']].corr().iloc[0, 1]
        latency_corr = tau_df[['tau', 'latency']].corr().iloc[0, 1]
        report.append(f"tau 与 timeout_rate 相关性: {timeout_corr:.3f} (预期 <0)\n")
        report.append(f"tau 与 latency 相关性: {latency_corr:.3f} (预期 >0)\n")
        report.append(f"结论:\n")
        report.append(f"- timeout_rate: {'✅ tau↑ 时 timeout_rate↓' if timeout_corr < 0 else '❌ 未观察到预期趋势'}\n")
        report.append(f"- latency: {'✅ tau↑ 时 latency↑' if latency_corr > 0 else '⚠️ 未观察到预期趋势'}\n")
    else:
        report.append("⚠️ 所有场景 timeout_rate 相同，无法验证\n")
    
    # 保存报告
    report_path = output_dir / "SUPPLEMENTARY_COUPLING_REPORT.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.writelines(report)
    
    print(f"\n报告已保存到: {report_path}")
    print("\n" + "="*80)
    print("补充场景耦合验证完成！")
    print("="*80)


if __name__ == "__main__":
    main()
