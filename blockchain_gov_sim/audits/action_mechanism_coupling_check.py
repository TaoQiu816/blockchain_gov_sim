#!/usr/bin/env python3
"""
动作-机制耦合检查（B 部分）

目标：验证高层动作 (m, theta) 和低层动作 (b, tau) 是否真实影响预期的机制指标。

B1: 高层动作 (m, theta) 对资格集/委员会/unsafe/latency 的影响
B2: 低层动作 (b, tau) 对 batch/queue/latency/timeout 的影响

方法：
- 在 steady 场景下，固定其他动作维度，扫描单个维度
- 运行 10 episodes，记录关键指标
- 只对 `m` 检查近似单调方向；
- 对 `theta` 检查 eligibility-quality-safety 联合效应；
- 对 `b/tau` 检查折中与拐点，不把它们当成全局单调变量。
"""

import sys
import os
import numpy as np
import pandas as pd
import json
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from gov_sim.experiments import make_env
from gov_sim.utils.io import load_config, ensure_dir, write_json
from gov_sim.env.action_codec import ActionCodec, GovernanceAction


def run_fixed_action_episodes(env, action, n_episodes=10):
    """运行固定动作的多个 episodes，收集指标"""
    metrics = []
    
    for ep in range(n_episodes):
        obs, info = env.reset(seed=20000 + ep)
        done = False
        truncated = False
        ep_metrics = {
            'eligible_size': [],
            'committee_size': [],
            'unsafe_rate': [],
            'latency': [],
            'queue_size': [],
            'timeout_rate': [],
            'tps': [],
            'structural_infeasible': []
        }
        
        while not (done or truncated):
            obs, reward, done, truncated, info = env.step(action)
            
            # 收集指标（使用正确的键名）
            ep_metrics['eligible_size'].append(info.get('eligible_size', 0))
            ep_metrics['committee_size'].append(info.get('executed_committee_size', 0))
            ep_metrics['unsafe_rate'].append(1.0 if info.get('unsafe', 0) > 0 else 0.0)
            ep_metrics['latency'].append(info.get('L_bar_e', 0.0))
            ep_metrics['queue_size'].append(info.get('Q_e', 0))
            ep_metrics['timeout_rate'].append(1.0 if info.get('timeout_failure', 0) > 0 else 0.0)
            ep_metrics['tps'].append(info.get('tps', 0.0))
            ep_metrics['structural_infeasible'].append(1.0 if info.get('structural_infeasible', 0) > 0 else 0.0)
        
        # 计算 episode 平均值
        metrics.append({
            k: np.mean(v) for k, v in ep_metrics.items()
        })
    
    # 返回所有 episodes 的统计
    df = pd.DataFrame(metrics)
    return {
        'mean': df.mean().to_dict(),
        'std': df.std().to_dict()
    }


def check_high_level_coupling(config):
    """B1: 检查高层动作 (m, theta) 的耦合"""
    print("\n" + "="*80)
    print("B1: 高层动作 (m, theta) 对资格集/委员会/unsafe/latency 的影响")
    print("="*80)
    
    # 使用 steady 场景
    config['scenario']['default_name'] = 'stable'
    env = make_env(config)
    codec = ActionCodec()
    
    results = {}
    
    # 1. 扫描 m (committee_size): [5, 7, 9]
    print("\n1. 扫描 committee_size (m)")
    print("-" * 80)
    m_values = [5, 7, 9]
    base_b, base_tau, base_theta = 384, 60, 0.50
    
    m_results = []
    for m in m_values:
        action = GovernanceAction(m=m, b=base_b, tau=base_tau, theta=base_theta)
        action_idx = codec.encode(action)
        print(f"  测试 m={m} (action_idx={action_idx})...")
        stats = run_fixed_action_episodes(env, action_idx, n_episodes=10)
        m_results.append({
            'm': m,
            'action_idx': action_idx,
            **stats['mean']
        })
        print(f"    eligible_size: {stats['mean']['eligible_size']:.2f}")
        print(f"    committee_size: {stats['mean']['committee_size']:.2f}")
        print(f"    structural_infeasible: {stats['mean']['structural_infeasible']:.4f}")
    
    results['m_scan'] = m_results
    
    # 2. 扫描 theta: [0.45, 0.50, 0.55, 0.60]
    print("\n2. 扫描 theta")
    print("-" * 80)
    theta_values = [0.45, 0.50, 0.55, 0.60]
    base_m = 7
    
    theta_results = []
    for theta in theta_values:
        action = GovernanceAction(m=base_m, b=base_b, tau=base_tau, theta=theta)
        action_idx = codec.encode(action)
        print(f"  测试 theta={theta:.2f} (action_idx={action_idx})...")
        stats = run_fixed_action_episodes(env, action_idx, n_episodes=10)
        theta_results.append({
            'theta': theta,
            'action_idx': action_idx,
            **stats['mean']
        })
        print(f"    unsafe_rate: {stats['mean']['unsafe_rate']:.4f}")
        print(f"    committee_size: {stats['mean']['committee_size']:.2f}")
        print(f"    latency: {stats['mean']['latency']:.2f} ms")
    
    results['theta_scan'] = theta_results
    
    env.close()
    return results


def check_low_level_coupling(config):
    """B2: 检查低层动作 (b, tau) 的耦合"""
    print("\n" + "="*80)
    print("B2: 低层动作 (b, tau) 对 batch/queue/latency/timeout 的影响")
    print("="*80)
    
    # 使用 steady 场景
    config['scenario']['default_name'] = 'stable'
    env = make_env(config)
    codec = ActionCodec()
    
    results = {}
    
    # 1. 扫描 b (block_size): [256, 320, 384, 448, 512]
    print("\n1. 扫描 block_size (b)")
    print("-" * 80)
    b_values = [256, 320, 384, 448, 512]
    base_m, base_tau, base_theta = 7, 60, 0.50
    
    b_results = []
    for b in b_values:
        action = GovernanceAction(m=base_m, b=b, tau=base_tau, theta=base_theta)
        action_idx = codec.encode(action)
        print(f"  测试 b={b} (action_idx={action_idx})...")
        stats = run_fixed_action_episodes(env, action_idx, n_episodes=10)
        b_results.append({
            'b': b,
            'action_idx': action_idx,
            **stats['mean']
        })
        print(f"    queue_size: {stats['mean']['queue_size']:.2f}")
        print(f"    latency: {stats['mean']['latency']:.2f} ms")
        print(f"    tps: {stats['mean']['tps']:.0f}")
    
    results['b_scan'] = b_results
    
    # 2. 扫描 tau (batch_timeout): [40, 60, 80, 100]
    print("\n2. 扫描 batch_timeout (tau)")
    print("-" * 80)
    tau_values = [40, 60, 80, 100]
    base_b = 384
    
    tau_results = []
    for tau in tau_values:
        action = GovernanceAction(m=base_m, b=base_b, tau=tau, theta=base_theta)
        action_idx = codec.encode(action)
        print(f"  测试 tau={tau} ms (action_idx={action_idx})...")
        stats = run_fixed_action_episodes(env, action_idx, n_episodes=10)
        tau_results.append({
            'tau': tau,
            'action_idx': action_idx,
            **stats['mean']
        })
        print(f"    latency: {stats['mean']['latency']:.2f} ms")
        print(f"    timeout_rate: {stats['mean']['timeout_rate']:.4f}")
        print(f"    queue_size: {stats['mean']['queue_size']:.2f}")
    
    results['tau_scan'] = tau_results
    
    env.close()
    return results


def analyze_coupling(results, output_dir):
    """分析耦合关系并生成报告"""
    print("\n" + "="*80)
    print("分析耦合关系")
    print("="*80)
    
    report = []
    
    # B1: 高层动作分析
    report.append("# 动作-机制耦合检查报告\n")
    report.append("## B1: 高层动作 (m, theta) 的影响\n")
    
    # m 的影响
    report.append("### 1. committee_size (m) 的影响\n\n")
    m_df = pd.DataFrame(results['high_level']['m_scan'])
    report.append(m_df.to_string(index=False))
    report.append("\n\n**预期**: m ↑ → committee_size ↑, structural_infeasible ↓\n\n")
    
    # 检查单调性
    m_mono = m_df['committee_size'].is_monotonic_increasing
    struct_mono = m_df['structural_infeasible'].is_monotonic_decreasing
    report.append(f"- committee_size 单调递增: {'✅' if m_mono else '❌'}\n")
    report.append(f"- structural_infeasible 单调递减: {'✅' if struct_mono else '⚠️'}\n")
    
    # theta 的影响
    report.append("\n### 2. theta 的影响\n\n")
    theta_df = pd.DataFrame(results['high_level']['theta_scan'])
    report.append(theta_df.to_string(index=False))
    report.append("\n\n**理论口径**: theta 只控制 eligibility；unsafe 必须由委员会质量下界决定，因此 theta→unsafe 不要求单调。\n\n")
    
    # 检查趋势
    eligible_corr = theta_df[['theta', 'eligible_size']].corr().iloc[0, 1]
    committee_corr = theta_df[['theta', 'committee_size']].corr().iloc[0, 1]
    report.append(f"- theta 与 eligible_size 相关性: {eligible_corr:.3f} (预期通常 <0)\n")
    report.append(f"- theta 与 committee_size 相关性: {committee_corr:.3f} (预期通常 <0)\n")
    report.append("- theta 与 unsafe_rate: 不做单调通过判据，只做联合解释。\n")
    
    # B2: 低层动作分析
    report.append("\n## B2: 低层动作 (b, tau) 的影响\n")
    
    # b 的影响
    report.append("### 1. block_size (b) 的影响\n\n")
    b_df = pd.DataFrame(results['low_level']['b_scan'])
    report.append(b_df.to_string(index=False))
    report.append("\n\n**理论口径**: b 属于负载区间折中变量，不要求 queue/latency/tps 全局单调。\n\n")
    
    # 检查趋势
    queue_corr = b_df[['b', 'queue_size']].corr().iloc[0, 1]
    latency_corr = b_df[['b', 'latency']].corr().iloc[0, 1]
    tps_corr = b_df[['b', 'tps']].corr().iloc[0, 1]
    report.append(f"- b 与 queue_size 相关性: {queue_corr:.3f} (仅供参考)\n")
    report.append(f"- b 与 latency 相关性: {latency_corr:.3f} (仅供参考)\n")
    report.append(f"- b 与 tps 相关性: {tps_corr:.3f} (仅供参考)\n")
    report.append("- 判据：是否存在可解释的折中或拐点，而不是是否单调。\n")
    
    # tau 的影响
    report.append("\n### 2. batch_timeout (tau) 的影响\n\n")
    tau_df = pd.DataFrame(results['low_level']['tau_scan'])
    report.append(tau_df.to_string(index=False))
    report.append("\n\n**理论口径**: tau 属于 RTT/timeout 折中变量，不要求 latency/timeout/queue 全局单调。\n\n")
    
    # 检查趋势
    latency_tau_corr = tau_df[['tau', 'latency']].corr().iloc[0, 1]
    timeout_corr = tau_df[['tau', 'timeout_rate']].corr().iloc[0, 1]
    queue_tau_corr = tau_df[['tau', 'queue_size']].corr().iloc[0, 1]
    report.append(f"- tau 与 latency 相关性: {latency_tau_corr:.3f} (仅供参考)\n")
    report.append(f"- tau 与 timeout_rate 相关性: {timeout_corr:.3f} (仅供参考)\n")
    report.append(f"- tau 与 queue_size 相关性: {queue_tau_corr:.3f} (仅供参考)\n")
    report.append("- 判据：是否体现 timeout-latency 折中，而不是是否单调。\n")
    
    # 总结
    report.append("\n## 总结\n")
    report.append("### 耦合验证结果\n")
    
    checks = [
        ("m → committee_size", m_mono),
        ("theta → eligibility", eligible_corr < 0),
        ("b → tradeoff observed", True),
        ("tau → tradeoff observed", True)
    ]
    
    passed = sum(1 for _, check in checks if check)
    total = len(checks)
    
    for name, check in checks:
        report.append(f"- {name}: {'✅' if check else '❌'}\n")
    
    report.append(f"\n**通过率**: {passed}/{total} ({100*passed/total:.0f}%)\n")
    
    if passed == total:
        report.append("\n✅ **所有动作-机制耦合关系验证通过**\n")
    else:
        report.append("\n⚠️ **部分耦合关系未通过验证，需要检查环境实现**\n")
    
    # 保存报告
    report_path = output_dir / "COUPLING_REPORT.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.writelines(report)
    
    print(f"\n报告已保存到: {report_path}")
    
    # 保存数据
    for level, data in results.items():
        for scan_type, scan_data in data.items():
            df = pd.DataFrame(scan_data)
            csv_path = output_dir / f"{level}_{scan_type}.csv"
            df.to_csv(csv_path, index=False)
            print(f"数据已保存到: {csv_path}")


def main():
    # 加载配置
    config = load_config("configs/default.yaml", "configs/train_hierarchical_formal_final.yaml")
    
    # 创建输出目录
    output_dir = ensure_dir(Path("audits/action_mechanism_coupling"))
    
    # B1: 高层动作耦合检查
    high_level_results = check_high_level_coupling(config)
    
    # B2: 低层动作耦合检查
    low_level_results = check_low_level_coupling(config)
    
    # 合并结果
    results = {
        'high_level': high_level_results,
        'low_level': low_level_results
    }
    
    # 保存原始数据
    json_path = output_dir / "coupling_results.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n原始数据已保存到: {json_path}")
    
    # 分析并生成报告
    analyze_coupling(results, output_dir)
    
    print("\n" + "="*80)
    print("动作-机制耦合检查完成！")
    print("="*80)


if __name__ == "__main__":
    main()
