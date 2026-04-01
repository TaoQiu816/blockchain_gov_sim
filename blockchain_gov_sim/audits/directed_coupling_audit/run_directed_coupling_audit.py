#!/usr/bin/env python3
"""
定向耦合审计脚本 - 在每个动作的主导场景中验证折中关系

审计内容:
1. b 扫描 —— 在 load_shock 场景 (吞吐/延迟权衡)
2. tau 扫描 —— 在 high_rtt_burst 场景 (超时/延迟权衡)
3. theta 扫描 —— 在 malicious_burst 场景 (安全性/活性权衡)
4. m 扫描 —— 在 churn_burst 场景 (委员会稳定性/结构可行性权衡)

约束:
- 不进入正式训练
- 不直接大改 chain_model
- 不简化动作空间
- 不弱化 hard 场景
- 必须基于真实运行证据
"""

import sys
import os
import copy
import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from gov_sim.experiments import make_env
from gov_sim.utils.io import load_config, ensure_dir
from gov_sim.env.action_codec import ActionCodec, GovernanceAction


def build_scene_config(base_config: dict, scene: str) -> dict:
    """构建特定场景配置，启用 training_mix 但只运行指定场景"""
    config = copy.deepcopy(base_config)
    
    # 获取场景 profile
    profile = config["scenario"]["training_mix"]["profiles"].get(scene, {})
    if not profile:
        raise ValueError(f"Unknown scene: {scene}")
    
    # 设置为只运行指定场景
    profile_copy = copy.deepcopy(profile)
    profile_copy["weight"] = 1.0
    
    config["scenario"]["training_mix"] = {
        "enabled": True,
        "enabled_in_train": False,
        "profiles": {scene: profile_copy},
    }
    
    return config


def run_fixed_action_episodes(
    env, 
    action: int, 
    n_episodes: int = 10,
    seed_offset: int = 0,
    verbose: bool = False
) -> dict[str, Any]:
    """运行固定动作的多个 episodes，收集详细指标
    
    返回:
        dict: 包含 mean, std 和 raw 数据的字典
    """
    metrics = []
    
    for ep in range(n_episodes):
        seed = 50000 + seed_offset * 1000 + ep
        obs, info = env.reset(seed=seed)
        done = False
        truncated = False
        
        ep_metrics = {
            # 吞吐指标
            'tps': [],
            'served_count': [],
            'queue_size': [],
            
            # 延迟指标
            'latency': [],
            'confirm_latency': [],
            'block_slack': [],
            
            # 超时和安全指标
            'timeout_failure': [],
            'unsafe': [],
            'structural_infeasible': [],
            
            # 委员会指标
            'eligible_size': [],
            'committee_size': [],
            'honest_ratio': [],
            'batch_fill_ratio': [],
            
            # 系统状态
            'arrivals': [],
            'churn': [],
            'rtt': [],
        }
        
        while not (done or truncated):
            obs, reward, done, truncated, info = env.step(action)
            
            # 收集每步指标
            ep_metrics['tps'].append(float(info.get('tps', 0.0)))
            ep_metrics['queue_size'].append(float(info.get('Q_e', 0)))
            ep_metrics['latency'].append(float(info.get('L_bar_e', 0.0)))
            ep_metrics['timeout_failure'].append(int(info.get('timeout_failure', 0)))
            ep_metrics['unsafe'].append(int(info.get('unsafe', 0)))
            ep_metrics['structural_infeasible'].append(int(info.get('structural_infeasible', 0)))
            ep_metrics['eligible_size'].append(int(info.get('eligible_size', 0)))
            ep_metrics['committee_size'].append(int(info.get('executed_committee_size', 0)))
            ep_metrics['honest_ratio'].append(float(info.get('h_e', 0.0)))
            ep_metrics['arrivals'].append(int(info.get('A_e', 0)))
            ep_metrics['churn'].append(float(info.get('chi_e', 0.0)))
            ep_metrics['rtt'].append(float(info.get('RTT_e', 0.0)))
            
            # 计算批次填充率 (served / capacity)
            b_val = info.get('b_e', 0)
            if b_val > 0:
                fill_ratio = min(1.0, info.get('A_e', 0) / b_val)
            else:
                fill_ratio = 0.0
            ep_metrics['batch_fill_ratio'].append(fill_ratio)
            
            # 区块松弛度
            ep_metrics['block_slack'].append(float(info.get('block_slack', 0.0)))
            
            # 确认延迟 (简化为平均延迟)
            ep_metrics['confirm_latency'].append(float(info.get('L_bar_e', 0.0)))
            ep_metrics['served_count'].append(float(info.get('A_e', 0) - info.get('Q_e', 0)))
        
        # 汇总 episode 指标
        episode_summary = {
            'mean_tps': np.mean(ep_metrics['tps']),
            'mean_queue_size': np.mean(ep_metrics['queue_size']),
            'mean_latency': np.mean(ep_metrics['latency']),
            'mean_confirm_latency': np.mean(ep_metrics['confirm_latency']),
            'mean_block_slack': np.mean(ep_metrics['block_slack']) if ep_metrics['block_slack'] else 0.0,
            'mean_batch_fill_ratio': np.mean(ep_metrics['batch_fill_ratio']),
            'mean_eligible_size': np.mean(ep_metrics['eligible_size']),
            'mean_committee_size': np.mean(ep_metrics['committee_size']),
            'mean_honest_ratio': np.mean(ep_metrics['honest_ratio']),
            'mean_arrivals': np.mean(ep_metrics['arrivals']),
            'mean_churn': np.mean(ep_metrics['churn']),
            'mean_rtt': np.mean(ep_metrics['rtt']),
            'total_served': sum(ep_metrics['served_count']),
            'timeout_rate': np.mean([1 if t > 0 else 0 for t in ep_metrics['timeout_failure']]),
            'unsafe_rate': np.mean([1 if u > 0 else 0 for u in ep_metrics['unsafe']]),
            'structural_infeasible_rate': np.mean([1 if s > 0 else 0 for s in ep_metrics['structural_infeasible']]),
        }
        metrics.append(episode_summary)
        
        if verbose:
            print(f"    Episode {ep+1}: TPS={episode_summary['mean_tps']:.1f}, "
                  f"latency={episode_summary['mean_latency']:.1f}ms, "
                  f"unsafe={episode_summary['unsafe_rate']:.2%}")
    
    # 计算统计量
    df = pd.DataFrame(metrics)
    return {
        'mean': df.mean().to_dict(),
        'std': df.std().to_dict(),
        'raw': metrics
    }


def run_multi_seed_experiment(
    config: dict,
    action: int,
    scene: str,
    n_seeds: int = 3,
    n_episodes_per_seed: int = 10,
    verbose: bool = True
) -> dict[str, Any]:
    """多 seed 实验运行"""
    scene_config = build_scene_config(config, scene)
    env = make_env(scene_config)
    
    all_results = []
    for seed_idx in range(n_seeds):
        if verbose:
            print(f"    Seed {seed_idx + 1}/{n_seeds}...")
        result = run_fixed_action_episodes(
            env, action, 
            n_episodes=n_episodes_per_seed,
            seed_offset=seed_idx,
            verbose=False
        )
        all_results.append(result['mean'])
    
    env.close()
    
    # 汇总多 seed 结果
    df = pd.DataFrame(all_results)
    return {
        'mean': df.mean().to_dict(),
        'std': df.std().to_dict(),
        'n_seeds': n_seeds,
        'n_episodes_per_seed': n_episodes_per_seed,
        'raw_seeds': all_results
    }


def audit_b_scan_load_shock(config: dict, verbose: bool = True) -> dict[str, Any]:
    """
    b 扫描 —— 在 load_shock 场景
    
    预期: b ↑ → queue_size ↓, TPS ↑, latency ↑ (吞吐/延迟权衡)
    """
    print("\n" + "="*80)
    print("审计 1: b 扫描 (load_shock 场景)")
    print("固定: m=7, theta=0.50, tau=60")
    print("扫描: b ∈ {256, 320, 384, 448, 512}")
    print("预期: b ↑ → queue_size ↓, TPS ↑, latency ↑")
    print("="*80)
    
    scene = 'load_shock'
    base_m, base_theta, base_tau = 7, 0.50, 60
    b_values = [256, 320, 384, 448, 512]
    
    codec = ActionCodec()
    results = {}
    
    for b in b_values:
        action = GovernanceAction(m=base_m, b=b, tau=base_tau, theta=base_theta)
        action_idx = codec.encode(action)
        
        if verbose:
            print(f"\n  测试 b={b} (action_idx={action_idx})...")
        
        result = run_multi_seed_experiment(
            config, action_idx, scene,
            n_seeds=3, n_episodes_per_seed=10,
            verbose=verbose
        )
        
        results[b] = result
        
        if verbose:
            mean = result['mean']
            std = result['std']
            print(f"    queue_size: {mean['mean_queue_size']:.2f} ± {std['mean_queue_size']:.2f}")
            print(f"    TPS: {mean['mean_tps']:.1f} ± {std['mean_tps']:.1f}")
            print(f"    latency: {mean['mean_latency']:.2f} ± {std['mean_latency']:.2f} ms")
            print(f"    block_slack: {mean['mean_block_slack']:.4f} ± {std['mean_block_slack']:.4f}")
            print(f"    batch_fill_ratio: {mean['mean_batch_fill_ratio']:.4f} ± {std['mean_batch_fill_ratio']:.4f}")
    
    return results


def audit_tau_scan_high_rtt_burst(config: dict, verbose: bool = True) -> dict[str, Any]:
    """
    tau 扫描 —— 在 high_rtt_burst 场景
    
    预期: tau ↑ → timeout_rate ↓, latency ↑ (超时/延迟权衡)
    """
    print("\n" + "="*80)
    print("审计 2: tau 扫描 (high_rtt_burst 场景)")
    print("固定: m=7, theta=0.50, b=384")
    print("扫描: tau ∈ {40, 60, 80, 100}")
    print("预期: tau ↑ → timeout_rate ↓, latency ↑")
    print("="*80)
    
    scene = 'high_rtt_burst'
    base_m, base_theta, base_b = 7, 0.50, 384
    tau_values = [40, 60, 80, 100]
    
    codec = ActionCodec()
    results = {}
    
    for tau in tau_values:
        action = GovernanceAction(m=base_m, b=base_b, tau=tau, theta=base_theta)
        action_idx = codec.encode(action)
        
        if verbose:
            print(f"\n  测试 tau={tau} ms (action_idx={action_idx})...")
        
        result = run_multi_seed_experiment(
            config, action_idx, scene,
            n_seeds=3, n_episodes_per_seed=10,
            verbose=verbose
        )
        
        results[tau] = result
        
        if verbose:
            mean = result['mean']
            std = result['std']
            print(f"    timeout_rate: {mean['timeout_rate']:.4f} ± {std['timeout_rate']:.4f}")
            print(f"    mean_latency: {mean['mean_latency']:.2f} ± {std['mean_latency']:.2f} ms")
            print(f"    confirm_latency: {mean['mean_confirm_latency']:.2f} ± {std['mean_confirm_latency']:.2f} ms")
            print(f"    queue_size: {mean['mean_queue_size']:.2f} ± {std['mean_queue_size']:.2f}")
            print(f"    total_served: {mean['total_served']:.0f} ± {std['total_served']:.0f}")
    
    return results


def audit_theta_scan_malicious_burst(config: dict, verbose: bool = True) -> dict[str, Any]:
    """
    theta 扫描 —— 在 malicious_burst 场景
    
    预期: theta ↑ → unsafe_rate ↓, committee_size ↓ (安全性/活性权衡)
    """
    print("\n" + "="*80)
    print("审计 3: theta 扫描 (malicious_burst 场景)")
    print("固定: m=7, b=384, tau=60")
    print("扫描: theta ∈ {0.45, 0.50, 0.55, 0.60}")
    print("预期: theta ↑ → unsafe_rate ↓, committee_size ↓")
    print("="*80)
    
    scene = 'malicious_burst'
    base_m, base_b, base_tau = 7, 384, 60
    theta_values = [0.45, 0.50, 0.55, 0.60]
    
    codec = ActionCodec()
    results = {}
    
    for theta in theta_values:
        action = GovernanceAction(m=base_m, b=base_b, tau=base_tau, theta=theta)
        action_idx = codec.encode(action)
        
        if verbose:
            print(f"\n  测试 theta={theta:.2f} (action_idx={action_idx})...")
        
        result = run_multi_seed_experiment(
            config, action_idx, scene,
            n_seeds=3, n_episodes_per_seed=10,
            verbose=verbose
        )
        
        results[theta] = result
        
        if verbose:
            mean = result['mean']
            std = result['std']
            print(f"    unsafe_rate: {mean['unsafe_rate']:.4f} ± {std['unsafe_rate']:.4f}")
            print(f"    honest_ratio: {mean['mean_honest_ratio']:.4f} ± {std['mean_honest_ratio']:.4f}")
            print(f"    committee_size: {mean['mean_committee_size']:.2f} ± {std['mean_committee_size']:.2f}")
            print(f"    eligible_size: {mean['mean_eligible_size']:.2f} ± {std['mean_eligible_size']:.2f}")
            print(f"    structural_infeasible_rate: {mean['structural_infeasible_rate']:.4f} ± {std['structural_infeasible_rate']:.4f}")
    
    return results


def audit_m_scan_churn_burst(config: dict, verbose: bool = True) -> dict[str, Any]:
    """
    m 扫描 —— 在 churn_burst 场景
    
    预期: m ↑ → committee_size ↑, structural_infeasible ↓ (委员会稳定性/结构可行性权衡)
    """
    print("\n" + "="*80)
    print("审计 4: m 扫描 (churn_burst 场景)")
    print("固定: theta=0.50, b=384, tau=60")
    print("扫描: m ∈ {5, 7, 9}")
    print("预期: m ↑ → committee_size ↑, structural_infeasible ↓")
    print("="*80)
    
    scene = 'churn_burst'
    base_theta, base_b, base_tau = 0.50, 384, 60
    m_values = [5, 7, 9]
    
    codec = ActionCodec()
    results = {}
    
    for m in m_values:
        action = GovernanceAction(m=m, b=base_b, tau=base_tau, theta=base_theta)
        action_idx = codec.encode(action)
        
        if verbose:
            print(f"\n  测试 m={m} (action_idx={action_idx})...")
        
        result = run_multi_seed_experiment(
            config, action_idx, scene,
            n_seeds=3, n_episodes_per_seed=10,
            verbose=verbose
        )
        
        results[m] = result
        
        if verbose:
            mean = result['mean']
            std = result['std']
            print(f"    committee_size: {mean['mean_committee_size']:.2f} ± {std['mean_committee_size']:.2f}")
            print(f"    structural_infeasible_rate: {mean['structural_infeasible_rate']:.4f} ± {std['structural_infeasible_rate']:.4f}")
            print(f"    mean_latency: {mean['mean_latency']:.2f} ± {std['mean_latency']:.2f} ms")
            print(f"    unsafe_rate: {mean['unsafe_rate']:.4f} ± {std['unsafe_rate']:.4f}")
            print(f"    eligible_size: {mean['mean_eligible_size']:.2f} ± {std['mean_eligible_size']:.2f}")
    
    return results


def analyze_correlation(results: dict, y_key: str, x_values: list) -> dict:
    """分析 x 和 y 之间的相关性"""
    y_means = [results[x]['mean'][y_key] for x in x_values]
    x_arr = np.array(x_values)
    y_arr = np.array(y_means)
    
    # 计算 Pearson 相关系数
    if len(x_values) > 2 and np.std(y_arr) > 0:
        corr = np.corrcoef(x_arr, y_arr)[0, 1]
    else:
        corr = 0.0
    
    # 判断单调性
    is_monotonic = all(y_means[i] <= y_means[i+1] for i in range(len(y_means)-1)) or \
                   all(y_means[i] >= y_means[i+1] for i in range(len(y_means)-1))
    
    return {
        'correlation': float(corr),
        'is_monotonic': is_monotonic,
        'x_values': x_values,
        'y_means': y_means
    }


def generate_judgment(results: dict[str, Any]) -> dict[str, Any]:
    """生成"是否允许改模型"的判据"""
    judgment = {
        'timestamp': datetime.now().isoformat(),
        'audits': {}
    }
    
    # 1. b 扫描判据
    b_results = results.get('b_scan_load_shock', {})
    if b_results:
        b_values = [256, 320, 384, 448, 512]
        
        # 检查 queue_size 随 b 增加 (相关性应 <0)
        queue_corr = analyze_correlation(b_results, 'mean_queue_size', b_values)
        # 检查 TPS 随 b 增加 (相关性应 >0)
        tps_corr = analyze_correlation(b_results, 'mean_tps', b_values)
        # 检查 latency 随 b 增加 (相关性应 >0)
        latency_corr = analyze_correlation(b_results, 'mean_latency', b_values)
        
        b_pass = (queue_corr['correlation'] < -0.5) and (tps_corr['correlation'] > 0.5)
        
        judgment['audits']['b_scan'] = {
            'queue_size_trend': queue_corr,
            'tps_trend': tps_corr,
            'latency_trend': latency_corr,
            'pass': b_pass,
            'reason': 'b增加应降低队列、提高吞吐' if b_pass else '未观察到b的预期效果，可能需要检查链模型',
            'allow_model_change': not b_pass  # 如果不通过，允许改模型
        }
    
    # 2. tau 扫描判据
    tau_results = results.get('tau_scan_high_rtt', {})
    if tau_results:
        tau_values = [40, 60, 80, 100]
        
        # 检查 timeout_rate 随 tau 减少 (相关性应 <0)
        timeout_corr = analyze_correlation(tau_results, 'timeout_rate', tau_values)
        # 检查 latency 随 tau 增加 (相关性应 >0)
        latency_corr = analyze_correlation(tau_results, 'mean_latency', tau_values)
        
        tau_pass = (timeout_corr['correlation'] < -0.5)
        
        judgment['audits']['tau_scan'] = {
            'timeout_rate_trend': timeout_corr,
            'latency_trend': latency_corr,
            'pass': tau_pass,
            'reason': 'tau增加应降低超时率' if tau_pass else '未观察到tau的预期效果，可能需要检查超时机制',
            'allow_model_change': not tau_pass
        }
    
    # 3. theta 扫描判据
    theta_results = results.get('theta_scan_malicious', {})
    if theta_results:
        theta_values = [0.45, 0.50, 0.55, 0.60]
        
        # 检查 unsafe_rate 随 theta 减少 (相关性应 <0)
        unsafe_corr = analyze_correlation(theta_results, 'unsafe_rate', theta_values)
        # 检查 committee_size 随 theta 减少 (相关性应 <0)
        committee_corr = analyze_correlation(theta_results, 'mean_committee_size', theta_values)
        
        theta_pass = (unsafe_corr['correlation'] < -0.5)
        
        judgment['audits']['theta_scan'] = {
            'unsafe_rate_trend': unsafe_corr,
            'committee_size_trend': committee_corr,
            'pass': theta_pass,
            'reason': 'theta增加应降低不安全率' if theta_pass else '未观察到theta的预期效果，可能需要检查安全机制',
            'allow_model_change': not theta_pass
        }
    
    # 4. m 扫描判据
    m_results = results.get('m_scan_churn', {})
    if m_results:
        m_values = [5, 7, 9]
        
        # 检查 committee_size 随 m 增加 (相关性应 >0)
        committee_corr = analyze_correlation(m_results, 'mean_committee_size', m_values)
        # 检查 structural_infeasible 随 m 减少 (相关性应 <0)
        infeasible_corr = analyze_correlation(m_results, 'structural_infeasible_rate', m_values)
        
        m_pass = (committee_corr['correlation'] > 0.8)  # m 应直接决定 committee_size
        
        judgment['audits']['m_scan'] = {
            'committee_size_trend': committee_corr,
            'structural_infeasible_trend': infeasible_corr,
            'pass': m_pass,
            'reason': 'm应直接决定委员会大小' if m_pass else 'm未正确影响委员会大小，需要检查',
            'allow_model_change': not m_pass
        }
    
    # 总体判据
    all_pass = all(a.get('pass', False) for a in judgment['audits'].values())
    any_allow_change = any(a.get('allow_model_change', False) for a in judgment['audits'].values())
    
    judgment['overall'] = {
        'all_pass': all_pass,
        'allow_any_model_change': any_allow_change,
        'conclusion': (
            '所有动作-机制耦合关系验证通过，不建议修改模型' 
            if all_pass else 
            '部分耦合关系未验证通过，建议检查对应机制或允许模型调整'
        )
    }
    
    return judgment


def export_results(results: dict[str, Any], output_dir: Path):
    """导出结果到 CSV 和 JSON"""
    
    # 导出各扫描结果到 CSV
    for scan_name, scan_key in [
        ('b_scan_load_shock', 'b_scan'),
        ('tau_scan_high_rtt', 'tau_scan'),
        ('theta_scan_malicious', 'theta_scan'),
        ('m_scan_churn', 'm_scan')
    ]:
        if scan_name not in results:
            continue
        
        scan_data = results[scan_name]
        rows = []
        for param_val, data in scan_data.items():
            row = {'param': param_val}
            row.update(data['mean'])
            row.update({f'{k}_std': v for k, v in data['std'].items()})
            rows.append(row)
        
        df = pd.DataFrame(rows)
        csv_path = output_dir / f"directed_coupling_{scan_key}.csv"
        df.to_csv(csv_path, index=False)
        print(f"导出: {csv_path}")
    
    # 导出完整结果到 JSON
    json_path = output_dir / "directed_coupling_full_results.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        # 转换为可序列化格式
        serializable = {}
        for scan_name, scan_data in results.items():
            if scan_name == 'judgment':
                continue
            serializable[scan_name] = {}
            for param_val, data in scan_data.items():
                serializable[scan_name][str(param_val)] = {
                    'mean': data['mean'],
                    'std': data['std'],
                    'n_seeds': data['n_seeds'],
                    'n_episodes_per_seed': data['n_episodes_per_seed']
                }
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"导出: {json_path}")


def generate_report(results: dict[str, Any], judgment: dict, output_dir: Path):
    """生成审计报告"""
    report = []
    
    report.append("# 定向耦合审计报告\n\n")
    report.append(f"生成时间: {judgment['timestamp']}\n\n")
    
    report.append("## 审计目标\n\n")
    report.append("验证每个动作在其主导场景中的折中关系:\n\n")
    report.append("1. **b 扫描** (load_shock): 验证区块大小对吞吐/延迟的影响\n")
    report.append("2. **tau 扫描** (high_rtt_burst): 验证批次超时对超时率/延迟的影响\n")
    report.append("3. **theta 扫描** (malicious_burst): 验证信任阈值对安全性/活性的影响\n")
    report.append("4. **m 扫描** (churn_burst): 验证委员会大小对稳定性/结构可行性的影响\n\n")
    
    # 1. b 扫描结果
    report.append("## 1. b 扫描 (load_shock 场景)\n\n")
    if 'b_scan_load_shock' in results:
        b_data = results['b_scan_load_shock']
        rows = []
        for b in [256, 320, 384, 448, 512]:
            if b in b_data:
                rows.append({
                    'b': b,
                    'queue_size': f"{b_data[b]['mean']['mean_queue_size']:.2f} ± {b_data[b]['std']['mean_queue_size']:.2f}",
                    'TPS': f"{b_data[b]['mean']['mean_tps']:.1f} ± {b_data[b]['std']['mean_tps']:.1f}",
                    'latency_ms': f"{b_data[b]['mean']['mean_latency']:.2f} ± {b_data[b]['std']['mean_latency']:.2f}",
                    'block_slack': f"{b_data[b]['mean']['mean_block_slack']:.4f}",
                    'batch_fill': f"{b_data[b]['mean']['mean_batch_fill_ratio']:.4f}",
                })
        df = pd.DataFrame(rows)
        report.append(df.to_markdown(index=False))
        report.append("\n\n")
        
        audit = judgment['audits'].get('b_scan', {})
        report.append(f"**判据结果**: {'✅ 通过' if audit.get('pass') else '❌ 未通过'}\n\n")
        report.append(f"- 预期: b ↑ → queue_size ↓, TPS ↑\n")
        report.append(f"- 分析: {audit.get('reason', 'N/A')}\n\n")
    
    # 2. tau 扫描结果
    report.append("## 2. tau 扫描 (high_rtt_burst 场景)\n\n")
    if 'tau_scan_high_rtt' in results:
        tau_data = results['tau_scan_high_rtt']
        rows = []
        for tau in [40, 60, 80, 100]:
            if tau in tau_data:
                rows.append({
                    'tau_ms': tau,
                    'timeout_rate': f"{tau_data[tau]['mean']['timeout_rate']:.4f} ± {tau_data[tau]['std']['timeout_rate']:.4f}",
                    'latency_ms': f"{tau_data[tau]['mean']['mean_latency']:.2f} ± {tau_data[tau]['std']['mean_latency']:.2f}",
                    'confirm_latency': f"{tau_data[tau]['mean']['mean_confirm_latency']:.2f}",
                    'queue_size': f"{tau_data[tau]['mean']['mean_queue_size']:.2f}",
                    'served': f"{tau_data[tau]['mean']['total_served']:.0f}",
                })
        df = pd.DataFrame(rows)
        report.append(df.to_markdown(index=False))
        report.append("\n\n")
        
        audit = judgment['audits'].get('tau_scan', {})
        report.append(f"**判据结果**: {'✅ 通过' if audit.get('pass') else '❌ 未通过'}\n\n")
        report.append(f"- 预期: tau ↑ → timeout_rate ↓\n")
        report.append(f"- 分析: {audit.get('reason', 'N/A')}\n\n")
    
    # 3. theta 扫描结果
    report.append("## 3. theta 扫描 (malicious_burst 场景)\n\n")
    if 'theta_scan_malicious' in results:
        theta_data = results['theta_scan_malicious']
        rows = []
        for theta in [0.45, 0.50, 0.55, 0.60]:
            if theta in theta_data:
                rows.append({
                    'theta': theta,
                    'unsafe_rate': f"{theta_data[theta]['mean']['unsafe_rate']:.4f} ± {theta_data[theta]['std']['unsafe_rate']:.4f}",
                    'honest_ratio': f"{theta_data[theta]['mean']['mean_honest_ratio']:.4f}",
                    'committee_size': f"{theta_data[theta]['mean']['mean_committee_size']:.2f} ± {theta_data[theta]['std']['mean_committee_size']:.2f}",
                    'eligible_size': f"{theta_data[theta]['mean']['mean_eligible_size']:.2f}",
                    'infeasible_rate': f"{theta_data[theta]['mean']['structural_infeasible_rate']:.4f}",
                })
        df = pd.DataFrame(rows)
        report.append(df.to_markdown(index=False))
        report.append("\n\n")
        
        audit = judgment['audits'].get('theta_scan', {})
        report.append(f"**判据结果**: {'✅ 通过' if audit.get('pass') else '❌ 未通过'}\n\n")
        report.append(f"- 预期: theta ↑ → unsafe_rate ↓, committee_size ↓\n")
        report.append(f"- 分析: {audit.get('reason', 'N/A')}\n\n")
    
    # 4. m 扫描结果
    report.append("## 4. m 扫描 (churn_burst 场景)\n\n")
    if 'm_scan_churn' in results:
        m_data = results['m_scan_churn']
        rows = []
        for m in [5, 7, 9]:
            if m in m_data:
                rows.append({
                    'm': m,
                    'committee_size': f"{m_data[m]['mean']['mean_committee_size']:.2f} ± {m_data[m]['std']['mean_committee_size']:.2f}",
                    'infeasible_rate': f"{m_data[m]['mean']['structural_infeasible_rate']:.4f} ± {m_data[m]['std']['structural_infeasible_rate']:.4f}",
                    'latency_ms': f"{m_data[m]['mean']['mean_latency']:.2f}",
                    'unsafe_rate': f"{m_data[m]['mean']['unsafe_rate']:.4f}",
                    'eligible_size': f"{m_data[m]['mean']['mean_eligible_size']:.2f}",
                })
        df = pd.DataFrame(rows)
        report.append(df.to_markdown(index=False))
        report.append("\n\n")
        
        audit = judgment['audits'].get('m_scan', {})
        report.append(f"**判据结果**: {'✅ 通过' if audit.get('pass') else '❌ 未通过'}\n\n")
        report.append(f"- 预期: m ↑ → committee_size ↑\n")
        report.append(f"- 分析: {audit.get('reason', 'N/A')}\n\n")
    
    # 总体判据
    report.append("## 总体判据\n\n")
    report.append(f"**所有审计通过**: {'✅ 是' if judgment['overall']['all_pass'] else '❌ 否'}\n\n")
    report.append(f"**允许模型修改**: {'⚠️ 是' if judgment['overall']['allow_any_model_change'] else '否'}\n\n")
    report.append(f"**结论**: {judgment['overall']['conclusion']}\n\n")
    
    report.append("---\n\n")
    report.append("*本报告基于真实运行证据生成，用于指导是否允许修改模型*\n")
    
    report_path = output_dir / "DIRECTED_COUPLING_REPORT.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.writelines(report)
    print(f"\n报告已保存到: {report_path}")


def main():
    print("="*80)
    print("定向耦合审计")
    print("按机制对应场景验证动作-机制耦合关系")
    print("="*80)
    
    # 加载配置
    config = load_config("configs/default.yaml", "configs/train_hierarchical_formal_final.yaml")
    
    # 创建输出目录
    output_dir = ensure_dir(Path("audits/directed_coupling_audit"))
    
    # 存储所有结果
    all_results = {}
    
    try:
        # 1. b 扫描 - load_shock 场景
        print("\n开始审计 1/4...")
        all_results['b_scan_load_shock'] = audit_b_scan_load_shock(config)
    except Exception as e:
        print(f"  ❌ b 扫描失败: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        # 2. tau 扫描 - high_rtt_burst 场景
        print("\n开始审计 2/4...")
        all_results['tau_scan_high_rtt'] = audit_tau_scan_high_rtt_burst(config)
    except Exception as e:
        print(f"  ❌ tau 扫描失败: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        # 3. theta 扫描 - malicious_burst 场景
        print("\n开始审计 3/4...")
        all_results['theta_scan_malicious'] = audit_theta_scan_malicious_burst(config)
    except Exception as e:
        print(f"  ❌ theta 扫描失败: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        # 4. m 扫描 - churn_burst 场景
        print("\n开始审计 4/4...")
        all_results['m_scan_churn'] = audit_m_scan_churn_burst(config)
    except Exception as e:
        print(f"  ❌ m 扫描失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 生成判据
    print("\n生成判据...")
    judgment = generate_judgment(all_results)
    all_results['judgment'] = judgment
    
    # 导出结果
    print("\n导出结果...")
    export_results(all_results, output_dir)
    
    # 保存判据
    judgment_path = output_dir / "judgment.json"
    with open(judgment_path, 'w', encoding='utf-8') as f:
        json.dump(judgment, f, indent=2, ensure_ascii=False)
    print(f"判据已保存到: {judgment_path}")
    
    # 生成报告
    generate_report(all_results, judgment, output_dir)
    
    print("\n" + "="*80)
    print("定向耦合审计完成！")
    print("="*80)
    
    return all_results


if __name__ == "__main__":
    results = main()
