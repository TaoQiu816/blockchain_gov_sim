"""场景机制检查脚本。

验证每个场景是否真实产生预期的机制效果：
- load_shock: 提高 arrival / queue pressure / TPS-pressure tradeoff
- high_rtt_burst: 提高 RTT / timeout risk / latency
- churn_burst: 提高资格不足与 structural infeasible
- malicious_burst: 提高 unsafe risk，使更高 theta / 更稳健 m 的收益可见
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from tqdm import tqdm

from gov_sim.experiments import make_env
from gov_sim.utils.io import load_config, ensure_dir, write_json


def run_scenario_check(scene: str, n_episodes: int = 20) -> dict:
    """运行单个场景并收集统计数据。"""
    
    # 加载配置
    base_config = load_config("configs/default.yaml", "configs/train_hierarchical_formal_final.yaml")
    
    # 构建场景配置
    config = base_config.copy()
    profile = config["scenario"]["training_mix"]["profiles"][scene].copy()
    profile["weight"] = 1.0
    config["scenario"]["training_mix"] = {
        "enabled": True,
        "enabled_in_train": False,
        "profiles": {scene: profile},
    }
    
    # 收集统计数据
    stats = {
        "arrivals": [],
        "load": [],
        "rtt": [],
        "churn": [],
        "queue_size": [],
        "latency": [],
        "tps": [],
        "unsafe_rate": [],
        "timeout_rate": [],
        "structural_infeasible_rate": [],
        "policy_invalid_rate": [],
        "eligible_size": [],
        "committee_size": [],
        "honest_ratio": [],
        "scenario_phase": [],
    }
    
    for ep in tqdm(range(n_episodes), desc=f"场景 {scene}"):
        env = make_env(config)
        obs, info = env.reset(seed=10000 + ep)
        
        done = False
        truncated = False
        step_count = 0
        
        while not (done or truncated):
            # 使用随机动作（仅用于测试场景机制）
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            
            # 收集统计（使用正确的键名）
            stats["arrivals"].append(info.get("A_e", 0))
            stats["load"].append(info.get("Q_e", 0.0) / 100.0)  # 归一化 queue 作为 load 指标
            stats["rtt"].append(info.get("RTT_e", 0.0))
            stats["churn"].append(info.get("chi_e", 0.0))
            stats["queue_size"].append(info.get("Q_e", 0))
            stats["latency"].append(info.get("L_bar_e", 0.0))
            stats["tps"].append(info.get("tps", 0.0))
            stats["unsafe_rate"].append(1.0 if info.get("unsafe", 0) > 0 else 0.0)
            stats["timeout_rate"].append(1.0 if info.get("timeout_failure", 0) > 0 else 0.0)
            stats["structural_infeasible_rate"].append(1.0 if info.get("structural_infeasible", 0) > 0 else 0.0)
            stats["policy_invalid_rate"].append(1.0 if info.get("policy_invalid", 0) > 0 else 0.0)
            stats["eligible_size"].append(info.get("eligible_size", 0))
            stats["committee_size"].append(info.get("executed_committee_size", 0))
            stats["honest_ratio"].append(info.get("h_e", 0.0))
            stats["scenario_phase"].append(info.get("scenario_phase", "unknown"))
            
            step_count += 1
        
        env.close()
    
    # 计算统计摘要
    summary = {}
    for key in stats:
        if key == "scenario_phase":
            # 统计 phase 分布
            phases = stats[key]
            unique_phases = list(set(phases))
            phase_counts = {p: phases.count(p) for p in unique_phases}
            summary[f"{key}_distribution"] = phase_counts
        else:
            values = np.array(stats[key])
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_std"] = float(np.std(values))
            summary[f"{key}_min"] = float(np.min(values))
            summary[f"{key}_max"] = float(np.max(values))
            summary[f"{key}_p50"] = float(np.median(values))
            summary[f"{key}_p95"] = float(np.percentile(values, 95))
    
    return summary


def compare_scenarios(summaries: dict[str, dict]) -> pd.DataFrame:
    """对比不同场景的关键指标。"""
    
    # 选择关键指标
    key_metrics = [
        "arrivals_mean",
        "load_mean",
        "rtt_mean",
        "churn_mean",
        "queue_size_mean",
        "latency_mean",
        "tps_mean",
        "unsafe_rate_mean",
        "timeout_rate_mean",
        "structural_infeasible_rate_mean",
        "eligible_size_mean",
        "committee_size_mean",
        "honest_ratio_mean",
    ]
    
    rows = []
    for scene, summary in summaries.items():
        row = {"scene": scene}
        for metric in key_metrics:
            row[metric] = summary.get(metric, 0.0)
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


def main():
    """主函数。"""
    
    scenes = ["load_shock", "high_rtt_burst", "churn_burst", "malicious_burst"]
    n_episodes = 20
    
    print("=" * 80)
    print("场景机制检查")
    print("=" * 80)
    print(f"每个场景运行 {n_episodes} episodes，使用随机动作")
    print()
    
    summaries = {}
    for scene in scenes:
        print(f"\n检查场景: {scene}")
        print("-" * 80)
        summary = run_scenario_check(scene, n_episodes)
        summaries[scene] = summary
        
        # 打印关键指标
        print(f"\n关键指标摘要:")
        print(f"  arrivals: {summary['arrivals_mean']:.1f} ± {summary['arrivals_std']:.1f}")
        print(f"  load: {summary['load_mean']:.3f} ± {summary['load_std']:.3f}")
        print(f"  rtt: {summary['rtt_mean']:.1f} ± {summary['rtt_std']:.1f} ms")
        print(f"  churn: {summary['churn_mean']:.3f} ± {summary['churn_std']:.3f}")
        print(f"  queue_size: {summary['queue_size_mean']:.1f} ± {summary['queue_size_std']:.1f}")
        print(f"  latency: {summary['latency_mean']:.2f} ± {summary['latency_std']:.2f} ms")
        print(f"  tps: {summary['tps_mean']:.1f} ± {summary['tps_std']:.1f}")
        print(f"  unsafe_rate: {summary['unsafe_rate_mean']:.4f} ± {summary['unsafe_rate_std']:.4f}")
        print(f"  timeout_rate: {summary['timeout_rate_mean']:.4f} ± {summary['timeout_rate_std']:.4f}")
        print(f"  structural_infeasible_rate: {summary['structural_infeasible_rate_mean']:.4f} ± {summary['structural_infeasible_rate_std']:.4f}")
        print(f"  eligible_size: {summary['eligible_size_mean']:.1f} ± {summary['eligible_size_std']:.1f}")
        print(f"  honest_ratio: {summary['honest_ratio_mean']:.3f} ± {summary['honest_ratio_std']:.3f}")
    
    # 生成对比表
    print("\n" + "=" * 80)
    print("场景对比表")
    print("=" * 80)
    df = compare_scenarios(summaries)
    print(df.to_string(index=False))
    
    # 保存结果
    output_dir = ensure_dir(Path("audits/scenario_mechanism_check"))
    df.to_csv(output_dir / "scenario_comparison.csv", index=False)
    write_json(output_dir / "scenario_summaries.json", summaries)
    
    print(f"\n结果已保存到: {output_dir}")
    
    # 生成判断报告
    print("\n" + "=" * 80)
    print("场景机制判断")
    print("=" * 80)
    
    # 判断 load_shock
    load_shock_arrivals = summaries["load_shock"]["arrivals_mean"]
    baseline_arrivals = np.mean([summaries[s]["arrivals_mean"] for s in ["high_rtt_burst", "churn_burst", "malicious_burst"]])
    print(f"\n1. load_shock:")
    print(f"   arrivals: {load_shock_arrivals:.1f} vs baseline {baseline_arrivals:.1f}")
    if load_shock_arrivals > baseline_arrivals * 1.2:
        print(f"   ✅ load_shock 显著提高了到达量")
    else:
        print(f"   ❌ load_shock 未显著提高到达量")
    
    # 判断 high_rtt_burst
    high_rtt_rtt = summaries["high_rtt_burst"]["rtt_mean"]
    baseline_rtt = np.mean([summaries[s]["rtt_mean"] for s in ["load_shock", "churn_burst", "malicious_burst"]])
    print(f"\n2. high_rtt_burst:")
    print(f"   rtt: {high_rtt_rtt:.1f} ms vs baseline {baseline_rtt:.1f} ms")
    if high_rtt_rtt > baseline_rtt * 1.5:
        print(f"   ✅ high_rtt_burst 显著提高了 RTT")
    else:
        print(f"   ❌ high_rtt_burst 未显著提高 RTT")
    
    # 判断 churn_burst
    churn_burst_churn = summaries["churn_burst"]["churn_mean"]
    baseline_churn = np.mean([summaries[s]["churn_mean"] for s in ["load_shock", "high_rtt_burst", "malicious_burst"]])
    churn_burst_infeasible = summaries["churn_burst"]["structural_infeasible_rate_mean"]
    baseline_infeasible = np.mean([summaries[s]["structural_infeasible_rate_mean"] for s in ["load_shock", "high_rtt_burst", "malicious_burst"]])
    print(f"\n3. churn_burst:")
    print(f"   churn: {churn_burst_churn:.3f} vs baseline {baseline_churn:.3f}")
    print(f"   structural_infeasible_rate: {churn_burst_infeasible:.4f} vs baseline {baseline_infeasible:.4f}")
    if churn_burst_churn > baseline_churn * 2.0:
        print(f"   ✅ churn_burst 显著提高了 churn")
    else:
        print(f"   ❌ churn_burst 未显著提高 churn")
    
    # 判断 malicious_burst
    malicious_burst_unsafe = summaries["malicious_burst"]["unsafe_rate_mean"]
    baseline_unsafe = np.mean([summaries[s]["unsafe_rate_mean"] for s in ["load_shock", "high_rtt_burst", "churn_burst"]])
    print(f"\n4. malicious_burst:")
    print(f"   unsafe_rate: {malicious_burst_unsafe:.4f} vs baseline {baseline_unsafe:.4f}")
    if malicious_burst_unsafe > baseline_unsafe * 2.0:
        print(f"   ✅ malicious_burst 显著提高了 unsafe risk")
    else:
        print(f"   ❌ malicious_burst 未显著提高 unsafe risk")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
