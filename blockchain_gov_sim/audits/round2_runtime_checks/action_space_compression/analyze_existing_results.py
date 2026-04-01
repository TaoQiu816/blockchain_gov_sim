"""分析已有评估结果中的backstop使用率和structural_infeasible率

这个脚本直接分析已有的hard_eval结果，提取关键证据
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import pandas as pd
import json

def analyze_hard_eval_results(eval_dir: Path):
    """分析hard_eval目录中的结果"""
    
    # 读取hard_compare.json
    compare_json_path = eval_dir / "hard_compare.json"
    if not compare_json_path.exists():
        print(f"未找到: {compare_json_path}")
        return None
    
    with open(compare_json_path, 'r') as f:
        data = json.load(f)
    
    results = []
    
    # 提取每个场景的hierarchical_learned结果
    for scenario_name, scenario_data in data.get("scenarios", {}).items():
        hier_data = scenario_data.get("hierarchical_learned", {})
        
        result = {
            "scenario": scenario_name,
            "TPS": hier_data.get("TPS", hier_data.get("tps", 0)),
            "mean_latency": hier_data.get("mean_latency", 0),
            "unsafe_rate": hier_data.get("unsafe_rate", 0),
            "timeout_rate": hier_data.get("timeout_failure_rate", 0),
            "policy_invalid_rate": hier_data.get("policy_invalid_rate", 0),
            "structural_infeasible_rate": hier_data.get("structural_infeasible_rate", 0),
            "used_backstop_template_rate": hier_data.get("used_backstop_template_rate", 0),
            "top_template": hier_data.get("top_template", ""),
            "template_dominant_ratio": hier_data.get("template_dominant_ratio", 0),
            "top_low_action": hier_data.get("top_low_action", ""),
            "low_action_dominant_ratio": hier_data.get("low_action_dominant_ratio", 0),
        }
        results.append(result)
    
    return pd.DataFrame(results)


def main():
    project_root = Path(__file__).parent.parent.parent.parent
    
    # 分析所有seed的结果
    seeds = [42, 43, 44, 45, 46]
    all_results = []
    
    for seed in seeds:
        eval_dir = project_root / f"outputs/formal_multiseed/hierarchical/formal_final_seed{seed}/hard_eval"
        if not eval_dir.exists():
            print(f"跳过不存在的目录: {eval_dir}")
            continue
        
        print(f"\n=== 分析 seed{seed} ===")
        df = analyze_hard_eval_results(eval_dir)
        if df is not None:
            df["seed"] = seed
            all_results.append(df)
            print(df[["scenario", "used_backstop_template_rate", "structural_infeasible_rate", 
                     "policy_invalid_rate", "top_template"]].to_string(index=False))
    
    if not all_results:
        print("未找到任何结果")
        return
    
    # 合并所有结果
    combined = pd.concat(all_results, ignore_index=True)
    
    # 保存
    output_dir = project_root / "audits/round2_runtime_checks/action_space_compression"
    output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_dir / "existing_results_analysis.csv", index=False)
    
    # 计算统计
    summary = combined.groupby("scenario").agg({
        "used_backstop_template_rate": ["mean", "std"],
        "structural_infeasible_rate": ["mean", "std"],
        "policy_invalid_rate": ["mean", "std"],
        "TPS": ["mean", "std"],
        "unsafe_rate": ["mean", "std"],
    }).reset_index()
    
    summary.columns = ["_".join(col).strip("_") for col in summary.columns.values]
    summary.to_csv(output_dir / "scenario_summary.csv", index=False)
    
    print("\n=== 场景汇总统计 ===")
    print(summary.to_string(index=False))
    
    # 关键发现
    print("\n=== 关键发现 ===")
    print(f"1. 平均backstop使用率: {combined['used_backstop_template_rate'].mean():.4f}")
    print(f"2. 平均structural_infeasible率: {combined['structural_infeasible_rate'].mean():.4f}")
    print(f"3. 平均policy_invalid率: {combined['policy_invalid_rate'].mean():.4f}")
    
    # 按场景分析
    print("\n按场景分析backstop使用:")
    for scenario in combined["scenario"].unique():
        scenario_data = combined[combined["scenario"] == scenario]
        print(f"  {scenario}: backstop={scenario_data['used_backstop_template_rate'].mean():.4f}, "
              f"structural_infeasible={scenario_data['structural_infeasible_rate'].mean():.4f}")
    
    print(f"\n结果已保存到: {output_dir}")


if __name__ == "__main__":
    main()
