"""详细分析评估CSV文件中的模板和低层动作使用情况"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import pandas as pd
import numpy as np
from collections import Counter

def analyze_detailed_csv(csv_path: Path):
    """分析单个评估CSV文件"""
    df = pd.DataFrame(columns=["selected_high_template", "executed_high_template", 
                                "selected_low_action", "executed_low_action",
                                "used_backstop_template"])
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"无法读取 {csv_path}: {e}")
        return None
    
    # 统计高层模板使用
    if "executed_high_template" in df.columns:
        template_counts = df["executed_high_template"].value_counts()
        total = len(df)
        template_dist = [(t, count, count/total) for t, count in template_counts.items()]
    else:
        template_dist = []
    
    # 统计低层动作使用
    if "executed_low_action" in df.columns:
        low_action_counts = df["executed_low_action"].value_counts()
        total = len(df)
        low_action_dist = [(a, count, count/total) for a, count in low_action_counts.items()]
    else:
        low_action_dist = []
    
    # 统计backstop使用
    if "used_backstop_template" in df.columns:
        backstop_rate = df["used_backstop_template"].mean()
    else:
        backstop_rate = 0.0
    
    return {
        "template_distribution": template_dist,
        "low_action_distribution": low_action_dist,
        "backstop_rate": backstop_rate,
        "total_steps": len(df)
    }


def main():
    project_root = Path(__file__).parent.parent.parent.parent
    
    # 分析seed42的四个场景
    seed = 42
    scenarios = ["load_shock", "high_rtt_burst", "churn_burst", "malicious_burst"]
    
    output_dir = project_root / "audits/round2_runtime_checks/action_space_compression"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    for scenario in scenarios:
        csv_path = project_root / f"outputs/formal_multiseed/hierarchical/formal_final_seed{seed}/hard_eval/{scenario}_hierarchical_learned.csv"
        
        if not csv_path.exists():
            print(f"未找到: {csv_path}")
            continue
        
        print(f"\n=== 分析 {scenario} ===")
        result = analyze_detailed_csv(csv_path)
        
        if result:
            print(f"总步数: {result['total_steps']}")
            print(f"Backstop使用率: {result['backstop_rate']:.4f}")
            
            print("\n高层模板分布 (top 5):")
            for template, count, ratio in sorted(result['template_distribution'], key=lambda x: x[1], reverse=True)[:5]:
                print(f"  {template}: {count} ({ratio:.4f})")
            
            print("\n低层动作分布 (top 5):")
            for action, count, ratio in sorted(result['low_action_distribution'], key=lambda x: x[1], reverse=True)[:5]:
                print(f"  {action}: {count} ({ratio:.4f})")
            
            # 保存详细分布
            template_df = pd.DataFrame(result['template_distribution'], 
                                      columns=["template", "count", "ratio"])
            template_df["scenario"] = scenario
            template_df.to_csv(output_dir / f"template_dist_{scenario}.csv", index=False)
            
            low_action_df = pd.DataFrame(result['low_action_distribution'],
                                        columns=["low_action", "count", "ratio"])
            low_action_df["scenario"] = scenario
            low_action_df.to_csv(output_dir / f"low_action_dist_{scenario}.csv", index=False)
    
    print(f"\n详细分布已保存到: {output_dir}")


if __name__ == "__main__":
    main()
