#!/usr/bin/env python3
"""
扫描 formal_final_seed* 的训练产物，核验高层动作空间维度
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import csv

# 使用绝对路径，从脚本所在位置向上两级到项目根目录
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "formal_multiseed" / "hierarchical"
AUDIT_ROOT = SCRIPT_DIR

def check_seed_high_dim(seed_dir: Path) -> Dict:
    """检查单个 seed 的高层动作空间维度"""
    result = {
        "seed": seed_dir.name,
        "config_exists": False,
        "stage2_exists": False,
        "stage3_exists": False,
        "has_m5_in_stage2": False,
        "has_m5_in_stage3": False,
        "m5_ratio_stage2": 0.0,
        "m5_ratio_stage3": 0.0,
        "high_dim_confirmed": False,
        "evidence_files": []
    }
    
    # 检查 config_snapshot.json
    config_path = seed_dir / "config_snapshot.json"
    if config_path.exists():
        result["config_exists"] = True
        result["evidence_files"].append(str(config_path))
    
    # 检查 stage2_high_train/train_audit.json
    stage2_audit = seed_dir / "stage2_high_train" / "train_audit.json"
    if stage2_audit.exists():
        result["stage2_exists"] = True
        result["evidence_files"].append(str(stage2_audit))
        
        with open(stage2_audit, 'r') as f:
            audit_data = json.load(f)
        
        # 检查 action_frequency 中是否有 m=5
        if "action_frequency" in audit_data and "m" in audit_data["action_frequency"]:
            m_freq = audit_data["action_frequency"]["m"]
            if "5" in m_freq:
                result["has_m5_in_stage2"] = True
                result["m5_ratio_stage2"] = m_freq["5"]
        
        # 检查 executed_high_template_distribution 中是否有 m=5
        if "executed_high_template_distribution" in audit_data:
            for template in audit_data["executed_high_template_distribution"]:
                action = template.get("action", "")
                if action.startswith("5|"):
                    result["has_m5_in_stage2"] = True
                    break
    
    # 检查 stage3_high_refine/round_*_high_train_audit.json
    stage3_dir = seed_dir / "stage3_high_refine"
    if stage3_dir.exists():
        for round_file in sorted(stage3_dir.glob("round_*_high_train_audit.json")):
            result["stage3_exists"] = True
            result["evidence_files"].append(str(round_file))
            
            with open(round_file, 'r') as f:
                audit_data = json.load(f)
            
            # 检查 action_frequency 中是否有 m=5
            if "action_frequency" in audit_data and "m" in audit_data["action_frequency"]:
                m_freq = audit_data["action_frequency"]["m"]
                if "5" in m_freq:
                    result["has_m5_in_stage3"] = True
                    result["m5_ratio_stage3"] = max(result["m5_ratio_stage3"], m_freq["5"])
            
            # 检查 executed_high_template_distribution 中是否有 m=5
            if "executed_high_template_distribution" in audit_data:
                for template in audit_data["executed_high_template_distribution"]:
                    action = template.get("action", "")
                    if action.startswith("5|"):
                        result["has_m5_in_stage3"] = True
                        break
    
    # 综合判断
    result["high_dim_confirmed"] = result["has_m5_in_stage2"] or result["has_m5_in_stage3"]
    
    return result

def scan_all_seeds() -> List[Dict]:
    """扫描所有 formal_final_seed* 目录"""
    results = []
    
    if not OUTPUT_ROOT.exists():
        print(f"输出目录不存在: {OUTPUT_ROOT}")
        return results
    
    # 扫描所有 formal_final_seed* 目录
    for seed_dir in sorted(OUTPUT_ROOT.glob("formal_final_seed*")):
        if seed_dir.is_dir():
            print(f"扫描: {seed_dir.name}")
            result = check_seed_high_dim(seed_dir)
            results.append(result)
    
    return results

def export_csv(results: List[Dict], output_path: Path):
    """导出 CSV 汇总表"""
    if not results:
        print("没有结果可导出")
        return
    
    fieldnames = [
        "seed",
        "config_exists",
        "stage2_exists",
        "stage3_exists",
        "has_m5_in_stage2",
        "has_m5_in_stage3",
        "m5_ratio_stage2",
        "m5_ratio_stage3",
        "high_dim_confirmed",
        "evidence_files"
    ]
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            row = result.copy()
            row["evidence_files"] = "; ".join(result["evidence_files"])
            writer.writerow(row)
    
    print(f"CSV 已导出: {output_path}")

def generate_report(results: List[Dict], output_path: Path):
    """生成 Markdown 报告"""
    with open(output_path, 'w') as f:
        f.write("# Checkpoint 高层动作空间维度审计报告\n\n")
        f.write(f"**审计时间**: {Path(__file__).stat().st_mtime}\n\n")
        f.write("## 审计目标\n\n")
        f.write("核验 `outputs/formal_multiseed/hierarchical/formal_final_seed*` 下的训练结果，\n")
        f.write("确认是否基于 12 维高层动作空间（包含 m=5 的 nominal 模板）训练。\n\n")
        
        f.write("## 审计方法\n\n")
        f.write("1. 扫描每个 seed 的 `config_snapshot.json`\n")
        f.write("2. 检查 `stage2_high_train/train_audit.json` 中的 action_frequency 和 executed_high_template_distribution\n")
        f.write("3. 检查 `stage3_high_refine/round_*_high_train_audit.json` 中的相同字段\n")
        f.write("4. 判断是否出现 m=5 的模板（nominal 模板的标志）\n\n")
        
        f.write("## 审计结果汇总\n\n")
        f.write("| Seed | Stage2 存在 | Stage3 存在 | Stage2 有 m=5 | Stage3 有 m=5 | m=5 比例 (Stage2) | m=5 比例 (Stage3) | 12 维确认 |\n")
        f.write("|------|------------|------------|---------------|---------------|-------------------|-------------------|----------|\n")
        
        for result in results:
            f.write(f"| {result['seed']} | ")
            f.write(f"{'✓' if result['stage2_exists'] else '✗'} | ")
            f.write(f"{'✓' if result['stage3_exists'] else '✗'} | ")
            f.write(f"{'✓' if result['has_m5_in_stage2'] else '✗'} | ")
            f.write(f"{'✓' if result['has_m5_in_stage3'] else '✗'} | ")
            f.write(f"{result['m5_ratio_stage2']:.4f} | ")
            f.write(f"{result['m5_ratio_stage3']:.4f} | ")
            f.write(f"{'✓' if result['high_dim_confirmed'] else '✗'} |\n")
        
        f.write("\n## 详细分析\n\n")
        
        for result in results:
            f.write(f"### {result['seed']}\n\n")
            
            if result['high_dim_confirmed']:
                f.write("**结论**: ✅ 确认为 12 维高层动作空间训练结果\n\n")
            else:
                f.write("**结论**: ❌ 未确认为 12 维高层动作空间训练结果\n\n")
            
            f.write("**证据来源**:\n")
            for evidence in result['evidence_files']:
                f.write(f"- `{evidence}`\n")
            
            f.write("\n**详细信息**:\n")
            f.write(f"- Config 存在: {'是' if result['config_exists'] else '否'}\n")
            f.write(f"- Stage2 训练审计存在: {'是' if result['stage2_exists'] else '否'}\n")
            f.write(f"- Stage3 训练审计存在: {'是' if result['stage3_exists'] else '否'}\n")
            
            if result['has_m5_in_stage2']:
                f.write(f"- Stage2 中 m=5 出现比例: {result['m5_ratio_stage2']:.4f}\n")
            
            if result['has_m5_in_stage3']:
                f.write(f"- Stage3 中 m=5 出现比例: {result['m5_ratio_stage3']:.4f}\n")
            
            f.write("\n")
        
        f.write("## 最终结论\n\n")
        
        confirmed_count = sum(1 for r in results if r['high_dim_confirmed'])
        total_count = len(results)
        
        f.write(f"- 总共扫描: {total_count} 个 seed\n")
        f.write(f"- 确认为 12 维: {confirmed_count} 个\n")
        f.write(f"- 未确认为 12 维: {total_count - confirmed_count} 个\n\n")
        
        if confirmed_count == total_count:
            f.write("✅ **所有 formal_final_seed* 结果均确认为 12 维高层动作空间训练结果，可以继续使用。**\n")
        elif confirmed_count == 0:
            f.write("❌ **所有 formal_final_seed* 结果均未确认为 12 维高层动作空间训练结果，必须重新训练。**\n")
        else:
            f.write("⚠️ **部分 formal_final_seed* 结果确认为 12 维，部分未确认。需要逐个检查。**\n\n")
            f.write("**可用结果**:\n")
            for r in results:
                if r['high_dim_confirmed']:
                    f.write(f"- {r['seed']}\n")
            f.write("\n**不可用结果（需重新训练）**:\n")
            for r in results:
                if not r['high_dim_confirmed']:
                    f.write(f"- {r['seed']}\n")
    
    print(f"报告已生成: {output_path}")

def main():
    """主函数"""
    print("=" * 60)
    print("Checkpoint 高层动作空间维度审计")
    print("=" * 60)
    print()
    
    # 扫描所有 seed
    results = scan_all_seeds()
    
    if not results:
        print("未找到任何 formal_final_seed* 目录")
        return
    
    # 创建输出目录
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    
    # 导出 CSV
    csv_path = AUDIT_ROOT / "checkpoint_high_dim_audit.csv"
    export_csv(results, csv_path)
    
    # 生成报告
    report_path = AUDIT_ROOT / "report_checkpoint_high_dim.md"
    generate_report(results, report_path)
    
    print()
    print("=" * 60)
    print("审计完成")
    print("=" * 60)

if __name__ == "__main__":
    main()
