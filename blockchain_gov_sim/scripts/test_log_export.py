"""测试日志导出功能。

该脚本用于验证新增的日志导出功能是否正常工作。
"""

from pathlib import Path
import sys

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from gov_sim.utils.log_export import export_training_summary, export_evaluation_summary


def test_training_export():
    """测试训练日志导出。"""
    print("测试训练日志导出...")
    
    # 查找一个已有的训练输出目录
    outputs_dir = project_root / "outputs"
    if not outputs_dir.exists():
        print("未找到 outputs 目录")
        return
    
    # 查找 hierarchical 训练输出
    hierarchical_dirs = list(outputs_dir.glob("formal_multiseed/hierarchical/*/stage2_high_train"))
    if not hierarchical_dirs:
        print("未找到 hierarchical 训练输出")
        return
    
    test_dir = hierarchical_dirs[0]
    print(f"使用测试目录: {test_dir}")
    
    episode_log = test_dir / "train_log.csv"
    step_log = test_dir / "train_log_steps.csv"
    audit = test_dir / "train_audit.json"
    
    if not episode_log.exists():
        print(f"未找到 episode 日志: {episode_log}")
        return
    
    try:
        summary = export_training_summary(
            stage_dir=test_dir,
            episode_log_path=episode_log if episode_log.exists() else None,
            step_log_path=step_log if step_log.exists() else None,
            audit_path=audit if audit.exists() else None,
        )
        
        print("\n导出成功！")
        print(f"- Episode 数量: {summary.get('episode_count', 0)}")
        print(f"- Step 数量: {summary.get('step_count', 0)}")
        
        if "episode_stats" in summary:
            stats = summary["episode_stats"]
            print(f"- 平均 Reward: {stats.get('mean_reward', 0):.2f}")
            print(f"- 平均 Cost: {stats.get('mean_cost', 0):.4f}")
            print(f"- 平均 TPS: {stats.get('mean_tps', 0):.2f}")
            print(f"- 最终 Lambda: {stats.get('final_lambda', 0):.4f}")
        
        if "step_stats" in summary:
            step_stats = summary["step_stats"]
            if "lambda_trajectory" in step_stats:
                lambda_traj = step_stats["lambda_trajectory"]
                print(f"\nLambda 轨迹:")
                print(f"  - Min: {lambda_traj['min']:.4f}")
                print(f"  - Max: {lambda_traj['max']:.4f}")
                print(f"  - Mean: {lambda_traj['mean']:.4f}")
                print(f"  - Final: {lambda_traj['final']:.4f}")
            
            if "high_template_usage" in step_stats:
                print(f"\nHigh Template 使用分布:")
                for item in step_stats["high_template_usage"][:3]:
                    print(f"  - {item['template']}: {item['ratio']:.3f}")
            
            if "low_action_usage" in step_stats:
                print(f"\nLow Action 使用分布:")
                for item in step_stats["low_action_usage"][:3]:
                    print(f"  - {item['action']}: {item['ratio']:.3f}")
        
        export_path = test_dir / "training_export_summary.json"
        print(f"\n导出文件: {export_path}")
        
    except Exception as e:
        print(f"导出失败: {e}")
        import traceback
        traceback.print_exc()


def test_evaluation_export():
    """测试评估日志导出。"""
    print("\n\n测试评估日志导出...")
    
    # 查找一个已有的评估输出目录
    outputs_dir = project_root / "outputs"
    eval_dirs = list(outputs_dir.glob("formal_multiseed/hierarchical/*/hard_eval"))
    
    if not eval_dirs:
        print("未找到评估输出")
        return
    
    test_dir = eval_dirs[0]
    print(f"使用测试目录: {test_dir}")
    
    # 查找所有 CSV 文件
    csv_files = list(test_dir.glob("*.csv"))
    if not csv_files:
        print("未找到 CSV 文件")
        return
    
    print(f"找到 {len(csv_files)} 个 CSV 文件")
    
    try:
        summary = export_evaluation_summary(
            eval_dir=test_dir,
            episode_csv_paths=csv_files,
            metadata={"test": True},
        )
        
        print("\n导出成功！")
        print(f"- Scenario 数量: {len(summary.get('scenarios', {}))}")
        
        for scenario, controllers in summary.get("scenarios", {}).items():
            print(f"\n{scenario}:")
            for controller, metrics in controllers.items():
                print(f"  {controller}:")
                print(f"    - TPS: {metrics.get('mean_tps', 0):.2f}")
                print(f"    - Latency: {metrics.get('mean_latency', 0):.2f}")
                print(f"    - Unsafe Rate: {metrics.get('unsafe_rate', 0):.4f}")
        
        export_path = test_dir / "evaluation_export_summary.json"
        print(f"\n导出文件: {export_path}")
        
    except Exception as e:
        print(f"导出失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_training_export()
    test_evaluation_export()
