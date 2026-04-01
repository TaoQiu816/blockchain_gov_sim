"""导出 Stage2/Stage3 low actor 冻结证据和 PPO-Lagrangian lambda/cost 运行轨迹。

目标：
1. 从已有 checkpoints 提取 low actor 参数 hash
2. 导出 optimizer param groups 和 requires_grad 状态
3. 从 train_log.csv 提取 lambda 和 cost 轨迹
4. 生成运行证据报告
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
import zipfile

import pandas as pd
import torch


def compute_param_hash(state_dict: dict[str, torch.Tensor], prefix: str = "") -> str:
    """计算参数字典的 SHA256 hash。"""
    relevant_keys = [k for k in sorted(state_dict.keys()) if k.startswith(prefix)]
    if not relevant_keys:
        return "NO_PARAMS"
    
    hasher = hashlib.sha256()
    for key in relevant_keys:
        param = state_dict[key]
        hasher.update(key.encode())
        hasher.update(param.cpu().numpy().tobytes())
    return hasher.hexdigest()[:16]


def extract_model_hashes(model_path: Path) -> dict[str, str]:
    """从 model.zip 提取各组件参数 hash。"""
    if not model_path.exists():
        return {}
    
    hashes = {}
    try:
        import io
        
        with zipfile.ZipFile(model_path, 'r') as zf:
            # 查找 data.pkl 文件
            data_pkl_path = None
            for name in zf.namelist():
                if name.endswith('data.pkl'):
                    data_pkl_path = name
                    break
            
            if not data_pkl_path:
                hashes['error'] = "No data.pkl found"
                return hashes
            
            # 读取 data.pkl
            with zf.open(data_pkl_path) as f:
                import pickle
                data = pickle.load(f)
                
                # 提取 policy state dict
                if 'policy' in data:
                    state_dict = data['policy']
                elif 'policy_state_dict' in data:
                    state_dict = data['policy_state_dict']
                else:
                    # 尝试直接使用 data 作为 state_dict
                    state_dict = data
                
                # 计算不同组件的 hash
                hashes['full_policy'] = compute_param_hash(state_dict)
                
                # 检查是否有 actor/critic 结构
                actor_keys = [k for k in state_dict.keys() if 'action_net' in k or 'policy_net' in k or 'mlp_extractor.policy' in k]
                critic_keys = [k for k in state_dict.keys() if 'value_net' in k or 'mlp_extractor.value' in k]
                
                if actor_keys:
                    hashes['actor'] = compute_param_hash(state_dict, 'mlp_extractor.policy')
                else:
                    hashes['actor'] = "NO_PARAMS"
                
                if critic_keys:
                    hashes['critic'] = compute_param_hash(state_dict, 'mlp_extractor.value')
                else:
                    hashes['critic'] = "NO_PARAMS"
                
                # 尝试提取 low_policy (如果是分层模型)
                low_keys = [k for k in state_dict.keys() if 'low_policy' in k or 'low_actor' in k]
                if low_keys:
                    hashes['low_actor'] = compute_param_hash(state_dict, 'low_policy.')
                else:
                    # Stage1 是 low policy 本身，所以用 full_policy
                    if 'stage1' in str(model_path):
                        hashes['low_actor'] = hashes['full_policy']
                    else:
                        hashes['low_actor'] = "NOT_FOUND"
    except Exception as e:
        hashes['error'] = str(e)
    
    return hashes


def scan_stage_checkpoints(stage_dir: Path) -> list[dict[str, Any]]:
    """扫描 stage 目录中的 checkpoints 和参数 hash。"""
    records = []
    
    # 主模型
    model_path = stage_dir / "model.zip"
    if model_path.exists():
        hashes = extract_model_hashes(model_path)
        records.append({
            'stage': stage_dir.name,
            'checkpoint': 'final',
            'path': str(model_path),
            **hashes
        })
    
    # 检查是否有中间 checkpoints
    for ckpt_path in sorted(stage_dir.glob("checkpoint_*.zip")):
        hashes = extract_model_hashes(ckpt_path)
        records.append({
            'stage': stage_dir.name,
            'checkpoint': ckpt_path.stem,
            'path': str(ckpt_path),
            **hashes
        })
    
    return records


def extract_lambda_cost_trace(train_log_path: Path) -> pd.DataFrame:
    """从 train_log.csv 提取 lambda 和 cost 轨迹。"""
    if not train_log_path.exists():
        return pd.DataFrame()
    
    df = pd.read_csv(train_log_path)
    
    # 选择关键列
    cols = ['timesteps']
    optional_cols = [
        'lagrangian_lambda', 'episode_cost', 'constraint_violation',
        'unsafe_rate', 'rolling_cost_mean', 'rolling_unsafe_mean',
        'rolling_lambda_mean', 'episode_reward', 'rolling_reward_mean'
    ]
    
    for col in optional_cols:
        if col in df.columns:
            cols.append(col)
    
    return df[cols].copy()


def analyze_freeze_evidence(output_dir: Path, seeds: list[int] = [42, 43, 44, 45, 46]) -> None:
    """分析所有 seeds 的冻结证据。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_hashes = []
    all_lambda_traces = []
    
    for seed in seeds:
        seed_dir = Path(f"outputs/formal_multiseed/hierarchical/formal_final_seed{seed}")
        if not seed_dir.exists():
            print(f"⚠️  Seed {seed} 目录不存在: {seed_dir}")
            continue
        
        print(f"\n=== 处理 Seed {seed} ===")
        
        # Stage1: low pretrain
        stage1_dir = seed_dir / "stage1_low_pretrain"
        if stage1_dir.exists():
            stage1_hashes = scan_stage_checkpoints(stage1_dir)
            for h in stage1_hashes:
                h['seed'] = seed
            all_hashes.extend(stage1_hashes)
            print(f"  Stage1: {len(stage1_hashes)} checkpoints")
        
        # Stage2: high train (low frozen)
        stage2_dir = seed_dir / "stage2_high_train"
        if stage2_dir.exists():
            stage2_hashes = scan_stage_checkpoints(stage2_dir)
            for h in stage2_hashes:
                h['seed'] = seed
            all_hashes.extend(stage2_hashes)
            print(f"  Stage2: {len(stage2_hashes)} checkpoints")
            
            # 提取 lambda 轨迹
            train_log = stage2_dir / "train_log.csv"
            if train_log.exists():
                trace = extract_lambda_cost_trace(train_log)
                trace['seed'] = seed
                trace['stage'] = 'stage2_high_train'
                all_lambda_traces.append(trace)
                print(f"  Stage2 lambda trace: {len(trace)} steps")
        
        # Stage3: high refine (low frozen)
        stage3_dir = seed_dir / "stage3_high_refine"
        if stage3_dir.exists():
            stage3_hashes = scan_stage_checkpoints(stage3_dir)
            for h in stage3_hashes:
                h['seed'] = seed
            all_hashes.extend(stage3_hashes)
            print(f"  Stage3: {len(stage3_hashes)} checkpoints")
            
            # 提取 lambda 轨迹
            train_log = stage3_dir / "train_log.csv"
            if train_log.exists():
                trace = extract_lambda_cost_trace(train_log)
                trace['seed'] = seed
                trace['stage'] = 'stage3_high_refine'
                all_lambda_traces.append(trace)
                print(f"  Stage3 lambda trace: {len(trace)} steps")
    
    # 保存 hash 轨迹
    if all_hashes:
        hash_df = pd.DataFrame(all_hashes)
        hash_csv = output_dir / "low_actor_hash_trace.csv"
        hash_df.to_csv(hash_csv, index=False)
        print(f"\n✓ 保存 hash 轨迹: {hash_csv}")
        print(f"  总计 {len(hash_df)} 条记录")
    
    # 保存 lambda 轨迹
    if all_lambda_traces:
        lambda_df = pd.concat(all_lambda_traces, ignore_index=True)
        lambda_csv = output_dir / "lambda_cost_trace.csv"
        lambda_df.to_csv(lambda_csv, index=False)
        print(f"\n✓ 保存 lambda/cost 轨迹: {lambda_csv}")
        print(f"  总计 {len(lambda_df)} 条记录")
    
    # 生成报告
    generate_freeze_report(output_dir, hash_df if all_hashes else None, lambda_df if all_lambda_traces else None)


def generate_freeze_report(output_dir: Path, hash_df: pd.DataFrame | None, lambda_df: pd.DataFrame | None) -> None:
    """生成冻结证据报告。"""
    report_path = output_dir / "freeze_evidence.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Stage2/Stage3 Low Actor 冻结证据与 PPO-Lagrangian 运行轨迹\n\n")
        f.write("## 一、Low Actor 参数冻结证据\n\n")
        
        if hash_df is not None and not hash_df.empty:
            # 按 seed 分组分析
            for seed in sorted(hash_df['seed'].unique()):
                seed_data = hash_df[hash_df['seed'] == seed]
                f.write(f"### Seed {seed}\n\n")
                
                # 提取各 stage 的 low_actor hash (如果列存在)
                if 'low_actor' in seed_data.columns:
                    stage1_hash = seed_data[seed_data['stage'].str.contains('stage1')]['low_actor'].values
                    stage2_hash = seed_data[seed_data['stage'].str.contains('stage2')]['low_actor'].values
                    stage3_hash = seed_data[seed_data['stage'].str.contains('stage3')]['low_actor'].values
                else:
                    stage1_hash = []
                    stage2_hash = []
                    stage3_hash = []
                
                f.write("| Stage | Low Actor Hash | 状态 |\n")
                f.write("|-------|----------------|------|\n")
                
                if len(stage1_hash) > 0:
                    f.write(f"| Stage1 (pretrain) | `{stage1_hash[0]}` | 训练完成 |\n")
                    s1_hash = stage1_hash[0]
                else:
                    f.write("| Stage1 (pretrain) | NOT_FOUND | ⚠️ 缺失 |\n")
                    s1_hash = None
                
                if len(stage2_hash) > 0:
                    s2_match = "✓ 冻结" if s1_hash and stage2_hash[0] == s1_hash else "❌ 变化"
                    f.write(f"| Stage2 (high train) | `{stage2_hash[0]}` | {s2_match} |\n")
                else:
                    f.write("| Stage2 (high train) | NOT_FOUND | ⚠️ 缺失 |\n")
                
                if len(stage3_hash) > 0:
                    s3_match = "✓ 冻结" if s1_hash and stage3_hash[0] == s1_hash else "❌ 变化"
                    f.write(f"| Stage3 (high refine) | `{stage3_hash[0]}` | {s3_match} |\n")
                else:
                    f.write("| Stage3 (high refine) | NOT_FOUND | ⚠️ 缺失 |\n")
                
                f.write("\n")
            
            # 总结
            f.write("### 冻结证据总结\n\n")
            
            freeze_pass = True
            for seed in sorted(hash_df['seed'].unique()):
                seed_data = hash_df[hash_df['seed'] == seed]
                
                if 'low_actor' not in seed_data.columns:
                    continue
                
                stage1_hash = seed_data[seed_data['stage'].str.contains('stage1')]['low_actor'].values
                stage2_hash = seed_data[seed_data['stage'].str.contains('stage2')]['low_actor'].values
                stage3_hash = seed_data[seed_data['stage'].str.contains('stage3')]['low_actor'].values
                
                if len(stage1_hash) > 0 and len(stage2_hash) > 0:
                    if stage1_hash[0] != stage2_hash[0]:
                        f.write(f"- ❌ Seed {seed}: Stage2 low actor hash 与 Stage1 不一致\n")
                        freeze_pass = False
                
                if len(stage1_hash) > 0 and len(stage3_hash) > 0:
                    if stage1_hash[0] != stage3_hash[0]:
                        f.write(f"- ❌ Seed {seed}: Stage3 low actor hash 与 Stage1 不一致\n")
                        freeze_pass = False
            
            if freeze_pass:
                f.write("- ✓ 所有 seeds 的 Stage2/Stage3 low actor 参数与 Stage1 一致\n")
            
            f.write("\n")
        else:
            f.write("⚠️ 未找到 checkpoint 数据\n\n")
        
        # Lambda 和 Cost 轨迹
        f.write("## 二、PPO-Lagrangian Lambda 与 Cost 运行轨迹\n\n")
        
        if lambda_df is not None and not lambda_df.empty:
            # Stage2 分析
            stage2_data = lambda_df[lambda_df['stage'] == 'stage2_high_train']
            if not stage2_data.empty:
                f.write("### Stage2 (High Train)\n\n")
                
                for seed in sorted(stage2_data['seed'].unique()):
                    seed_data = stage2_data[stage2_data['seed'] == seed]
                    
                    if 'lagrangian_lambda' in seed_data.columns:
                        lambda_vals = seed_data['lagrangian_lambda'].dropna()
                        f.write(f"**Seed {seed}:**\n")
                        f.write(f"- Lambda 初始值: {lambda_vals.iloc[0]:.6f}\n")
                        f.write(f"- Lambda 最终值: {lambda_vals.iloc[-1]:.6f}\n")
                        f.write(f"- Lambda 变化: {lambda_vals.iloc[-1] - lambda_vals.iloc[0]:.6f}\n")
                        
                        if 'episode_cost' in seed_data.columns:
                            cost_vals = seed_data['episode_cost'].dropna()
                            f.write(f"- Cost 均值: {cost_vals.mean():.6f}\n")
                            f.write(f"- Cost 最大值: {cost_vals.max():.6f}\n")
                        
                        if 'unsafe_rate' in seed_data.columns:
                            unsafe_vals = seed_data['unsafe_rate'].dropna()
                            f.write(f"- Unsafe rate 均值: {unsafe_vals.mean():.6f}\n")
                        
                        f.write("\n")
            
            # Stage3 分析
            stage3_data = lambda_df[lambda_df['stage'] == 'stage3_high_refine']
            if not stage3_data.empty:
                f.write("### Stage3 (High Refine)\n\n")
                
                for seed in sorted(stage3_data['seed'].unique()):
                    seed_data = stage3_data[stage3_data['seed'] == seed]
                    
                    if 'lagrangian_lambda' in seed_data.columns:
                        lambda_vals = seed_data['lagrangian_lambda'].dropna()
                        f.write(f"**Seed {seed}:**\n")
                        f.write(f"- Lambda 初始值: {lambda_vals.iloc[0]:.6f}\n")
                        f.write(f"- Lambda 最终值: {lambda_vals.iloc[-1]:.6f}\n")
                        f.write(f"- Lambda 变化: {lambda_vals.iloc[-1] - lambda_vals.iloc[0]:.6f}\n")
                        
                        if 'episode_cost' in seed_data.columns:
                            cost_vals = seed_data['episode_cost'].dropna()
                            f.write(f"- Cost 均值: {cost_vals.mean():.6f}\n")
                            f.write(f"- Cost 最大值: {cost_vals.max():.6f}\n")
                        
                        if 'unsafe_rate' in seed_data.columns:
                            unsafe_vals = seed_data['unsafe_rate'].dropna()
                            f.write(f"- Unsafe rate 均值: {unsafe_vals.mean():.6f}\n")
                        
                        f.write("\n")
            
            # Lambda 更新证据
            f.write("### Lambda 更新证据\n\n")
            
            lambda_updated = False
            for stage in ['stage2_high_train', 'stage3_high_refine']:
                stage_data = lambda_df[lambda_df['stage'] == stage]
                if not stage_data.empty and 'lagrangian_lambda' in stage_data.columns:
                    for seed in sorted(stage_data['seed'].unique()):
                        seed_data = stage_data[stage_data['seed'] == seed]
                        lambda_vals = seed_data['lagrangian_lambda'].dropna()
                        
                        if len(lambda_vals) > 1:
                            lambda_change = abs(lambda_vals.iloc[-1] - lambda_vals.iloc[0])
                            if lambda_change > 1e-6:
                                f.write(f"- ✓ {stage} Seed {seed}: Lambda 有更新 (变化 {lambda_change:.6f})\n")
                                lambda_updated = True
                            else:
                                f.write(f"- ⚠️ {stage} Seed {seed}: Lambda 未更新 (变化 {lambda_change:.6f})\n")
            
            if not lambda_updated:
                f.write("- ❌ 所有 stage 的 lambda 均未更新\n")
            
            f.write("\n")
        else:
            f.write("⚠️ 未找到 lambda/cost 轨迹数据\n\n")
        
        # 最终结论
        f.write("## 三、最终结论\n\n")
        
        conclusions = []
        
        # 1. Stage2 low actor 冻结
        if hash_df is not None and not hash_df.empty and 'low_actor' in hash_df.columns:
            stage2_frozen = True
            for seed in sorted(hash_df['seed'].unique()):
                seed_data = hash_df[hash_df['seed'] == seed]
                stage1_hash = seed_data[seed_data['stage'].str.contains('stage1')]['low_actor'].values
                stage2_hash = seed_data[seed_data['stage'].str.contains('stage2')]['low_actor'].values
                if len(stage1_hash) > 0 and len(stage2_hash) > 0 and stage1_hash[0] != stage2_hash[0]:
                    stage2_frozen = False
            
            conclusions.append(("Stage2 low actor 参数冻结", "通过" if stage2_frozen else "不通过"))
        else:
            conclusions.append(("Stage2 low actor 参数冻结", "存疑（无数据）"))
        
        # 2. Stage3 low actor 冻结
        if hash_df is not None and not hash_df.empty and 'low_actor' in hash_df.columns:
            stage3_frozen = True
            for seed in sorted(hash_df['seed'].unique()):
                seed_data = hash_df[hash_df['seed'] == seed]
                stage1_hash = seed_data[seed_data['stage'].str.contains('stage1')]['low_actor'].values
                stage3_hash = seed_data[seed_data['stage'].str.contains('stage3')]['low_actor'].values
                if len(stage1_hash) > 0 and len(stage3_hash) > 0 and stage1_hash[0] != stage3_hash[0]:
                    stage3_frozen = False
            
            conclusions.append(("Stage3 low actor 参数冻结", "通过" if stage3_frozen else "不通过"))
        else:
            conclusions.append(("Stage3 low actor 参数冻结", "存疑（无数据）"))
        
        # 3. Lambda 更新
        if lambda_df is not None and not lambda_df.empty and 'lagrangian_lambda' in lambda_df.columns:
            lambda_updated = False
            for stage in ['stage2_high_train', 'stage3_high_refine']:
                stage_data = lambda_df[lambda_df['stage'] == stage]
                if not stage_data.empty:
                    for seed in sorted(stage_data['seed'].unique()):
                        seed_data = stage_data[stage_data['seed'] == seed]
                        lambda_vals = seed_data['lagrangian_lambda'].dropna()
                        if len(lambda_vals) > 1 and abs(lambda_vals.iloc[-1] - lambda_vals.iloc[0]) > 1e-6:
                            lambda_updated = True
            
            conclusions.append(("PPO-Lagrangian Lambda 更新", "通过" if lambda_updated else "不通过"))
        else:
            conclusions.append(("PPO-Lagrangian Lambda 更新", "存疑（无数据）"))
        
        # 4. Cost 约束生效
        if lambda_df is not None and not lambda_df.empty and 'episode_cost' in lambda_df.columns:
            cost_recorded = lambda_df['episode_cost'].notna().any()
            conclusions.append(("Cost 约束记录", "通过" if cost_recorded else "不通过"))
        else:
            conclusions.append(("Cost 约束记录", "存疑（无数据）"))
        
        f.write("| 检查项 | 结论 |\n")
        f.write("|--------|------|\n")
        for item, result in conclusions:
            emoji = "✓" if result == "通过" else ("⚠️" if "存疑" in result else "❌")
            f.write(f"| {item} | {emoji} {result} |\n")
        
        f.write("\n")
        
        # 总体评估
        all_pass = all(result == "通过" for _, result in conclusions)
        any_fail = any("不通过" in result for _, result in conclusions)
        
        if all_pass:
            f.write("**总体评估：✓ 通过**\n\n")
            f.write("所有检查项均通过，训练协议符合预期。\n")
        elif any_fail:
            f.write("**总体评估：❌ 不通过**\n\n")
            f.write("存在检查项不通过，需要进一步调查。\n")
        else:
            f.write("**总体评估：⚠️ 存疑**\n\n")
            f.write("部分检查项缺少数据，无法完全验证。\n")
    
    print(f"\n✓ 生成报告: {report_path}")


if __name__ == "__main__":
    output_dir = Path("audits/round2_runtime_checks/training_protocol")
    analyze_freeze_evidence(output_dir)
    print("\n✓ 完成")
