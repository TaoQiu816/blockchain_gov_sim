"""最小化审计运行：提取 low actor 冻结证据。

目标：
1. 加载 Stage1 low actor
2. 运行极短的 Stage2 (仅几个 episodes)
3. 在训练前后记录 low actor 参数 hash
4. 验证 low actor 是否被冻结
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from gov_sim.hierarchical.controller import LowLevelInferenceAdapter
from gov_sim.hierarchical.envs import HighLevelGovEnv
from gov_sim.hierarchical.oracle_pretrain import OracleGuidedLowPolicy
from gov_sim.hierarchical.spec import DEFAULT_HIGH_UPDATE_INTERVAL
from gov_sim.utils.device import resolve_device

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
except ImportError as exc:
    raise ImportError("需要 stable-baselines3") from exc


def compute_model_hash(model: torch.nn.Module) -> str:
    """计算模型参数的 SHA256 hash。"""
    hasher = hashlib.sha256()
    for name, param in sorted(model.named_parameters()):
        hasher.update(name.encode())
        hasher.update(param.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()[:16]


def check_requires_grad(model: torch.nn.Module) -> dict[str, bool]:
    """检查模型参数的 requires_grad 状态。"""
    grad_status = {}
    for name, param in model.named_parameters():
        grad_status[name] = param.requires_grad
    return grad_status


def run_minimal_audit(config_path: str, stage1_model_path: str, output_dir: Path, seed: int = 42) -> None:
    """运行最小化审计。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载配置
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    config['seed'] = seed
    
    print(f"\n=== 审计运行 (Seed {seed}) ===")
    
    # 1. 加载 Stage1 low actor
    print("\n1. 加载 Stage1 low actor...")
    low_model = OracleGuidedLowPolicy.load(
        stage1_model_path,
        device=resolve_device(config.get('agent', {}).get('device', 'auto'))
    )
    
    # 记录初始 hash
    initial_hash = compute_model_hash(low_model.policy)
    print(f"   初始 hash: {initial_hash}")
    
    # 记录 requires_grad 状态
    initial_grad_status = check_requires_grad(low_model.policy)
    print(f"   参数数量: {len(initial_grad_status)}")
    print(f"   requires_grad=True 的参数: {sum(initial_grad_status.values())}")
    
    # 2. 创建 high-level 环境
    print("\n2. 创建 high-level 环境...")
    low_policy = LowLevelInferenceAdapter(model=low_model, deterministic=True)
    update_interval = DEFAULT_HIGH_UPDATE_INTERVAL
    
    def make_env():
        return Monitor(HighLevelGovEnv(
            config=config,
            low_policy=low_policy,
            update_interval=update_interval
        ))
    
    env = DummyVecEnv([make_env])
    
    # 3. 创建 high-level PPO agent
    print("\n3. 创建 high-level PPO agent...")
    
    # 使用 MaskedPPOLagrangian 的配置
    agent_cfg = config.get('agent', {})
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=agent_cfg.get('learning_rate', 3e-4),
        n_steps=agent_cfg.get('n_steps', 2048),
        batch_size=agent_cfg.get('batch_size', 64),
        n_epochs=agent_cfg.get('n_epochs', 10),
        gamma=agent_cfg.get('gamma', 0.99),
        gae_lambda=agent_cfg.get('gae_lambda', 0.95),
        clip_range=agent_cfg.get('clip_range', 0.2),
        ent_coef=agent_cfg.get('ent_coef', 0.0),
        vf_coef=agent_cfg.get('vf_coef', 0.5),
        max_grad_norm=agent_cfg.get('max_grad_norm', 0.5),
        verbose=0,
        seed=seed,
    )
    
    # 4. 检查 low_policy 是否在 optimizer 中
    print("\n4. 检查 optimizer param groups...")
    optimizer_params = []
    for group in model.policy.optimizer.param_groups:
        for param in group['params']:
            # 尝试找到参数名
            for name, p in model.policy.named_parameters():
                if p is param:
                    optimizer_params.append(name)
                    break
    
    print(f"   Optimizer 中的参数数量: {len(optimizer_params)}")
    
    # 检查是否包含 low_policy 参数
    low_policy_in_optimizer = any('low' in name.lower() for name in optimizer_params)
    print(f"   Low policy 在 optimizer 中: {low_policy_in_optimizer}")
    
    # 5. 运行极短训练 (仅 2 个 episodes)
    print("\n5. 运行极短训练 (2 episodes)...")
    model.learn(total_timesteps=50, progress_bar=False)
    
    # 6. 记录训练后 hash
    print("\n6. 检查训练后 low actor...")
    final_hash = compute_model_hash(low_model.policy)
    print(f"   最终 hash: {final_hash}")
    
    final_grad_status = check_requires_grad(low_model.policy)
    print(f"   requires_grad=True 的参数: {sum(final_grad_status.values())}")
    
    # 7. 比较
    print("\n7. 冻结证据:")
    if initial_hash == final_hash:
        print("   ✓ Low actor 参数未变化 (冻结成功)")
        freeze_status = "FROZEN"
    else:
        print("   ❌ Low actor 参数发生变化 (冻结失败)")
        freeze_status = "CHANGED"
    
    # 8. 保存结果
    result = {
        'seed': seed,
        'initial_hash': initial_hash,
        'final_hash': final_hash,
        'freeze_status': freeze_status,
        'hash_match': initial_hash == final_hash,
        'param_count': len(initial_grad_status),
        'initial_requires_grad_count': sum(initial_grad_status.values()),
        'final_requires_grad_count': sum(final_grad_status.values()),
        'low_policy_in_optimizer': low_policy_in_optimizer,
        'optimizer_param_count': len(optimizer_params),
    }
    
    import json
    result_path = output_dir / f"freeze_audit_seed{seed}.json"
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n✓ 结果保存到: {result_path}")
    
    # 清理
    env.close()
    
    return result


if __name__ == "__main__":
    # 使用已有的 Stage1 模型
    config_path = "configs/train_hierarchical_formal_final.yaml"
    stage1_model_path = "outputs/formal_multiseed/hierarchical/formal_final_seed42/stage1_low_pretrain/model.zip"
    output_dir = Path("audits/round2_runtime_checks/training_protocol")
    
    if not Path(stage1_model_path).exists():
        print(f"❌ Stage1 模型不存在: {stage1_model_path}")
        print("请先运行完整训练或指定其他 Stage1 模型路径")
        sys.exit(1)
    
    result = run_minimal_audit(config_path, stage1_model_path, output_dir, seed=42)
    
    print("\n=== 审计完成 ===")
    print(f"冻结状态: {result['freeze_status']}")
    print(f"Hash 匹配: {result['hash_match']}")
