"""固定高层模板前沿评估脚本 - 语义验收版本。

该脚本用于评估 12 个固定高层模板在 4 个 hard 场景下的性能，
目标是判断 learned high policy 集中在 7|0.55 是合理收敛还是训练问题。

关键改进：
1. 补充 requested vs applied 模板字段
2. 固定并保存 eval seed manifest
3. 补充 metadata 中的 low actor / deterministic 证据
4. 支持最小 smoke 验收模式

使用方法:
    # 最小验收 (1 ckpt, 1 scene, 2 templates, 5 episodes)
    python scripts/eval_fixed_high_frontier.py --mode minimal
    
    # Smoke 模式 (N_eval=20)
    python scripts/eval_fixed_high_frontier.py --mode smoke
    
    # 正式模式 (N_eval=64)
    python scripts/eval_fixed_high_frontier.py --mode full
"""

from __future__ import annotations

import argparse
import hashlib
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from gov_sim.experiments import make_env
from gov_sim.hierarchical.controller import (
    HierarchicalPolicyController,
    LowLevelInferenceAdapter,
)
from gov_sim.hierarchical.fixed_template_selector import FixedTemplateSelector
from gov_sim.hierarchical.spec import HIGH_LEVEL_TEMPLATES, HighLevelAction, HierarchicalActionCodec
from gov_sim.modules.metrics_tracker import MetricsTracker
from gov_sim.utils.io import ensure_dir, load_config, write_json

try:
    from stable_baselines3 import PPO
    from sb3_contrib import MaskablePPO
    from gov_sim.agent.masked_ppo_lagrangian import MaskablePPOLagrangian
    from gov_sim.hierarchical.oracle_pretrain import OracleGuidedLowPolicy
except ImportError as exc:
    raise ImportError("需要 stable-baselines3 和 sb3-contrib，请执行: pip install -r base_env_delta_requirements.txt") from exc

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 固定配置
CHECKPOINTS = [
    "outputs/formal_multiseed/hierarchical/formal_final_seed42/stage3_high_refine/high_model.zip",
    "outputs/formal_multiseed/hierarchical/formal_final_seed43/stage3_high_refine/high_model.zip",
    "outputs/formal_multiseed/hierarchical/formal_final_seed44/stage3_high_refine/high_model.zip",
    "outputs/formal_multiseed/hierarchical/formal_final_seed45/stage3_high_refine/high_model.zip",
    "outputs/formal_multiseed/hierarchical/formal_final_seed46/stage3_high_refine/high_model.zip",
]

LOW_MODEL_PATHS = [
    "outputs/formal_multiseed/hierarchical/formal_final_seed42/stage3_high_refine/low_model.zip",
    "outputs/formal_multiseed/hierarchical/formal_final_seed43/stage3_high_refine/low_model.zip",
    "outputs/formal_multiseed/hierarchical/formal_final_seed44/stage3_high_refine/low_model.zip",
    "outputs/formal_multiseed/hierarchical/formal_final_seed45/stage3_high_refine/low_model.zip",
    "outputs/formal_multiseed/hierarchical/formal_final_seed46/stage3_high_refine/low_model.zip",
]

SCENES = ["load_shock", "high_rtt_burst", "churn_burst", "malicious_burst"]

# 12 个固定模板
FIXED_TEMPLATES = HIGH_LEVEL_TEMPLATES  # (5,0.45), (5,0.50), ..., (9,0.60)


def _compute_model_hash(model_path: Path) -> str:
    """计算模型文件的 hash。"""
    if not model_path.exists():
        return "not_found"
    
    with open(model_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()[:16]


def _generate_eval_seeds(scene: str, n_episodes: int, base_seed: int = 10000) -> list[int]:
    """为场景生成固定的 eval seeds。"""
    scene_offset = SCENES.index(scene) * 1000 if scene in SCENES else 0
    return [base_seed + scene_offset + i for i in range(n_episodes)]


def _build_scene_config(base_config: dict[str, Any], scene: str) -> dict[str, Any]:
    """构建场景配置。"""
    config = deepcopy(base_config)
    profile = deepcopy(config["scenario"]["training_mix"]["profiles"][scene])
    profile["weight"] = 1.0
    config["scenario"]["training_mix"] = {
        "enabled": True,
        "enabled_in_train": False,
        "profiles": {scene: profile},
    }
    return config


def _evaluate_fixed_template(
    low_model: Any,
    template: tuple[int, float],
    config: dict[str, Any],
    eval_seeds: list[int],
    checkpoint_name: str,
    scene: str,
) -> list[dict[str, Any]]:
    """评估单个固定模板。"""
    codec = HierarchicalActionCodec()
    high_action = HighLevelAction(m=template[0], theta=template[1])
    template_selector = FixedTemplateSelector(template=high_action, codec=codec)
    low_adapter = LowLevelInferenceAdapter(model=low_model, deterministic=True, codec=codec)
    
    env = make_env(config)
    env.set_invalid_action_mode("raise")
    
    rows: list[dict[str, Any]] = []
    
    for eval_seed in eval_seeds:
        obs, _ = env.reset(seed=eval_seed)
        template_selector.reset()
        low_adapter.reset()
        
        done = False
        truncated = False
        step_count = 0
        tracker = MetricsTracker()
        
        selected_b_list: list[int] = []
        selected_tau_list: list[int] = []
        
        # 记录 episode 级别的 remap 信息
        episode_remap_count = 0
        
        while not (done or truncated):
            # 获取固定模板
            current_template = template_selector.template_for_step(env.unwrapped)
            
            # 检查是否发生 remap
            if template_selector.was_remapped:
                episode_remap_count += 1
            
            # 低层选择动作
            _, low_action, _ = low_adapter.select(env.unwrapped, current_template)
            
            # 执行动作
            flat_action_idx = codec.flat_index(high_action=current_template, low_action=low_action)
            obs, reward, done, truncated, info = env.step(flat_action_idx)
            
            selected_b_list.append(low_action.b)
            selected_tau_list.append(low_action.tau)
            
            tracker.record(
                reward=float(info.get("reward", 0.0)),
                cost=float(info.get("cost", 0.0)),
                tps=float(info.get("tps", 0.0)),
                latency=float(info.get("L_bar_e", 0.0)),
                unsafe=int(info.get("unsafe", 0)),
                timeout=int(info.get("timeout_failure", 0)),
                policy_invalid=int(info.get("policy_invalid", 0)),
                structural_infeasible=int(info.get("structural_infeasible", 0)),
                queue_mean=float(info.get("queue_mean", 0.0)),
                queue_p95=float(info.get("queue_p95", 0.0)),
            )
            
            template_selector.on_step_complete()
            step_count += 1
        
        # 获取 remap 信息
        remap_info = template_selector.get_remap_info()
        
        summary = tracker.summary()
        row = {
            "checkpoint_name": checkpoint_name,
            "scene": scene,
            "eval_seed": eval_seed,
            "template_idx": FIXED_TEMPLATES.index(template),
            "m": template[0],
            "theta": template[1],
            "steps": step_count,
            "TPS": summary["tps_mean"],
            "mean_latency": summary["latency_mean"],
            "unsafe_rate": summary["unsafe_rate"],
            "timeout_rate": summary["timeout_rate"],
            "policy_invalid_rate": summary.get("policy_invalid_rate", 0.0),
            "structural_infeasible_rate": summary.get("structural_infeasible_rate", 0.0),
            "queue_mean": summary.get("queue_mean_mean", 0.0),
            "queue_p95": summary.get("queue_p95_mean", 0.0),
            "avg_selected_b": float(np.mean(selected_b_list)) if selected_b_list else 0.0,
            "avg_selected_tau": float(np.mean(selected_tau_list)) if selected_tau_list else 0.0,
            "episode_remap_count": episode_remap_count,
        }
        
        # 补充 requested vs applied 字段
        row.update(remap_info)
        
        rows.append(row)
    
    return rows


def _evaluate_learned_baseline(
    high_model: Any,
    low_model: Any,
    config: dict[str, Any],
    eval_seeds: list[int],
    checkpoint_name: str,
    scene: str,
) -> list[dict[str, Any]]:
    """评估 learned high + learned low baseline。"""
    controller = HierarchicalPolicyController(
        high_model=high_model,
        low_model=low_model,
        update_interval=10,
        deterministic=True,
    )
    
    env = make_env(config)
    env.set_invalid_action_mode("raise")
    
    rows: list[dict[str, Any]] = []
    
    for eval_seed in eval_seeds:
        obs, _ = env.reset(seed=eval_seed)
        controller.reset()
        
        done = False
        truncated = False
        step_count = 0
        tracker = MetricsTracker()
        
        while not (done or truncated):
            flat_action_idx = controller.select_action(env, obs)
            obs, reward, done, truncated, info = env.step(flat_action_idx)
            
            tracker.record(
                reward=float(info.get("reward", 0.0)),
                cost=float(info.get("cost", 0.0)),
                tps=float(info.get("tps", 0.0)),
                latency=float(info.get("L_bar_e", 0.0)),
                unsafe=int(info.get("unsafe", 0)),
                timeout=int(info.get("timeout_failure", 0)),
                policy_invalid=int(info.get("policy_invalid", 0)),
                structural_infeasible=int(info.get("structural_infeasible", 0)),
                queue_mean=float(info.get("queue_mean", 0.0)),
                queue_p95=float(info.get("queue_p95", 0.0)),
            )
            
            controller.on_step_complete()
            step_count += 1
        
        summary = tracker.summary()
        rows.append({
            "checkpoint_name": checkpoint_name,
            "scene": scene,
            "eval_seed": eval_seed,
            "steps": step_count,
            "TPS": summary["tps_mean"],
            "mean_latency": summary["latency_mean"],
            "unsafe_rate": summary["unsafe_rate"],
            "timeout_rate": summary["timeout_rate"],
            "policy_invalid_rate": summary.get("policy_invalid_rate", 0.0),
            "structural_infeasible_rate": summary.get("structural_infeasible_rate", 0.0),
            "queue_mean": summary.get("queue_mean_mean", 0.0),
            "queue_p95": summary.get("queue_p95_mean", 0.0),
        })
    
    return rows


def run_evaluation(mode: str, output_dir: Path) -> None:
    """运行完整评估。"""
    # 根据模式确定评估参数
    if mode == "minimal":
        n_episodes = 5
        checkpoints_to_eval = [CHECKPOINTS[0]]
        low_models_to_eval = [LOW_MODEL_PATHS[0]]
        scenes_to_eval = [SCENES[0]]
        templates_to_eval = [FIXED_TEMPLATES[0], FIXED_TEMPLATES[6]]  # (5,0.45) 和 (7,0.45)
    elif mode == "smoke":
        n_episodes = 20
        checkpoints_to_eval = CHECKPOINTS
        low_models_to_eval = LOW_MODEL_PATHS
        scenes_to_eval = SCENES
        templates_to_eval = FIXED_TEMPLATES
    else:  # full
        n_episodes = 64
        checkpoints_to_eval = CHECKPOINTS
        low_models_to_eval = LOW_MODEL_PATHS
        scenes_to_eval = SCENES
        templates_to_eval = FIXED_TEMPLATES
    
    base_config = load_config("configs/default.yaml", "configs/train_hierarchical_formal_final.yaml")
    
    logger.info(f"开始固定高层模板前沿评估 (mode={mode}, n_episodes={n_episodes})")
    
    # 生成 eval seed manifest
    eval_seed_manifest: dict[str, list[int]] = {}
    for scene in scenes_to_eval:
        eval_seed_manifest[scene] = _generate_eval_seeds(scene, n_episodes)
    
    # 保存 seed manifest
    write_json(output_dir / "eval_seed_manifest.json", eval_seed_manifest)
    logger.info(f"保存 eval seed manifest: {output_dir / 'eval_seed_manifest.json'}")
    
    all_fixed_rows: list[dict[str, Any]] = []
    all_learned_rows: list[dict[str, Any]] = []
    
    checkpoint_metadata: list[dict[str, Any]] = []
    
    for ckpt_idx, (high_ckpt_path, low_ckpt_path) in enumerate(zip(checkpoints_to_eval, low_models_to_eval)):
        train_seed = 42 + ckpt_idx
        ckpt_name = f"formal_final_seed{train_seed}"
        
        logger.info(f"加载 checkpoint: {ckpt_name}")
        
        high_ckpt = Path(high_ckpt_path)
        low_ckpt = Path(low_ckpt_path)
        
        if not high_ckpt.exists():
            logger.warning(f"跳过不存在的 checkpoint: {high_ckpt}")
            continue
        if not low_ckpt.exists():
            logger.warning(f"跳过不存在的 low model: {low_ckpt}")
            continue
        
        # 使用 MaskablePPOLagrangian 加载 high 模型
        try:
            high_model = MaskablePPOLagrangian.load(high_ckpt)
        except Exception:
            try:
                high_model = MaskablePPO.load(high_ckpt)
            except Exception:
                high_model = PPO.load(high_ckpt)
        
        # 使用 OracleGuidedLowPolicy 加载 low 模型
        try:
            low_model = OracleGuidedLowPolicy.load(low_ckpt)
        except Exception:
            try:
                low_model = MaskablePPOLagrangian.load(low_ckpt)
            except Exception:
                try:
                    low_model = MaskablePPO.load(low_ckpt)
                except Exception:
                    low_model = PPO.load(low_ckpt)
        
        # 计算模型 hash
        high_hash = _compute_model_hash(high_ckpt)
        low_hash = _compute_model_hash(low_ckpt)
        
        checkpoint_metadata.append({
            "checkpoint_name": ckpt_name,
            "train_seed": train_seed,
            "high_model_path": str(high_ckpt),
            "low_model_path": str(low_ckpt),
            "high_model_hash": high_hash,
            "low_model_hash": low_hash,
        })
        
        for scene in scenes_to_eval:
            logger.info(f"  场景: {scene}")
            scene_config = _build_scene_config(base_config, scene)
            eval_seeds = eval_seed_manifest[scene]
            
            # 评估 learned baseline
            logger.info(f"    评估 learned baseline...")
            learned_rows = _evaluate_learned_baseline(
                high_model=high_model,
                low_model=low_model,
                config=scene_config,
                eval_seeds=eval_seeds,
                checkpoint_name=ckpt_name,
                scene=scene,
            )
            all_learned_rows.extend(learned_rows)
            
            # 评估固定模板
            for template in tqdm(templates_to_eval, desc=f"    固定模板"):
                fixed_rows = _evaluate_fixed_template(
                    low_model=low_model,
                    template=template,
                    config=scene_config,
                    eval_seeds=eval_seeds,
                    checkpoint_name=ckpt_name,
                    scene=scene,
                )
                all_fixed_rows.extend(fixed_rows)
    
    # 导出结果
    logger.info("导出结果...")
    
    # 1. 固定模板逐 episode 明细
    fixed_df = pd.DataFrame(all_fixed_rows)
    fixed_df.to_csv(output_dir / "fixed_high_episode_metrics.csv", index=False)
    
    # 2. learned baseline 逐 episode 明细
    learned_df = pd.DataFrame(all_learned_rows)
    learned_df.to_csv(output_dir / "learned_high_episode_metrics.csv", index=False)
    
    # 3. 固定模板按 checkpoint 聚合
    if not fixed_df.empty:
        fixed_by_ckpt = fixed_df.groupby(["checkpoint_name", "scene", "template_idx", "m", "theta"]).agg({
            "TPS": ["mean", "std"],
            "mean_latency": ["mean", "std"],
            "unsafe_rate": ["mean", "std"],
            "timeout_rate": ["mean", "std"],
            "policy_invalid_rate": ["mean", "std"],
            "structural_infeasible_rate": ["mean", "std"],
        }).reset_index()
        fixed_by_ckpt.columns = ["_".join(col).strip("_") for col in fixed_by_ckpt.columns.values]
        fixed_by_ckpt.to_csv(output_dir / "fixed_high_summary_by_ckpt.csv", index=False)
    
    # 4. 固定模板跨 seeds 聚合
    if not fixed_df.empty:
        fixed_across_seeds = fixed_df.groupby(["scene", "template_idx", "m", "theta"]).agg({
            "TPS": ["mean", "std"],
            "mean_latency": ["mean", "std"],
            "unsafe_rate": ["mean", "std"],
            "timeout_rate": ["mean", "std"],
            "policy_invalid_rate": ["mean", "std"],
            "structural_infeasible_rate": ["mean", "std"],
        }).reset_index()
        fixed_across_seeds.columns = ["_".join(col).strip("_") for col in fixed_across_seeds.columns.values]
        fixed_across_seeds.to_csv(output_dir / "fixed_high_summary_across_seeds.csv", index=False)
    
    # 5. learned baseline 聚合
    if not learned_df.empty:
        learned_summary = learned_df.groupby(["checkpoint_name", "scene"]).agg({
            "TPS": ["mean", "std"],
            "mean_latency": ["mean", "std"],
            "unsafe_rate": ["mean", "std"],
            "timeout_rate": ["mean", "std"],
            "policy_invalid_rate": ["mean", "std"],
            "structural_infeasible_rate": ["mean", "std"],
        }).reset_index()
        learned_summary.columns = ["_".join(col).strip("_") for col in learned_summary.columns.values]
        learned_summary.to_csv(output_dir / "learned_high_baseline_summary.csv", index=False)
    
    logger.info(f"评估完成，结果保存到: {output_dir}")
    
    # 生成元数据
    metadata = {
        "mode": mode,
        "n_episodes": n_episodes,
        "n_checkpoints": len(checkpoints_to_eval),
        "n_scenes": len(scenes_to_eval),
        "n_templates": len(templates_to_eval),
        "total_fixed_episodes": len(all_fixed_rows),
        "total_learned_episodes": len(all_learned_rows),
        "deterministic_eval": True,
        "high_policy_mode": "fixed_template",
        "seed_manifest_path": "eval_seed_manifest.json",
        "checkpoint_metadata": checkpoint_metadata,
    }
    write_json(output_dir / "metadata.json", metadata)


def main() -> None:
    parser = argparse.ArgumentParser(description="固定高层模板前沿评估")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["minimal", "smoke", "full"],
        default="minimal",
        help="评估模式: minimal (最小验收) / smoke (N=20) / full (N=64)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis/fixed_high_frontier",
        help="输出目录",
    )
    
    args = parser.parse_args()
    output_dir = ensure_dir(Path(args.output_dir))
    
    run_evaluation(mode=args.mode, output_dir=output_dir)


if __name__ == "__main__":
    main()
