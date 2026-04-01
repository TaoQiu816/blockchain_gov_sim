"""固定高层模板前沿评估脚本 - 使用 BlockchainGovEnv。

该脚本使用 base env 进行评估，避免 LowLevelGovEnv 的兼容性问题。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import hashlib
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from gov_sim.env.gov_env import BlockchainGovEnv
from gov_sim.experiments import make_env
from gov_sim.hierarchical.fixed_template_selector import FixedTemplateSelector
from gov_sim.hierarchical.controller import LowLevelInferenceAdapter, build_low_level_mask
from gov_sim.hierarchical.spec import HIGH_LEVEL_TEMPLATES, HighLevelAction, HierarchicalActionCodec
from gov_sim.modules.metrics_tracker import MetricsTracker
from gov_sim.utils.io import ensure_dir, load_config, write_json

try:
    from gov_sim.hierarchical.oracle_pretrain import OracleGuidedLowPolicy
    from gov_sim.agent.masked_ppo_lagrangian import MaskablePPOLagrangian
except ImportError as exc:
    raise ImportError("需要 stable-baselines3 和 sb3-contrib") from exc

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 固定配置
CHECKPOINTS = [
    "outputs/formal_multiseed/hierarchical/formal_final_seed42/stage3_high_refine/low_model.zip",
    "outputs/formal_multiseed/hierarchical/formal_final_seed43/stage3_high_refine/low_model.zip",
    "outputs/formal_multiseed/hierarchical/formal_final_seed44/stage3_high_refine/low_model.zip",
    "outputs/formal_multiseed/hierarchical/formal_final_seed45/stage3_high_refine/low_model.zip",
    "outputs/formal_multiseed/hierarchical/formal_final_seed46/stage3_high_refine/low_model.zip",
]

SCENES = ["load_shock", "high_rtt_burst", "churn_burst", "malicious_burst"]
FIXED_TEMPLATES = HIGH_LEVEL_TEMPLATES  # 全部 12 个模板


def _compute_model_hash(model_path: Path) -> str:
    if not model_path.exists():
        return "not_found"
    with open(model_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()[:16]


def _generate_eval_seeds(scene: str, n_episodes: int, base_seed: int = 10000) -> list[int]:
    scene_offset = SCENES.index(scene) * 1000 if scene in SCENES else 0
    return [base_seed + scene_offset + i for i in range(n_episodes)]


def _build_scene_config(base_config: dict[str, Any], scene: str) -> dict[str, Any]:
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
    """使用 BlockchainGovEnv 评估固定模板。"""
    codec = HierarchicalActionCodec()
    high_action = HighLevelAction(m=template[0], theta=template[1])
    template_selector = FixedTemplateSelector(template=high_action, codec=codec)
    low_adapter = LowLevelInferenceAdapter(model=low_model, deterministic=True, codec=codec)
    
    rows: list[dict[str, Any]] = []
    
    for eval_seed in eval_seeds:
        # 创建环境
        env = make_env(config)
        env.set_invalid_action_mode("raise")
        
        obs, _ = env.reset(seed=eval_seed)
        template_selector.reset()
        low_adapter.reset()
        
        done = False
        truncated = False
        step_count = 0
        tracker = MetricsTracker()
        
        selected_b_list: list[int] = []
        selected_tau_list: list[int] = []
        episode_remap_count = 0
        
        while not (done or truncated):
            # 获取固定模板
            try:
                current_template = template_selector.template_for_step(env)
            except Exception as e:
                # 没有合法模板，跳过该 episode
                logger.warning(f"跳过 episode {eval_seed}: {e}")
                break
            
            if template_selector.was_remapped:
                episode_remap_count += 1
            
            # 低层选择动作
            low_mask = build_low_level_mask(env, codec, current_template)
            if low_mask.sum() == 0:
                # 没有合法动作，跳过
                break
            
            _, low_action, _ = low_adapter.select(env, current_template)
            
            # 执行动作
            flat_action_idx = codec.flat_index(high_action=current_template, low_action=low_action)
            obs, reward, done, truncated, info = env.step(flat_action_idx)
            
            selected_b_list.append(low_action.b)
            selected_tau_list.append(low_action.tau)
            
            tracker.update(info)
            
            template_selector.on_step_complete()
            step_count += 1
        
        remap_info = template_selector.get_remap_info()
        
        summary = tracker.summary()
        row = {
            "checkpoint_name": checkpoint_name,
            "scene": scene,
            "eval_seed": eval_seed,
            "template_idx": HIGH_LEVEL_TEMPLATES.index(template),
            "m": template[0],
            "theta": template[1],
            "steps": step_count,
            "TPS": summary.get("tps", 0.0),
            "mean_latency": summary.get("mean_latency", 0.0),
            "unsafe_rate": summary.get("unsafe_rate", 0.0),
            "timeout_rate": summary.get("timeout_failure_rate", 0.0),
            "policy_invalid_rate": summary.get("policy_invalid_rate", 0.0),
            "structural_infeasible_rate": summary.get("structural_infeasible_rate", 0.0),
            "queue_peak": summary.get("queue_peak", 0.0),
            "avg_selected_b": float(np.mean(selected_b_list)) if selected_b_list else 0.0,
            "avg_selected_tau": float(np.mean(selected_tau_list)) if selected_tau_list else 0.0,
            "episode_remap_count": episode_remap_count,
        }
        row.update(remap_info)
        rows.append(row)
        
        env.close()
    
    return rows


def run_evaluation(mode: str, output_dir: Path) -> None:
    if mode == "minimal":
        n_episodes = 5
    else:
        n_episodes = 20 if mode == "smoke" else 64
    
    base_config = load_config("configs/default.yaml", "configs/train_hierarchical_formal_final.yaml")
    
    logger.info(f"开始固定高层模板前沿评估 (mode={mode}, n_episodes={n_episodes})")
    
    # 生成 eval seed manifest
    eval_seed_manifest: dict[str, list[int]] = {}
    for scene in SCENES:
        eval_seed_manifest[scene] = _generate_eval_seeds(scene, n_episodes)
    
    write_json(output_dir / "eval_seed_manifest.json", eval_seed_manifest)
    logger.info(f"保存 eval seed manifest: {output_dir / 'eval_seed_manifest.json'}")
    
    all_fixed_rows: list[dict[str, Any]] = []
    checkpoint_metadata: list[dict[str, Any]] = []
    
    for ckpt_path in CHECKPOINTS:
        ckpt = Path(ckpt_path)
        ckpt_name = ckpt.stem
        
        logger.info(f"加载 checkpoint: {ckpt_name}")
        
        if not ckpt.exists():
            logger.warning(f"跳过不存在的 checkpoint: {ckpt}")
            continue
        
        low_model = OracleGuidedLowPolicy.load(ckpt)
        low_hash = _compute_model_hash(ckpt)
        
        checkpoint_metadata.append({
            "checkpoint_name": ckpt_name,
            "low_model_path": str(ckpt),
            "low_model_hash": low_hash,
        })
        
        for scene in SCENES:
            logger.info(f"  场景: {scene}")
            scene_config = _build_scene_config(base_config, scene)
            eval_seeds = eval_seed_manifest[scene]
            
            for template in tqdm(FIXED_TEMPLATES, desc=f"    固定模板"):
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
    
    fixed_df = pd.DataFrame(all_fixed_rows)
    fixed_df.to_csv(output_dir / "fixed_high_episode_metrics.csv", index=False)
    
    if not fixed_df.empty:
        fixed_across_seeds = fixed_df.groupby(["scene", "template_idx", "m", "theta"]).agg({
            "TPS": ["mean", "std"],
            "mean_latency": ["mean", "std"],
            "unsafe_rate": ["mean", "std"],
            "timeout_rate": ["mean", "std"],
        }).reset_index()
        fixed_across_seeds.columns = ["_".join(col).strip("_") for col in fixed_across_seeds.columns.values]
        fixed_across_seeds.to_csv(output_dir / "fixed_high_summary_across_seeds.csv", index=False)
    
    logger.info(f"评估完成，结果保存到: {output_dir}")
    
    metadata = {
        "mode": mode,
        "n_episodes": n_episodes,
        "n_checkpoints": len(CHECKPOINTS),
        "n_scenes": len(SCENES),
        "n_templates": len(FIXED_TEMPLATES),
        "total_fixed_episodes": len(all_fixed_rows),
        "deterministic_eval": True,
        "high_policy_mode": "fixed_template",
        "seed_manifest_path": "eval_seed_manifest.json",
        "checkpoint_metadata": checkpoint_metadata,
    }
    write_json(output_dir / "metadata.json", metadata)


def main() -> None:
    parser = argparse.ArgumentParser(description="固定高层模板前沿评估")
    parser.add_argument("--mode", type=str, choices=["minimal", "smoke", "full"], default="minimal")
    parser.add_argument("--output-dir", type=str, default="analysis/fixed_high_frontier")
    
    args = parser.parse_args()
    output_dir = ensure_dir(Path(args.output_dir))
    
    run_evaluation(mode=args.mode, output_dir=output_dir)


if __name__ == "__main__":
    main()
