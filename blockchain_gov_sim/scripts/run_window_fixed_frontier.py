"""窗口级 fixed frontier 分析。

按高层 chunk (K=10) 为单位，对每个 hard scene / train seed / eval episode：
1. 固定当前 seed 对应的最终 low actor
2. 在每个 chunk 起点复制环境状态
3. 对 12 个 nominal 高层模板逐个评估一个 chunk
4. 依据安全优先排序选出该 chunk 的最优 / 次优模板
5. 用该 chunk 最优模板推进主环境到下一个 chunk

输出：
- window_fixed_frontier.csv
- window_fixed_frontier_summary.csv
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path
import sys
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO

from gov_sim.agent.masked_ppo_lagrangian import MaskablePPOLagrangian
from gov_sim.hierarchical.controller import HierarchicalActionConstraintError, LowLevelInferenceAdapter
from gov_sim.hierarchical.envs import HighLevelGovEnv
from gov_sim.hierarchical.oracle_pretrain import OracleGuidedLowPolicy
from gov_sim.hierarchical.spec import HIGH_LEVEL_TEMPLATES, HierarchicalActionCodec
from gov_sim.utils.io import ensure_dir, load_config, write_json

OUTPUT_DIR = ensure_dir(PROJECT_ROOT / "outputs" / "chapter4_window_frontier")
SCENES = ["load_shock", "high_rtt_burst", "churn_burst", "malicious_burst"]
TRAIN_SEEDS = (42, 43, 44)


def _build_scene_config(base_config: dict[str, Any], scene: str, seed: int) -> dict[str, Any]:
    cfg = deepcopy(base_config)
    profile = deepcopy(cfg["scenario"]["training_mix"]["profiles"][scene])
    profile["weight"] = 1.0
    cfg["scenario"]["training_mix"] = {
        "enabled": True,
        "enabled_in_train": False,
        "profiles": {scene: profile},
    }
    cfg["seed"] = int(seed)
    return cfg


def _load_low_model(model_path: Path) -> Any:
    last_error: Exception | None = None
    for loader in (OracleGuidedLowPolicy, MaskablePPOLagrangian, MaskablePPO, PPO):
        try:
            return loader.load(model_path)
        except Exception as exc:  # pragma: no cover - 仅在模型格式兼容失败时触发
            last_error = exc
    raise RuntimeError(f"Unable to load low model at {model_path}: {last_error}")


def _ranking_key(metrics: dict[str, Any]) -> tuple[float, float, float, float, float]:
    return (
        float(metrics.get("structural_infeasible_rate", 0.0)),
        float(metrics.get("unsafe_rate", 0.0)),
        float(metrics.get("timeout_rate", 0.0)),
        float(metrics.get("mean_latency", 0.0)),
        -float(metrics.get("TPS", 0.0)),
    )


def _phase_label(chunk_id: int, total_chunks: int) -> str:
    if total_chunks <= 1:
        return "early"
    ratio = (chunk_id + 1) / max(total_chunks, 1)
    if ratio <= 1.0 / 3.0:
        return "early"
    if ratio <= 2.0 / 3.0:
        return "mid"
    return "late"


def _chunk_eval_record(
    env: HighLevelGovEnv,
    template_idx: int,
    template_label: str,
) -> dict[str, Any]:
    candidate_env = deepcopy(env)
    _, reward, terminated, truncated, info = candidate_env.step(template_idx)
    return {
        "template_idx": int(template_idx),
        "template": template_label,
        "reward": float(reward),
        "TPS": float(info.get("tps", 0.0)),
        "mean_latency": float(info.get("L_bar_e", 0.0)),
        "unsafe_rate": float(info.get("unsafe", 0.0)),
        "timeout_rate": float(info.get("timeout_failure", 0.0)),
        "structural_infeasible_rate": float(info.get("structural_infeasible", 0.0)),
        "terminated": int(bool(terminated)),
        "truncated": int(bool(truncated)),
        "info": info,
        "next_env": candidate_env,
    }


def _summarize(records: list[dict[str, Any]], phase_counts: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    overall_counter = Counter(record["best_template"] for record in records)
    total = max(len(records), 1)
    rows.append(
        {
            "scope": "overall",
            "group": "all",
            "best_template": overall_counter.most_common(1)[0][0] if overall_counter else "",
            "count": int(sum(overall_counter.values())),
            "ratio": float(overall_counter.get("5|0.60", 0) / total),
            "unique_best_templates": int(len(overall_counter)),
            "template_distribution": "; ".join(f"{tpl}:{cnt}" for tpl, cnt in overall_counter.most_common()),
        }
    )

    for scene in SCENES:
        scene_records = [record for record in records if record["scene"] == scene]
        counter = Counter(record["best_template"] for record in scene_records)
        denom = max(len(scene_records), 1)
        rows.append(
            {
                "scope": "scene",
                "group": scene,
                "best_template": counter.most_common(1)[0][0] if counter else "",
                "count": int(sum(counter.values())),
                "ratio": float(counter.get("5|0.60", 0) / denom),
                "unique_best_templates": int(len(counter)),
                "template_distribution": "; ".join(f"{tpl}:{cnt}" for tpl, cnt in counter.most_common()),
            }
        )

    for entry in phase_counts:
        rows.append(entry)

    return pd.DataFrame(rows)


def main() -> None:
    base_config = load_config(
        str(PROJECT_ROOT / "configs" / "default.yaml"),
        str(PROJECT_ROOT / "configs" / "train_hierarchical_formal_locked.yaml"),
        str(PROJECT_ROOT / "configs" / "default_theory_best_workpoint.yaml"),
        str(PROJECT_ROOT / "configs" / "override_skip_flat_comparator.yaml"),
    )
    codec = HierarchicalActionCodec()
    compare_episodes = int(base_config["hierarchical"]["compare"]["episodes"])
    update_interval = int(base_config["hierarchical"]["update_interval"])

    window_rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []

    for train_seed in TRAIN_SEEDS:
        run_dir = PROJECT_ROOT / "outputs" / "chapter4_final_theory_locked" / "hierarchical" / f"chapter4_final_theory_locked_seed{train_seed}"
        low_model = _load_low_model(run_dir / "stage3_high_refine" / "low_model.zip")
        low_policy = LowLevelInferenceAdapter(model=low_model, deterministic=True, codec=codec)

        for scene in SCENES:
            scene_config = _build_scene_config(base_config=base_config, scene=scene, seed=train_seed)

            for episode_id in range(compare_episodes):
                episode_seed = int(train_seed + episode_id)
                env = HighLevelGovEnv(config=scene_config, low_policy=deepcopy(low_policy), update_interval=update_interval)
                _, _ = env.reset(seed=episode_seed)

                episode_chunk_rows: list[dict[str, Any]] = []
                chunk_id = 0
                done = False

                while not done:
                    current_mask = env.action_masks().astype(bool)
                    legal_indices = [idx for idx, legal in enumerate(current_mask) if bool(legal)]
                    if not legal_indices:
                        break

                    candidate_records = [
                        _chunk_eval_record(env=env, template_idx=idx, template_label=f"{HIGH_LEVEL_TEMPLATES[idx][0]}|{HIGH_LEVEL_TEMPLATES[idx][1]:.2f}")
                        for idx in legal_indices
                    ]
                    candidate_records.sort(key=_ranking_key)
                    best = candidate_records[0]
                    second = candidate_records[1] if len(candidate_records) > 1 else None

                    env = best["next_env"]
                    best_info = best["info"]
                    done = bool(best["terminated"] or best["truncated"])

                    row = {
                        "scene": scene,
                        "seed": int(train_seed),
                        "episode_id": int(episode_id),
                        "episode_seed": int(episode_seed),
                        "chunk_id": int(chunk_id),
                        "legal_template_count": int(len(legal_indices)),
                        "best_template": str(best["template"]),
                        "best_template_metrics": (
                            f"unsafe={best['unsafe_rate']:.4f},timeout={best['timeout_rate']:.4f},"
                            f"latency={best['mean_latency']:.4f},TPS={best['TPS']:.4f}"
                        ),
                        "best_unsafe": float(best["unsafe_rate"]),
                        "best_timeout": float(best["timeout_rate"]),
                        "best_latency": float(best["mean_latency"]),
                        "best_TPS": float(best["TPS"]),
                        "second_template": "" if second is None else str(second["template"]),
                        "second_best_metrics": (
                            "" if second is None else
                            f"unsafe={second['unsafe_rate']:.4f},timeout={second['timeout_rate']:.4f},"
                            f"latency={second['mean_latency']:.4f},TPS={second['TPS']:.4f}"
                        ),
                        "second_unsafe": float(second["unsafe_rate"]) if second is not None else 0.0,
                        "second_timeout": float(second["timeout_rate"]) if second is not None else 0.0,
                        "second_latency": float(second["mean_latency"]) if second is not None else 0.0,
                        "second_TPS": float(second["TPS"]) if second is not None else 0.0,
                        "is_5_0_60_best": int(best["template"] == "5|0.60"),
                        "applied_executed_high_template": str(best_info.get("executed_high_template", "")),
                        "selected_high_template": str(best_info.get("selected_high_template", "")),
                        "selected_low_action": str(best_info.get("selected_low_action", "")),
                        "executed_low_action": str(best_info.get("executed_low_action", "")),
                        "eligible_size": float(best_info.get("eligible_size", 0.0)),
                        "h_LCB": float(best_info.get("h_LCB_e", best_info.get("h_LCB", 0.0))),
                        "committee_honest_ratio": float(best_info.get("committee_honest_ratio", best_info.get("h_e", 0.0))),
                        "structural_infeasible": float(best_info.get("structural_infeasible", 0.0)),
                    }
                    episode_chunk_rows.append(row)
                    chunk_id += 1

                total_chunks = len(episode_chunk_rows)
                for row in episode_chunk_rows:
                    row["phase"] = _phase_label(chunk_id=row["chunk_id"], total_chunks=total_chunks)
                window_rows.extend(episode_chunk_rows)

    window_df = pd.DataFrame(window_rows)
    window_df.to_csv(OUTPUT_DIR / "window_fixed_frontier.csv", index=False)

    for scene in SCENES:
        for phase in ("early", "mid", "late"):
            part = window_df[(window_df["scene"] == scene) & (window_df["phase"] == phase)]
            counter = Counter(str(value) for value in part["best_template"])
            denom = max(len(part), 1)
            phase_rows.append(
                {
                    "scope": "scene_phase",
                    "group": f"{scene}:{phase}",
                    "best_template": counter.most_common(1)[0][0] if counter else "",
                    "count": int(sum(counter.values())),
                    "ratio": float(counter.get("5|0.60", 0) / denom),
                    "unique_best_templates": int(len(counter)),
                    "template_distribution": "; ".join(f"{tpl}:{cnt}" for tpl, cnt in counter.most_common()),
                }
            )

    summary_df = _summarize(records=window_rows, phase_counts=phase_rows)
    summary_df.to_csv(OUTPUT_DIR / "window_fixed_frontier_summary.csv", index=False)

    write_json(
        OUTPUT_DIR / "window_fixed_frontier_meta.json",
        {
            "train_seeds": list(TRAIN_SEEDS),
            "scenes": list(SCENES),
            "compare_episodes": int(compare_episodes),
            "update_interval": int(update_interval),
            "ranking_rule": [
                "structural_infeasible_rate asc",
                "unsafe_rate asc",
                "timeout_rate asc",
                "mean_latency asc",
                "TPS desc",
            ],
        },
    )


if __name__ == "__main__":
    main()
