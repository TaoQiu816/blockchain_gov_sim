"""Stage1.5 high-level imitation warm-start."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from gov_sim.experiments import build_model
from gov_sim.hierarchical.controller import LowLevelInferenceAdapter
from gov_sim.hierarchical.envs import HighLevelGovEnv
from gov_sim.hierarchical.spec import DEFAULT_HIGH_UPDATE_INTERVAL, HIGH_LEVEL_TEMPLATES, HierarchicalActionCodec
from gov_sim.utils.device import resolve_device
from gov_sim.utils.io import deep_update, write_json


def _scenario_only_training_config(base_config: dict[str, Any], scenario_name: str) -> dict[str, Any]:
    config = deepcopy(base_config)
    scenario_cfg = config.setdefault("scenario", {})
    training_mix_cfg = scenario_cfg.setdefault("training_mix", {})
    profile = deepcopy(training_mix_cfg.get("profiles", {}).get(scenario_name, {}))
    profile["weight"] = 1.0
    scenario_cfg["training_mix"] = {
        "enabled": True,
        "enabled_in_train": True,
        "profiles": {scenario_name: profile},
    }
    return config


def _agent_override_config(base_config: dict[str, Any], overrides: dict[str, Any], run_name: str, output_root: Path) -> dict[str, Any]:
    updated = deepcopy(base_config)
    updated["agent"] = deep_update(updated["agent"], overrides)
    updated["run_name"] = run_name
    updated["output_root"] = str(output_root)
    return updated


def _training_stage_config(config: dict[str, Any]) -> dict[str, Any]:
    updated = deepcopy(config)
    scenario_cfg = updated.setdefault("scenario", {})
    training_mix_cfg = scenario_cfg.setdefault("training_mix", {})
    training_mix_cfg["enabled"] = bool(training_mix_cfg.get("enabled_in_train", False))
    return updated


def _resolved_model_path(stage15_cfg: dict[str, Any], frontier_seed: int) -> Path:
    source_root = Path(str(stage15_cfg.get("source_run_root", "outputs/chapter4_final_theory_locked/hierarchical")))
    run_pattern = str(stage15_cfg.get("source_run_name_pattern", "chapter4_final_theory_locked_seed{seed}"))
    low_relpath = str(stage15_cfg.get("source_low_model_relpath", "stage3_high_refine/low_model.zip"))
    return source_root / run_pattern.format(seed=int(frontier_seed)) / low_relpath


def _safe_int(value: Any, default: int = -1) -> int:
    if value is None:
        return default
    if isinstance(value, float) and math.isnan(value):
        return default
    text = str(value).strip()
    if not text:
        return default
    return int(text)


def _parse_high_template_repr(template_repr: str, codec: HierarchicalActionCodec) -> int:
    text = str(template_repr).strip()
    if not text:
        raise ValueError("Empty high template repr.")
    m_text, theta_text = text.split("|", maxsplit=1)
    m = int(m_text)
    theta = round(float(theta_text), 2)
    for idx, action in enumerate(codec.high_actions):
        if int(action.m) == m and round(float(action.theta), 2) == theta:
            return int(idx)
    raise ValueError(f"Unknown high template repr: {template_repr}")


@dataclass
class HighImitationDataset:
    states: np.ndarray
    action_masks: np.ndarray
    soft_targets: np.ndarray
    reward_per_template: np.ndarray
    cost_per_template: np.ndarray
    targets: np.ndarray
    second_targets: np.ndarray
    scene_ids: np.ndarray
    phase_ids: np.ndarray
    frontier_seeds: np.ndarray
    episode_ids: np.ndarray
    chunk_ids: np.ndarray
    best_unsafe: np.ndarray
    best_timeout: np.ndarray
    best_latency: np.ndarray
    best_tps: np.ndarray
    best_structural_infeasible: np.ndarray
    feasible_template_counts: np.ndarray
    soft_target_entropy: np.ndarray
    records: list[dict[str, Any]]
    scene_vocab: tuple[str, ...]
    phase_vocab: tuple[str, ...]

    def as_tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            torch.as_tensor(self.states, dtype=torch.float32),
            torch.as_tensor(self.action_masks, dtype=torch.float32),
            torch.as_tensor(self.soft_targets, dtype=torch.float32),
            torch.as_tensor(self.targets, dtype=torch.long),
            torch.as_tensor(self.scene_ids, dtype=torch.long),
            torch.as_tensor(self.phase_ids, dtype=torch.long),
        )

    def summary(self) -> dict[str, Any]:
        target_counter = Counter(int(target) for target in self.targets.tolist())
        scene_counter = Counter(self.scene_vocab[int(idx)] for idx in self.scene_ids.tolist())
        phase_counter = Counter(self.phase_vocab[int(idx)] for idx in self.phase_ids.tolist())
        return {
            "num_samples": int(self.states.shape[0]),
            "state_dim": int(self.states.shape[1]),
            "num_templates": int(len(HIGH_LEVEL_TEMPLATES)),
            "mean_legal_template_count": float(np.mean(self.action_masks.sum(axis=1))) if self.action_masks.size else 0.0,
            "mean_feasible_template_count": float(np.mean(self.feasible_template_counts)) if self.feasible_template_counts.size else 0.0,
            "mean_soft_target_entropy": float(np.mean(self.soft_target_entropy)) if self.soft_target_entropy.size else 0.0,
            "mean_best_unsafe": float(np.mean(self.best_unsafe)) if self.best_unsafe.size else 0.0,
            "mean_best_timeout": float(np.mean(self.best_timeout)) if self.best_timeout.size else 0.0,
            "mean_best_latency": float(np.mean(self.best_latency)) if self.best_latency.size else 0.0,
            "mean_best_tps": float(np.mean(self.best_tps)) if self.best_tps.size else 0.0,
            "scene_counts": {scene: int(scene_counter[scene]) for scene in self.scene_vocab},
            "phase_counts": {phase: int(phase_counter[phase]) for phase in self.phase_vocab},
            "target_template_counts": {
                f"{HIGH_LEVEL_TEMPLATES[idx][0]}|{HIGH_LEVEL_TEMPLATES[idx][1]:.2f}": int(count)
                for idx, count in sorted(target_counter.items())
            },
        }


def _template_label_from_idx(template_idx: int) -> str:
    m, theta = HIGH_LEVEL_TEMPLATES[int(template_idx)]
    return f"{m}|{theta:.2f}"


def _distribution_rows(records_df: pd.DataFrame) -> list[dict[str, Any]]:
    if records_df.empty:
        return []
    rows: list[dict[str, Any]] = []
    grouped = (
        records_df.groupby(["scene", "phase", "oracle_best_template"], sort=True)
        .size()
        .reset_index(name="count")
        .sort_values(["scene", "phase", "oracle_best_template"])
    )
    for row in grouped.itertuples(index=False):
        rows.append(
            {
                "scene": str(row.scene),
                "phase": str(row.phase),
                "template": str(row.oracle_best_template),
                "count": int(row.count),
            }
        )
    return rows


def _template_count_map(targets: np.ndarray) -> dict[str, int]:
    counter = Counter(int(target) for target in targets.tolist())
    return {
        _template_label_from_idx(template_idx): int(counter.get(template_idx, 0))
        for template_idx in range(len(HIGH_LEVEL_TEMPLATES))
    }


def _ranking_key(metrics: dict[str, Any]) -> tuple[float, float, float, float, float]:
    return (
        float(metrics.get("structural_infeasible_rate", 0.0)),
        float(metrics.get("unsafe_rate", 0.0)),
        float(metrics.get("timeout_rate", 0.0)),
        float(metrics.get("mean_latency", 0.0)),
        -float(metrics.get("TPS", 0.0)),
    )


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
        "normalized_cost": float(info.get("high_chunk_normalized_cost", info.get("cost", 0.0))),
        "terminated": int(bool(terminated)),
        "truncated": int(bool(truncated)),
        "info": info,
        "next_env": candidate_env,
    }


def _stable_softmax(logits: np.ndarray) -> np.ndarray:
    if logits.size == 0:
        return np.zeros((0,), dtype=np.float32)
    shifted = logits - float(np.max(logits))
    exp_values = np.exp(shifted)
    denom = float(np.sum(exp_values))
    if not math.isfinite(denom) or denom <= 0.0:
        return np.full(logits.shape, 1.0 / max(logits.size, 1), dtype=np.float32)
    return (exp_values / denom).astype(np.float32)


def _soft_target_distribution(
    reward_per_template: np.ndarray,
    cost_per_template: np.ndarray,
    legal_mask: np.ndarray,
    *,
    cost_limit: float,
    tau_reward: float,
    tau_cost: float,
) -> tuple[np.ndarray, int]:
    legal_indices = np.flatnonzero(legal_mask.astype(bool))
    probs = np.zeros_like(reward_per_template, dtype=np.float32)
    if legal_indices.size == 0:
        return probs, 0

    legal_costs = cost_per_template[legal_indices]
    feasible_mask = np.isfinite(legal_costs) & (legal_costs <= float(cost_limit))
    feasible_count = int(np.sum(feasible_mask))
    if feasible_count > 0:
        legal_rewards = reward_per_template[legal_indices][feasible_mask]
        legal_probs = _stable_softmax(legal_rewards.astype(np.float64) / max(float(tau_reward), 1.0e-8))
        probs[legal_indices[feasible_mask]] = legal_probs
        return probs, feasible_count

    legal_cost_logits = -cost_per_template[legal_indices].astype(np.float64) / max(float(tau_cost), 1.0e-8)
    legal_probs = _stable_softmax(legal_cost_logits)
    probs[legal_indices] = legal_probs
    return probs, 0


def _balanced_stage15_indices(
    records_df: pd.DataFrame,
    stage15_cfg: dict[str, Any],
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    balancing_cfg = dict(stage15_cfg.get("balancing", {}))
    if not bool(balancing_cfg.get("enabled", False)):
        return np.arange(len(records_df), dtype=np.int64), {}

    strategy = str(balancing_cfg.get("strategy", "scene_phase_template_uniform")).strip().lower()
    if strategy != "scene_phase_template_uniform":
        raise ValueError(f"Unsupported stage15 balancing strategy: {strategy}")

    rng = np.random.default_rng(int(balancing_cfg.get("seed", seed)))
    sampled_indices: list[np.ndarray] = []
    stratum_rows: list[dict[str, Any]] = []

    for (scene, phase), group_df in records_df.groupby(["scene", "phase"], sort=True):
        template_groups = []
        for template, template_df in group_df.groupby("oracle_best_template", sort=True):
            template_groups.append((str(template), template_df.index.to_numpy(dtype=np.int64)))
        if not template_groups:
            continue

        active_templates = len(template_groups)
        total_count = int(len(group_df))
        base_target = total_count // active_templates
        remainder = total_count % active_templates
        extras = {
            template: int(rank < remainder)
            for rank, (template, _) in enumerate(
                sorted(template_groups, key=lambda item: (len(item[1]), item[0]))
            )
        }

        for template, template_indices in template_groups:
            target_count = int(base_target + extras.get(template, 0))
            original_count = int(len(template_indices))
            replace = bool(original_count < target_count)
            chosen = (
                rng.choice(template_indices, size=target_count, replace=replace)
                if target_count > 0
                else np.empty((0,), dtype=np.int64)
            )
            sampled_indices.append(np.asarray(chosen, dtype=np.int64))
            stratum_rows.append(
                {
                    "scene": str(scene),
                    "phase": str(phase),
                    "template": str(template),
                    "original_count": original_count,
                    "balanced_count": target_count,
                    "delta": int(target_count - original_count),
                    "sampled_with_replacement": int(replace),
                }
            )

    if not sampled_indices:
        raise RuntimeError("Stage1.5 balancing produced an empty dataset.")

    merged = np.concatenate(sampled_indices, axis=0)
    shuffled = rng.permutation(merged)
    summary = {
        "enabled": True,
        "strategy": strategy,
        "seed": int(balancing_cfg.get("seed", seed)),
        "original_num_samples": int(len(records_df)),
        "balanced_num_samples": int(len(shuffled)),
        "strata": stratum_rows,
    }
    return shuffled.astype(np.int64), summary


def _subset_high_imitation_dataset(dataset: HighImitationDataset, indices: np.ndarray) -> HighImitationDataset:
    selected_records: list[dict[str, Any]] = []
    for sample_copy_idx, index in enumerate(indices.tolist()):
        record = dict(dataset.records[int(index)])
        record["sample_copy_idx"] = int(sample_copy_idx)
        record["source_record_index"] = int(index)
        selected_records.append(record)
    return HighImitationDataset(
        states=dataset.states[indices].copy(),
        action_masks=dataset.action_masks[indices].copy(),
        soft_targets=dataset.soft_targets[indices].copy(),
        reward_per_template=dataset.reward_per_template[indices].copy(),
        cost_per_template=dataset.cost_per_template[indices].copy(),
        targets=dataset.targets[indices].copy(),
        second_targets=dataset.second_targets[indices].copy(),
        scene_ids=dataset.scene_ids[indices].copy(),
        phase_ids=dataset.phase_ids[indices].copy(),
        frontier_seeds=dataset.frontier_seeds[indices].copy(),
        episode_ids=dataset.episode_ids[indices].copy(),
        chunk_ids=dataset.chunk_ids[indices].copy(),
        best_unsafe=dataset.best_unsafe[indices].copy(),
        best_timeout=dataset.best_timeout[indices].copy(),
        best_latency=dataset.best_latency[indices].copy(),
        best_tps=dataset.best_tps[indices].copy(),
        best_structural_infeasible=dataset.best_structural_infeasible[indices].copy(),
        feasible_template_counts=dataset.feasible_template_counts[indices].copy(),
        soft_target_entropy=dataset.soft_target_entropy[indices].copy(),
        records=selected_records,
        scene_vocab=dataset.scene_vocab,
        phase_vocab=dataset.phase_vocab,
    )


def build_stage15_high_imitation_dataset(
    base_config: dict[str, Any],
    stage_cfg: dict[str, Any],
    output_dir: Path,
) -> tuple[HighImitationDataset, dict[str, Any]]:
    hier_cfg = dict(stage_cfg.get("hierarchical", {}))
    stage15_cfg = dict(hier_cfg.get("stage15", {}))
    frontier_csv = Path(str(stage15_cfg.get("frontier_csv", "")))
    if not frontier_csv.is_absolute():
        frontier_csv = (Path.cwd() / frontier_csv).resolve()
    if not frontier_csv.exists():
        raise FileNotFoundError(f"Stage1.5 frontier csv does not exist: {frontier_csv}")

    frontier_df = pd.read_csv(frontier_csv)
    if frontier_df.empty:
        raise RuntimeError("Stage1.5 frontier csv is empty.")
    source_seeds = {int(seed) for seed in stage15_cfg.get("source_seeds", sorted(frontier_df["seed"].unique().tolist()))}
    frontier_df = frontier_df[frontier_df["seed"].astype(int).isin(source_seeds)].copy()
    frontier_df = frontier_df.sort_values(["scene", "seed", "episode_id", "chunk_id"]).reset_index(drop=True)

    codec = HierarchicalActionCodec()
    update_interval = int(hier_cfg.get("update_interval", DEFAULT_HIGH_UPDATE_INTERVAL))
    scene_vocab = tuple(sorted(str(scene) for scene in frontier_df["scene"].astype(str).unique().tolist()))
    phase_vocab = tuple(sorted(str(phase) for phase in frontier_df["phase"].astype(str).unique().tolist()))
    scene_to_id = {scene: idx for idx, scene in enumerate(scene_vocab)}
    phase_to_id = {phase: idx for idx, phase in enumerate(phase_vocab)}

    low_model_cache: dict[int, Any] = {}
    states: list[np.ndarray] = []
    action_masks: list[np.ndarray] = []
    soft_targets: list[np.ndarray] = []
    reward_per_template: list[np.ndarray] = []
    cost_per_template: list[np.ndarray] = []
    targets: list[int] = []
    second_targets: list[int] = []
    scene_ids: list[int] = []
    phase_ids: list[int] = []
    frontier_seeds: list[int] = []
    episode_ids: list[int] = []
    chunk_ids: list[int] = []
    best_unsafe: list[float] = []
    best_timeout: list[float] = []
    best_latency: list[float] = []
    best_tps: list[float] = []
    best_structural_infeasible: list[float] = []
    feasible_template_counts: list[int] = []
    soft_target_entropy: list[float] = []
    records: list[dict[str, Any]] = []
    high_agent_cfg = dict(hier_cfg.get("high_agent", {}))
    cost_limit = float(stage15_cfg.get("cost_limit", high_agent_cfg.get("cost_limit", 0.10)))
    tau_reward = float(stage15_cfg.get("soft_target_tau_reward", 50.0))
    tau_cost = float(stage15_cfg.get("soft_target_tau_cost", 0.02))

    for (scene, frontier_seed, episode_id), chunk_df in frontier_df.groupby(["scene", "seed", "episode_id"], sort=True):
        frontier_seed = int(frontier_seed)
        if frontier_seed not in low_model_cache:
            low_model_path = _resolved_model_path(stage15_cfg=stage15_cfg, frontier_seed=frontier_seed)
            if not low_model_path.is_absolute():
                low_model_path = (Path.cwd() / low_model_path).resolve()
            if not low_model_path.exists():
                raise FileNotFoundError(f"Stage1.5 source low model does not exist: {low_model_path}")
            from gov_sim.hierarchical.oracle_pretrain import OracleGuidedLowPolicy

            low_model_cache[frontier_seed] = OracleGuidedLowPolicy.load(low_model_path, device=resolve_device("cpu"))
        scenario_config = _scenario_only_training_config(base_config=base_config, scenario_name=str(scene))
        env = HighLevelGovEnv(
            config=scenario_config,
            low_policy=LowLevelInferenceAdapter(model=low_model_cache[frontier_seed], deterministic=True),
            update_interval=update_interval,
        )
        episode_seed = int(chunk_df["episode_seed"].iloc[0])
        obs, _ = env.reset(seed=episode_seed)
        for _, row in chunk_df.sort_values("chunk_id").iterrows():
            current_chunk_id = int(row["chunk_id"])
            best_template = str(row["best_template"]).strip()
            best_idx = _parse_high_template_repr(best_template, codec=codec)
            second_best_text = str(row["second_template"]).strip()
            second_best_idx = _safe_int(
                _parse_high_template_repr(second_best_text, codec=codec) if second_best_text and second_best_text.lower() != "nan" else None
            )
            state = np.asarray(obs["state"], dtype=np.float32).copy()
            mask = np.asarray(obs["action_mask"], dtype=np.int8).copy()
            if mask.shape[0] != codec.high_dim:
                raise RuntimeError(f"Unexpected high mask shape in stage1.5 dataset build: {mask.shape}")
            if int(mask[best_idx]) != 1:
                raise RuntimeError(
                    f"Stage1.5 oracle target is illegal under replayed high-state: "
                    f"scene={scene}, seed={frontier_seed}, episode={episode_id}, chunk={current_chunk_id}, template={best_template}"
                )
            legal_indices = np.flatnonzero(mask.astype(bool))
            candidate_rewards = np.full(codec.high_dim, np.nan, dtype=np.float32)
            candidate_costs = np.full(codec.high_dim, np.nan, dtype=np.float32)
            candidate_records = [
                _chunk_eval_record(
                    env=env,
                    template_idx=int(idx),
                    template_label=f"{HIGH_LEVEL_TEMPLATES[int(idx)][0]}|{HIGH_LEVEL_TEMPLATES[int(idx)][1]:.2f}",
                )
                for idx in legal_indices.tolist()
            ]
            candidate_by_idx = {int(record["template_idx"]): record for record in candidate_records}
            for candidate in candidate_records:
                template_idx = int(candidate["template_idx"])
                candidate_rewards[template_idx] = float(candidate["reward"])
                candidate_costs[template_idx] = float(candidate["normalized_cost"])
            soft_target, feasible_count = _soft_target_distribution(
                reward_per_template=candidate_rewards,
                cost_per_template=candidate_costs,
                legal_mask=mask,
                cost_limit=cost_limit,
                tau_reward=tau_reward,
                tau_cost=tau_cost,
            )
            target_entropy = float(
                -np.sum(soft_target[soft_target > 0.0] * np.log(np.clip(soft_target[soft_target > 0.0], 1.0e-8, 1.0)))
            )
            states.append(state)
            action_masks.append(mask)
            soft_targets.append(soft_target)
            reward_per_template.append(candidate_rewards)
            cost_per_template.append(candidate_costs)
            targets.append(int(best_idx))
            second_targets.append(int(second_best_idx))
            scene_ids.append(int(scene_to_id[str(scene)]))
            phase_ids.append(int(phase_to_id[str(row['phase'])]))
            frontier_seeds.append(int(frontier_seed))
            episode_ids.append(int(episode_id))
            chunk_ids.append(int(current_chunk_id))
            best_unsafe.append(float(row["best_unsafe"]))
            best_timeout.append(float(row["best_timeout"]))
            best_latency.append(float(row["best_latency"]))
            best_tps.append(float(row["best_TPS"]))
            best_structural_infeasible.append(float(row.get("structural_infeasible", 0.0)))
            feasible_template_counts.append(int(feasible_count))
            soft_target_entropy.append(target_entropy)
            records.append(
                {
                    "scene": str(scene),
                    "frontier_seed": int(frontier_seed),
                    "episode_id": int(episode_id),
                    "episode_seed": int(episode_seed),
                    "chunk_id": int(current_chunk_id),
                    "phase": str(row["phase"]),
                    "oracle_best_template": best_template,
                    "oracle_best_template_idx": int(best_idx),
                    "oracle_second_best_template": str(row["second_template"]).strip(),
                    "oracle_second_best_template_idx": int(second_best_idx),
                    "legal_template_count": int(mask.sum()),
                    "best_unsafe": float(row["best_unsafe"]),
                    "best_timeout": float(row["best_timeout"]),
                    "best_latency": float(row["best_latency"]),
                    "best_TPS": float(row["best_TPS"]),
                    "best_structural_infeasible": float(row.get("structural_infeasible", 0.0)),
                    "feasible_template_count": int(feasible_count),
                    "soft_target_entropy": float(target_entropy),
                    "soft_target_top_template": _template_label_from_idx(int(np.argmax(soft_target))),
                }
            )
            next_record = candidate_by_idx.get(int(best_idx))
            if next_record is None:
                raise RuntimeError(
                    f"Stage1.5 replay missing candidate record for oracle template {best_template} "
                    f"at scene={scene}, seed={frontier_seed}, episode={episode_id}, chunk={current_chunk_id}"
                )
            env = next_record["next_env"]
            terminated = bool(next_record["terminated"])
            truncated = bool(next_record["truncated"])
            if terminated or truncated:
                break

    if not states:
        raise RuntimeError("Stage1.5 imitation dataset is empty.")

    dataset = HighImitationDataset(
        states=np.asarray(states, dtype=np.float32),
        action_masks=np.asarray(action_masks, dtype=np.int8),
        soft_targets=np.asarray(soft_targets, dtype=np.float32),
        reward_per_template=np.asarray(reward_per_template, dtype=np.float32),
        cost_per_template=np.asarray(cost_per_template, dtype=np.float32),
        targets=np.asarray(targets, dtype=np.int64),
        second_targets=np.asarray(second_targets, dtype=np.int64),
        scene_ids=np.asarray(scene_ids, dtype=np.int64),
        phase_ids=np.asarray(phase_ids, dtype=np.int64),
        frontier_seeds=np.asarray(frontier_seeds, dtype=np.int64),
        episode_ids=np.asarray(episode_ids, dtype=np.int64),
        chunk_ids=np.asarray(chunk_ids, dtype=np.int64),
        best_unsafe=np.asarray(best_unsafe, dtype=np.float32),
        best_timeout=np.asarray(best_timeout, dtype=np.float32),
        best_latency=np.asarray(best_latency, dtype=np.float32),
        best_tps=np.asarray(best_tps, dtype=np.float32),
        best_structural_infeasible=np.asarray(best_structural_infeasible, dtype=np.float32),
        feasible_template_counts=np.asarray(feasible_template_counts, dtype=np.int64),
        soft_target_entropy=np.asarray(soft_target_entropy, dtype=np.float32),
        records=records,
        scene_vocab=scene_vocab,
        phase_vocab=phase_vocab,
    )
    original_records_df = pd.DataFrame(dataset.records)
    original_summary = dataset.summary()
    balance_indices, balance_summary = _balanced_stage15_indices(
        records_df=original_records_df,
        stage15_cfg=stage15_cfg,
        seed=int(base_config.get("seed", 0)),
    )
    dataset = _subset_high_imitation_dataset(dataset=dataset, indices=balance_indices)
    balanced_records_df = pd.DataFrame(dataset.records)

    np.savez_compressed(
        output_dir / "high_imitation_dataset.npz",
        states=dataset.states,
        action_masks=dataset.action_masks,
        soft_targets=dataset.soft_targets,
        reward_per_template=dataset.reward_per_template,
        cost_per_template=dataset.cost_per_template,
        targets=dataset.targets,
        second_targets=dataset.second_targets,
        scene_ids=dataset.scene_ids,
        phase_ids=dataset.phase_ids,
        frontier_seeds=dataset.frontier_seeds,
        episode_ids=dataset.episode_ids,
        chunk_ids=dataset.chunk_ids,
        best_unsafe=dataset.best_unsafe,
        best_timeout=dataset.best_timeout,
        best_latency=dataset.best_latency,
        best_tps=dataset.best_tps,
        best_structural_infeasible=dataset.best_structural_infeasible,
        feasible_template_counts=dataset.feasible_template_counts,
        soft_target_entropy=dataset.soft_target_entropy,
        scene_vocab=np.asarray(dataset.scene_vocab),
        phase_vocab=np.asarray(dataset.phase_vocab),
    )
    original_records_df.to_csv(output_dir / "high_imitation_dataset_records_original.csv", index=False)
    balanced_records_df.to_csv(output_dir / "high_imitation_dataset_records.csv", index=False)
    pd.DataFrame(_distribution_rows(original_records_df)).to_csv(output_dir / "label_distribution_before.csv", index=False)
    pd.DataFrame(_distribution_rows(balanced_records_df)).to_csv(output_dir / "label_distribution_after.csv", index=False)
    dataset_summary = dataset.summary()
    dataset_summary.update(
        {
            "frontier_csv": str(frontier_csv),
            "dataset_path": str(output_dir / "high_imitation_dataset.npz"),
            "record_path": str(output_dir / "high_imitation_dataset_records.csv"),
            "original_record_path": str(output_dir / "high_imitation_dataset_records_original.csv"),
            "label_distribution_before_path": str(output_dir / "label_distribution_before.csv"),
            "label_distribution_after_path": str(output_dir / "label_distribution_after.csv"),
            "source_seeds": sorted(int(seed) for seed in source_seeds),
            "cost_limit": cost_limit,
            "soft_target_tau_reward": tau_reward,
            "soft_target_tau_cost": tau_cost,
            "balancing": balance_summary,
            "original_target_template_counts": original_summary["target_template_counts"],
            "balanced_target_template_counts": _template_count_map(dataset.targets),
            "original_scene_phase_template_counts": _distribution_rows(original_records_df),
            "balanced_scene_phase_template_counts": _distribution_rows(balanced_records_df),
        }
    )
    write_json(output_dir / "high_imitation_dataset_summary.json", dataset_summary)
    return dataset, dataset_summary


def _policy_actor_parameters(policy: Any) -> list[torch.nn.Parameter]:
    modules = [
        getattr(policy, "pi_features_extractor", None),
        getattr(policy, "mlp_extractor", None),
        getattr(getattr(policy, "mlp_extractor", None), "policy_net", None),
        getattr(policy, "action_net", None),
    ]
    params: list[torch.nn.Parameter] = []
    seen_ids: set[int] = set()
    for module in modules:
        if module is None:
            continue
        for parameter in module.parameters():
            if not parameter.requires_grad or id(parameter) in seen_ids:
                continue
            seen_ids.add(id(parameter))
            params.append(parameter)
    return params


def _masked_distribution(model: Any, state_batch: torch.Tensor, mask_batch: torch.Tensor) -> Any:
    obs = {
        "state": state_batch.to(model.device).float(),
        "action_mask": mask_batch.to(model.device).float(),
    }
    return model.policy.get_distribution(obs, action_masks=mask_batch.to(model.device).bool())


def train_stage15_supervised_high_policy(
    base_config: dict[str, Any],
    stage_cfg: dict[str, Any],
    low_model: Any,
    stage_dir: Path,
) -> tuple[Any, dict[str, Any]]:
    dataset, dataset_summary = build_stage15_high_imitation_dataset(base_config=base_config, stage_cfg=stage_cfg, output_dir=stage_dir)
    states_tensor, masks_tensor, soft_target_tensor, target_tensor, scene_tensor, phase_tensor = dataset.as_tensors()
    train_dataset = TensorDataset(states_tensor, masks_tensor, soft_target_tensor, target_tensor, scene_tensor, phase_tensor)
    hier_cfg = dict(stage_cfg.get("hierarchical", {}))
    stage15_cfg = dict(hier_cfg.get("stage15", {}))
    supervised_cfg = dict(stage15_cfg.get("supervised", {}))
    high_agent_cfg = dict(hier_cfg.get("high_agent", {}))
    batch_size = int(supervised_cfg.get("batch_size", max(high_agent_cfg.get("batch_size", 64), 64)))
    example_budget = int(stage15_cfg.get("total_timesteps", 24000))
    learning_rate = float(supervised_cfg.get("learning_rate", high_agent_cfg.get("learning_rate", 2.5e-4)))
    update_interval = int(hier_cfg.get("update_interval", DEFAULT_HIGH_UPDATE_INTERVAL))

    model_cfg = _training_stage_config(
        _agent_override_config(base_config, overrides=high_agent_cfg, run_name="stage15_high_imitation", output_root=stage_dir.parent)
    )
    env = HighLevelGovEnv(config=model_cfg, low_policy=LowLevelInferenceAdapter(model=low_model, deterministic=True), update_interval=update_interval)
    model = build_model(config=model_cfg, env=env, use_lagrangian=True)
    actor_optimizer = torch.optim.Adam(_policy_actor_parameters(model.policy), lr=learning_rate)
    data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    epochs = max(1, int(math.ceil(example_budget / max(len(train_dataset), 1))))
    scene_names = dataset.scene_vocab
    phase_names = dataset.phase_vocab
    history: list[dict[str, Any]] = []
    scene_history: list[dict[str, Any]] = []
    phase_history: list[dict[str, Any]] = []
    seen_examples = 0

    for epoch in range(epochs):
        model.policy.train()
        epoch_loss = 0.0
        epoch_examples = 0
        for batch_states, batch_masks, batch_soft_targets, batch_targets, _, _ in data_loader:
            batch_states = batch_states.to(model.device)
            batch_masks = batch_masks.to(model.device)
            batch_soft_targets = batch_soft_targets.to(model.device)
            batch_targets = batch_targets.to(model.device)
            distribution = _masked_distribution(model=model, state_batch=batch_states, mask_batch=batch_masks)
            probs = torch.clamp(distribution.distribution.probs, min=1.0e-8)
            loss = -(batch_soft_targets * torch.log(probs)).sum(dim=1).mean()
            actor_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            actor_optimizer.step()
            batch_size_actual = int(batch_targets.shape[0])
            epoch_loss += float(loss.item()) * batch_size_actual
            epoch_examples += batch_size_actual
            seen_examples += batch_size_actual

        model.policy.eval()
        with torch.no_grad():
            distribution = _masked_distribution(model=model, state_batch=states_tensor, mask_batch=masks_tensor)
            probs = distribution.distribution.probs.detach().cpu()
            predictions = torch.argmax(probs, dim=1)
            soft_target_predictions = torch.argmax(soft_target_tensor, dim=1)
            top3 = torch.topk(probs, k=min(3, probs.shape[1]), dim=1).indices
            targets_cpu = target_tensor.cpu()
            scene_cpu = scene_tensor.cpu()
            phase_cpu = phase_tensor.cpu()
            top1_acc = float((predictions == targets_cpu).float().mean().item())
            top3_acc = float((top3 == targets_cpu.unsqueeze(1)).any(dim=1).float().mean().item())
            soft_top1_acc = float((predictions == soft_target_predictions).float().mean().item())
            history.append(
                {
                    "epoch": epoch + 1,
                    "seen_examples": int(seen_examples),
                    "imitation_loss": float(epoch_loss / max(epoch_examples, 1)),
                    "top1_accuracy": top1_acc,
                    "top3_accuracy": top3_acc,
                    "soft_target_top1_accuracy": soft_top1_acc,
                }
            )
            for scene_id, scene_name in enumerate(scene_names):
                mask = scene_cpu == int(scene_id)
                if int(mask.sum()) == 0:
                    continue
                scene_history.append(
                    {
                        "epoch": epoch + 1,
                        "scene": scene_name,
                        "top1_accuracy": float((predictions[mask] == targets_cpu[mask]).float().mean().item()),
                        "top3_accuracy": float(
                            (top3[mask] == targets_cpu[mask].unsqueeze(1)).any(dim=1).float().mean().item()
                        ),
                        "count": int(mask.sum().item()),
                    }
                )
            for phase_id, phase_name in enumerate(phase_names):
                mask = phase_cpu == int(phase_id)
                if int(mask.sum()) == 0:
                    continue
                phase_history.append(
                    {
                        "epoch": epoch + 1,
                        "phase": phase_name,
                        "top1_accuracy": float((predictions[mask] == targets_cpu[mask]).float().mean().item()),
                        "top3_accuracy": float(
                            (top3[mask] == targets_cpu[mask].unsqueeze(1)).any(dim=1).float().mean().item()
                        ),
                        "count": int(mask.sum().item()),
                    }
                )
        if seen_examples >= example_budget:
            break

    pd.DataFrame(history).to_csv(stage_dir / "train_log.csv", index=False)
    pd.DataFrame(scene_history).to_csv(stage_dir / "scene_accuracy.csv", index=False)
    pd.DataFrame(phase_history).to_csv(stage_dir / "phase_accuracy.csv", index=False)
    model.save(str(stage_dir / "model"))
    final_scene_accuracy = {
        entry["scene"]: float(entry["top1_accuracy"])
        for entry in scene_history
        if int(entry["epoch"]) == int(history[-1]["epoch"])
    }
    final_phase_accuracy = {
        entry["phase"]: float(entry["top1_accuracy"])
        for entry in phase_history
        if int(entry["epoch"]) == int(history[-1]["epoch"])
    }
    summary = {
        "stage": "stage15_high_imitation",
        "supervised_example_budget": int(example_budget),
        "epochs": int(len(history)),
        "batch_size": int(batch_size),
        "learning_rate": learning_rate,
        "device": str(getattr(model, "device", "unknown")),
        "dataset": dataset_summary,
        "loss": {
            "type": "masked_soft_target_kl",
            "label_smoothing": 0.0,
            "auxiliary_targets": 0,
        },
        "final_imitation_loss": float(history[-1]["imitation_loss"]),
        "final_top1_accuracy": float(history[-1]["top1_accuracy"]),
        "final_top3_accuracy": float(history[-1]["top3_accuracy"]),
        "final_soft_target_top1_accuracy": float(history[-1]["soft_target_top1_accuracy"]),
        "scene_top1_accuracy": final_scene_accuracy,
        "phase_top1_accuracy": final_phase_accuracy,
    }
    write_json(stage_dir / "train_audit.json", summary)
    return model, summary
