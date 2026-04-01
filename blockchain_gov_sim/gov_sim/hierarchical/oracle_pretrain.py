"""Stage1 oracle-guided 低层监督预训练。"""

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
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from gov_sim.env.gov_env import BlockchainGovEnv
from gov_sim.hierarchical.controller import build_low_level_mask
from gov_sim.hierarchical.observation import LOW_STATE_DIM, build_low_level_state
from gov_sim.hierarchical.spec import HIGH_LEVEL_TEMPLATES, LOW_LEVEL_ACTIONS, HighLevelAction, HierarchicalActionCodec
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


def _hidden_dims_from_agent(agent_cfg: dict[str, Any]) -> tuple[int, ...]:
    policy_kwargs = dict(agent_cfg.get("policy_kwargs") or {})
    net_arch = policy_kwargs.get("net_arch", {})
    hidden = net_arch.get("pi") or net_arch.get("vf") or [256, 256]
    return tuple(int(dim) for dim in hidden)


def _activation_from_agent(agent_cfg: dict[str, Any]) -> type[nn.Module]:
    policy_kwargs = dict(agent_cfg.get("policy_kwargs") or {})
    activation_name = str(policy_kwargs.get("activation_fn", "ReLU"))
    activation_map = {
        "ReLU": nn.ReLU,
        "Tanh": nn.Tanh,
        "ELU": nn.ELU,
    }
    if activation_name not in activation_map:
        raise ValueError(f"Unsupported activation_fn for oracle low pretrain: {activation_name}")
    return activation_map[activation_name]


def _normalize_phase(raw_phase: str) -> str:
    phase = str(raw_phase).strip().lower()
    if phase in {"pre_shock", "default"}:
        return "base"
    return phase or "base"


def _phase_targets_for_scenario(scenario_name: str) -> tuple[str, ...]:
    if scenario_name == "load_shock":
        return ("base", "shock")
    return ("base", "burst")


def _resolve_high_templates(dataset_cfg: dict[str, Any], codec: HierarchicalActionCodec) -> list[HighLevelAction]:
    coverage = str(dataset_cfg.get("high_template_coverage", "all")).strip().lower()
    if coverage == "all":
        return [HighLevelAction(m=m, theta=theta) for m, theta in HIGH_LEVEL_TEMPLATES]
    templates = dataset_cfg.get("high_templates", [])
    if not templates:
        raise ValueError("stage1.dataset.high_templates is required when high_template_coverage != 'all'")
    resolved: list[HighLevelAction] = []
    for entry in templates:
        m, theta = entry
        resolved.append(HighLevelAction(m=int(m), theta=float(theta)))
    return resolved


def _resolved_save_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.suffix:
        return resolved
    return resolved.with_suffix(".zip")


def _evaluate_low_action_reward(env: BlockchainGovEnv, codec: HierarchicalActionCodec, high_action: HighLevelAction, low_idx: int) -> float:
    trial_env = deepcopy(env)
    flat_idx = codec.flat_index(high_action=high_action, low_action=codec.decode_low(int(low_idx)))
    _, reward, _, _, _ = trial_env.step(flat_idx)
    return float(reward)


@dataclass
class OracleLowDataset:
    states: np.ndarray
    high_indices: np.ndarray
    targets: np.ndarray
    legal_counts: np.ndarray
    oracle_margins: np.ndarray
    records: list[dict[str, Any]]
    state_snapshot_count: int

    def as_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.as_tensor(self.states, dtype=torch.float32),
            torch.as_tensor(self.high_indices, dtype=torch.long),
            torch.as_tensor(self.targets, dtype=torch.long),
        )

    def summary(self) -> dict[str, Any]:
        target_counter = Counter(int(target) for target in self.targets.tolist())
        phase_counter = Counter((str(record["scenario_type"]), str(record["scenario_phase"])) for record in self.records)
        template_counter = Counter(str(record["high_template"]) for record in self.records)
        return {
            "num_samples": int(self.states.shape[0]),
            "num_state_snapshots": int(self.state_snapshot_count),
            "high_template_count": int(len(template_counter)),
            "scenario_phase_counts": {
                f"{scenario}|{phase}": int(count) for (scenario, phase), count in sorted(phase_counter.items())
            },
            "high_template_counts": {template: int(count) for template, count in sorted(template_counter.items())},
            "target_action_counts": {
                f"{LOW_LEVEL_ACTIONS[action][0]}|{LOW_LEVEL_ACTIONS[action][1]}": int(count)
                for action, count in sorted(target_counter.items())
            },
            "mean_legal_low_actions": float(np.mean(self.legal_counts)) if self.legal_counts.size else 0.0,
            "mean_oracle_margin": float(np.mean(self.oracle_margins)) if self.oracle_margins.size else 0.0,
        }


def build_stage1_oracle_dataset(
    base_config: dict[str, Any],
    stage_cfg: dict[str, Any],
    output_dir: Path,
) -> tuple[OracleLowDataset, dict[str, Any]]:
    stage1_cfg = dict(stage_cfg.get("hierarchical", {}).get("stage1", {}))
    dataset_cfg = dict(stage1_cfg.get("dataset", {}))
    scenario_names = [str(name) for name in dataset_cfg.get("scenario_names", ["load_shock", "high_rtt_burst", "churn_burst", "malicious_burst"])]
    max_states_per_phase = int(dataset_cfg.get("max_states_per_phase", 12))
    max_episodes_per_scenario = int(dataset_cfg.get("max_episodes_per_scenario", 8))
    codec = HierarchicalActionCodec()
    high_templates = _resolve_high_templates(dataset_cfg=dataset_cfg, codec=codec)
    rng = np.random.default_rng(int(stage_cfg.get("seed", 42)) + 701)

    records: list[dict[str, Any]] = []
    states: list[np.ndarray] = []
    high_indices: list[int] = []
    targets: list[int] = []
    legal_counts: list[int] = []
    oracle_margins: list[float] = []
    snapshot_total = 0

    for scenario_idx, scenario_name in enumerate(scenario_names):
        scenario_config = _scenario_only_training_config(base_config=base_config, scenario_name=scenario_name)
        env = BlockchainGovEnv(config=scenario_config)
        env.set_invalid_action_mode("raise")
        target_phases = _phase_targets_for_scenario(scenario_name)
        phase_counts: Counter[str] = Counter()
        scenario_seed_base = int(stage_cfg.get("seed", 42)) + (scenario_idx + 1) * 1000

        for episode in range(max_episodes_per_scenario):
            if all(phase_counts[phase] >= max_states_per_phase for phase in target_phases):
                break
            _, _ = env.reset(seed=scenario_seed_base + episode)
            done = False
            while not done:
                raw_phase = str(env.current_scenario.scenario_phase) if env.current_scenario is not None else "base"
                phase = _normalize_phase(raw_phase)
                if phase in target_phases and phase_counts[phase] < max_states_per_phase:
                    snapshot_total += 1
                    for high_index, high_action in enumerate(high_templates):
                        low_mask = build_low_level_mask(env=env, codec=codec, high_action=high_action)
                        legal_indices = np.flatnonzero(low_mask)
                        if legal_indices.size == 0:
                            continue
                        rewards = np.full(codec.low_dim, -np.inf, dtype=np.float32)
                        for low_idx in legal_indices:
                            rewards[int(low_idx)] = _evaluate_low_action_reward(
                                env=env,
                                codec=codec,
                                high_action=high_action,
                                low_idx=int(low_idx),
                            )
                        best_idx = int(np.argmax(rewards))
                        sorted_rewards = np.sort(rewards[np.isfinite(rewards)])
                        second_best = float(sorted_rewards[-2]) if sorted_rewards.size >= 2 else float(sorted_rewards[-1])
                        best_reward = float(rewards[best_idx])
                        states.append(build_low_level_state(env, high_action=high_action).astype(np.float32))
                        high_indices.append(int(high_index))
                        targets.append(best_idx)
                        legal_counts.append(int(legal_indices.size))
                        oracle_margins.append(float(best_reward - second_best))
                        records.append(
                            {
                                "scenario_type": scenario_name,
                                "scenario_phase": phase,
                                "high_template": f"{int(high_action.m)}|{float(high_action.theta):.2f}",
                                "target_action_idx": best_idx,
                                "target_action": f"{LOW_LEVEL_ACTIONS[best_idx][0]}|{LOW_LEVEL_ACTIONS[best_idx][1]}",
                                "oracle_reward": best_reward,
                                "oracle_margin": float(best_reward - second_best),
                                "num_legal_low_actions": int(legal_indices.size),
                            }
                        )
                    phase_counts[phase] += 1

                legal_flat = np.flatnonzero(env.current_mask.astype(bool))
                if legal_flat.size == 0:
                    break
                rollout_action = int(rng.choice(legal_flat))
                _, _, terminated, truncated, _ = env.step(rollout_action)
                done = bool(terminated or truncated)

    if not states:
        raise RuntimeError("Stage1 oracle dataset is empty.")

    dataset = OracleLowDataset(
        states=np.asarray(states, dtype=np.float32),
        high_indices=np.asarray(high_indices, dtype=np.int64),
        targets=np.asarray(targets, dtype=np.int64),
        legal_counts=np.asarray(legal_counts, dtype=np.int64),
        oracle_margins=np.asarray(oracle_margins, dtype=np.float32),
        records=records,
        state_snapshot_count=int(snapshot_total),
    )
    np.savez_compressed(
        output_dir / "oracle_dataset.npz",
        states=dataset.states,
        high_indices=dataset.high_indices,
        targets=dataset.targets,
        legal_counts=dataset.legal_counts,
        oracle_margins=dataset.oracle_margins,
    )
    dataset_summary = dataset.summary()
    dataset_summary.update(
        {
            "scenario_names": scenario_names,
            "target_phases": {name: list(_phase_targets_for_scenario(name)) for name in scenario_names},
            "high_templates": [f"{template.m}|{template.theta:.2f}" for template in high_templates],
            "dataset_path": str(output_dir / "oracle_dataset.npz"),
        }
    )
    pd.DataFrame(records).to_csv(output_dir / "oracle_dataset_records.csv", index=False)
    write_json(output_dir / "oracle_dataset_summary.json", dataset_summary)
    return dataset, dataset_summary


class OracleLowPolicyNetwork(nn.Module):
    """监督式 low actor，仅输出 20 维动作 logits。"""

    def __init__(
        self,
        state_dim: int,
        high_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...],
        activation_cls: type[nn.Module],
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        input_dim = state_dim + high_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, int(hidden_dim)))
            layers.append(activation_cls())
            input_dim = int(hidden_dim)
        self.policy_net = nn.Sequential(*layers)
        self.action_net = nn.Linear(input_dim, action_dim)
        self.high_dim = int(high_dim)

    def forward(self, state_tensor: torch.Tensor, high_indices: torch.Tensor) -> torch.Tensor:
        high_one_hot = F.one_hot(high_indices.long(), num_classes=self.high_dim).float()
        features = torch.cat([state_tensor.float(), high_one_hot], dim=1)
        if len(self.policy_net) > 0:
            features = self.policy_net(features)
        return self.action_net(features)


class OracleGuidedLowPolicy:
    """供 stage2/stage3 推理复用的监督 low model。"""

    model_type = "oracle_guided_low_policy"

    def __init__(
        self,
        *,
        learning_rate: float,
        hidden_dims: tuple[int, ...],
        activation_name: str,
        device: str,
        codec: HierarchicalActionCodec | None = None,
        encoding_templates: tuple[HighLevelAction, ...] | None = None,
    ) -> None:
        self.codec = codec or HierarchicalActionCodec()
        self.device = torch.device(device)
        activation_map = {
            "ReLU": nn.ReLU,
            "Tanh": nn.Tanh,
            "ELU": nn.ELU,
        }
        if activation_name not in activation_map:
            raise ValueError(f"Unsupported activation_name: {activation_name}")
        resolved_templates = tuple(encoding_templates or self.codec.high_actions)
        self.policy = OracleLowPolicyNetwork(
            state_dim=LOW_STATE_DIM,
            high_dim=len(resolved_templates),
            action_dim=self.codec.low_dim,
            hidden_dims=hidden_dims,
            activation_cls=activation_map[activation_name],
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=float(learning_rate))
        self.learning_rate = float(learning_rate)
        self.hidden_dims = tuple(int(dim) for dim in hidden_dims)
        self.activation_name = str(activation_name)
        self.encoding_templates = resolved_templates
        self._template_tensor = torch.as_tensor(
            [[float(action.m), float(action.theta)] for action in self.encoding_templates],
            dtype=torch.float32,
            device=self.device,
        )

    def _infer_high_indices(self, state_tensor: torch.Tensor) -> torch.Tensor:
        template_features = state_tensor[:, -2:].float()
        distances = (
            torch.abs(template_features[:, None, 0] - self._template_tensor[None, :, 0])
            + 10.0 * torch.abs(template_features[:, None, 1] - self._template_tensor[None, :, 1])
        )
        return torch.argmin(distances, dim=1)

    def predict_logits(self, obs: dict[str, np.ndarray] | dict[str, torch.Tensor]) -> torch.Tensor:
        state = obs["state"]
        if isinstance(state, np.ndarray):
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        else:
            state_tensor = state.to(self.device).float()
        if state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0)
        high_indices = self._infer_high_indices(state_tensor)
        self.policy.eval()
        with torch.no_grad():
            return self.policy(state_tensor, high_indices)

    def predict(
        self,
        obs: dict[str, np.ndarray] | dict[str, torch.Tensor],
        deterministic: bool = True,
        action_masks: np.ndarray | torch.Tensor | None = None,
    ) -> tuple[np.ndarray, None]:
        logits = self.predict_logits(obs)
        if action_masks is not None:
            mask_tensor = torch.as_tensor(action_masks, dtype=torch.bool, device=self.device)
            if mask_tensor.ndim == 1:
                mask_tensor = mask_tensor.unsqueeze(0)
            logits = logits.masked_fill(~mask_tensor, torch.finfo(logits.dtype).min)
        if deterministic:
            actions = torch.argmax(logits, dim=1)
        else:
            probs = torch.softmax(logits, dim=1)
            actions = torch.multinomial(probs, num_samples=1).squeeze(1)
        return actions.detach().cpu().numpy(), None

    def save(self, path: str | Path) -> None:
        save_path = _resolved_save_path(path)
        payload = {
            "model_type": self.model_type,
            "learning_rate": self.learning_rate,
            "hidden_dims": list(self.hidden_dims),
            "activation_name": self.activation_name,
            "encoding_templates": [(int(action.m), float(action.theta)) for action in self.encoding_templates],
            "state_dict": self.policy.state_dict(),
        }
        torch.save(payload, save_path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: str | None = None,
        codec: HierarchicalActionCodec | None = None,
    ) -> "OracleGuidedLowPolicy":
        load_path = _resolved_save_path(path)
        payload = torch.load(load_path, map_location=device or "cpu")
        encoding_templates = tuple(
            HighLevelAction(m=int(m), theta=float(theta))
            for m, theta in payload.get("encoding_templates", HIGH_LEVEL_TEMPLATES)
        )
        model = cls(
            learning_rate=float(payload["learning_rate"]),
            hidden_dims=tuple(int(dim) for dim in payload["hidden_dims"]),
            activation_name=str(payload["activation_name"]),
            device=str(device or "cpu"),
            codec=codec,
            encoding_templates=encoding_templates,
        )
        model.policy.load_state_dict(payload["state_dict"])
        model.policy.eval()
        return model


def train_stage1_supervised_low_policy(
    base_config: dict[str, Any],
    stage_cfg: dict[str, Any],
    stage_dir: Path,
) -> tuple[OracleGuidedLowPolicy, dict[str, Any]]:
    dataset, dataset_summary = build_stage1_oracle_dataset(base_config=base_config, stage_cfg=stage_cfg, output_dir=stage_dir)
    states_tensor, high_tensor, target_tensor = dataset.as_tensors()
    train_dataset = TensorDataset(states_tensor, high_tensor, target_tensor)
    stage1_cfg = dict(stage_cfg.get("hierarchical", {}).get("stage1", {}))
    supervised_cfg = dict(stage1_cfg.get("supervised", {}))
    agent_cfg = dict(stage_cfg.get("agent", {}))
    batch_size = int(supervised_cfg.get("batch_size", 128))
    example_budget = int(stage1_cfg.get("total_timesteps", 24000))
    model = OracleGuidedLowPolicy(
        learning_rate=float(agent_cfg.get("learning_rate", 2.5e-4)),
        hidden_dims=_hidden_dims_from_agent(agent_cfg),
        activation_name=str((agent_cfg.get("policy_kwargs") or {}).get("activation_fn", "ReLU")),
        device=resolve_device(agent_cfg.get("device")),
    )
    data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    epochs = max(1, int(math.ceil(example_budget / max(len(train_dataset), 1))))
    history: list[dict[str, Any]] = []
    seen_examples = 0

    for epoch in range(epochs):
        model.policy.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_examples = 0
        for batch_states, batch_high, batch_targets in data_loader:
            batch_states = batch_states.to(model.device)
            batch_high = batch_high.to(model.device)
            batch_targets = batch_targets.to(model.device)
            logits = model.policy(batch_states, batch_high)
            loss = F.cross_entropy(logits, batch_targets)
            model.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            model.optimizer.step()
            epoch_loss += float(loss.item()) * int(batch_targets.shape[0])
            epoch_correct += int((torch.argmax(logits, dim=1) == batch_targets).sum().item())
            epoch_examples += int(batch_targets.shape[0])
            seen_examples += int(batch_targets.shape[0])
        history.append(
            {
                "epoch": epoch + 1,
                "seen_examples": int(seen_examples),
                "train_loss": float(epoch_loss / max(epoch_examples, 1)),
                "train_accuracy": float(epoch_correct / max(epoch_examples, 1)),
            }
        )
        if seen_examples >= example_budget:
            break

    pd.DataFrame(history).to_csv(stage_dir / "train_log.csv", index=False)
    model.save(stage_dir / "model")
    summary = {
        "stage": "stage1_low_pretrain",
        "train_mode": "oracle_supervised",
        "supervised_example_budget": int(example_budget),
        "epochs": int(len(history)),
        "batch_size": int(batch_size),
        "device": str(model.device),
        "dataset": dataset_summary,
        "final_train_loss": float(history[-1]["train_loss"]),
        "final_train_accuracy": float(history[-1]["train_accuracy"]),
    }
    write_json(stage_dir / "train_audit.json", summary)
    return model, summary
