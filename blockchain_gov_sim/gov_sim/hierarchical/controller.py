"""分层治理器执行逻辑与推理适配器。"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from gov_sim.env.gov_env import BlockchainGovEnv
from gov_sim.env.action_codec import GovernanceAction
from gov_sim.hierarchical.observation import build_high_level_state, build_low_level_state
from gov_sim.hierarchical.spec import DEFAULT_HIGH_UPDATE_INTERVAL, HighLevelAction, HierarchicalActionCodec, LowLevelAction


def _predict_masked(model: Any, obs: dict[str, np.ndarray], mask: np.ndarray, deterministic: bool) -> int:
    action, _ = model.predict(obs, deterministic=deterministic, action_masks=mask)
    if isinstance(action, np.ndarray):
        return int(np.asarray(action).reshape(-1)[0])
    return int(action)


def _nearest_high_legal_index(
    codec: HierarchicalActionCodec,
    legal_indices: np.ndarray,
    requested: HighLevelAction | None,
) -> int:
    if requested is None:
        return int(legal_indices[0])
    scored = []
    for idx in legal_indices:
        candidate = codec.decode_high(int(idx))
        distance = abs(candidate.m - requested.m) + 10.0 * abs(candidate.theta - requested.theta)
        scored.append((distance, int(idx)))
    scored.sort(key=lambda item: (item[0], item[1]))
    return int(scored[0][1])


def _nearest_low_legal_index(
    codec: HierarchicalActionCodec,
    legal_indices: np.ndarray,
    requested: LowLevelAction | None,
) -> int:
    if requested is None:
        return int(legal_indices[0])
    scored = []
    for idx in legal_indices:
        candidate = codec.decode_low(int(idx))
        distance = abs(candidate.b - requested.b) / 64.0 + abs(candidate.tau - requested.tau) / 20.0
        scored.append((distance, int(idx)))
    scored.sort(key=lambda item: (item[0], item[1]))
    return int(scored[0][1])


def build_low_level_mask(env: BlockchainGovEnv, codec: HierarchicalActionCodec, high_action: HighLevelAction) -> np.ndarray:
    """把 240 维 flat mask 投影到最终 20 维低层动作空间。"""

    mask = np.zeros(codec.low_dim, dtype=np.int8)
    for idx, low_action in enumerate(codec.low_actions):
        flat_idx = codec.flat_index(high_action=high_action, low_action=low_action)
        mask[idx] = int(env.current_mask[flat_idx])
    if mask.sum() == 0:
        return mask
    return mask


def build_high_level_mask(env: BlockchainGovEnv, codec: HierarchicalActionCodec) -> np.ndarray:
    """高层 nominal 模板合法性：12 个 policy 模板中至少存在一个可执行的低层动作。"""

    mask = np.zeros(codec.high_dim, dtype=np.int8)
    for idx, high_action in enumerate(codec.high_actions):
        mask[idx] = int(build_low_level_mask(env=env, codec=codec, high_action=high_action).sum() > 0)
    return mask


def count_legal_high_templates(env: BlockchainGovEnv, codec: HierarchicalActionCodec) -> int:
    return int(build_high_level_mask(env=env, codec=codec).sum())


def summarize_low_level_usage(low_actions: list[LowLevelAction]) -> tuple[int, int]:
    """返回一段 high chunk 中最常见的 `(b,tau)`。"""

    if not low_actions:
        return 0, 0
    counter = Counter((action.b, action.tau) for action in low_actions)
    (b, tau), _ = counter.most_common(1)[0]
    return int(b), int(tau)


class HierarchicalActionConstraintError(RuntimeError):
    """当前状态下不存在合法的 hierarchical 动作。"""


def assert_hierarchical_action(codec: HierarchicalActionCodec, action: GovernanceAction) -> None:
    high_action, low_action = codec.split_governance_action(action)
    if high_action not in codec.high_actions:
        raise HierarchicalActionConstraintError(f"Unsupported high-level action: {high_action}")
    if low_action not in codec.low_actions:
        raise HierarchicalActionConstraintError(f"Unsupported low-level action: {low_action}")


def _resolve_high_action(
    env: BlockchainGovEnv,
    codec: HierarchicalActionCodec,
    requested_idx: int | None,
) -> tuple[int, HighLevelAction, np.ndarray, str, HighLevelAction | None, str]:
    mask = build_high_level_mask(env=env, codec=codec)
    legal_indices = np.flatnonzero(mask)
    if legal_indices.size == 0:
        raise HierarchicalActionConstraintError("No legal high-level template is available.")
    requested_action = None
    if requested_idx is not None and 0 <= int(requested_idx) < codec.high_dim:
        requested_action = codec.decode_high(int(requested_idx))
    if requested_idx is None:
        chosen_idx = int(legal_indices[0])
        source = "hierarchical_remap"
    else:
        chosen_idx = int(requested_idx)
        source = "selected"
        if chosen_idx < 0 or chosen_idx >= codec.high_dim or not mask[chosen_idx]:
            chosen_idx = _nearest_high_legal_index(codec=codec, legal_indices=legal_indices, requested=requested_action)
            source = "hierarchical_remap"
    chosen_action = codec.decode_high(chosen_idx)
    return chosen_idx, chosen_action, mask.astype(np.int8), source, requested_action, codec.high_action_repr(requested_action or chosen_action)


def _resolve_low_action(
    env: BlockchainGovEnv,
    codec: HierarchicalActionCodec,
    high_action: HighLevelAction,
    requested_idx: int | None,
) -> tuple[int, LowLevelAction, np.ndarray, str]:
    mask = build_low_level_mask(env=env, codec=codec, high_action=high_action)
    legal_indices = np.flatnonzero(mask)
    if legal_indices.size == 0:
        raise HierarchicalActionConstraintError(
            f"No legal low-level action is available for template {(high_action.m, high_action.theta)}."
        )
    if requested_idx is None:
        chosen_idx = int(legal_indices[0])
        source = "hierarchical_remap"
    else:
        requested_action = codec.decode_low(int(requested_idx)) if 0 <= int(requested_idx) < codec.low_dim else None
        chosen_idx = int(requested_idx)
        source = "selected"
        if chosen_idx < 0 or chosen_idx >= codec.low_dim or not mask[chosen_idx]:
            chosen_idx = _nearest_low_legal_index(codec=codec, legal_indices=legal_indices, requested=requested_action)
            source = "hierarchical_remap"
    return chosen_idx, codec.decode_low(chosen_idx), mask.astype(np.int8), source


def resolve_template(
    env: BlockchainGovEnv,
    codec: HierarchicalActionCodec,
    requested: HighLevelAction | None,
) -> tuple[HighLevelAction, np.ndarray, str, str]:
    requested_idx = None if requested is None or not codec.is_nominal_template(requested) else codec.encode_high(requested)
    _, high_action, mask, source, _, policy_high_template = _resolve_high_action(env=env, codec=codec, requested_idx=requested_idx)
    return high_action, mask, source, policy_high_template


def build_hierarchical_action_metadata(
    codec: HierarchicalActionCodec,
    selected_high_template: HighLevelAction | None,
    executed_high_template: HighLevelAction | None,
    selected_low_action: LowLevelAction | None,
    executed_low_action: LowLevelAction | None,
    num_legal_high_templates: int,
    action_source: str,
    policy_high_template: str | None = None,
) -> dict[str, int | str]:
    return {
        "policy_high_template": str(
            policy_high_template if policy_high_template is not None else codec.high_action_repr(selected_high_template)
        ),
        "selected_high_template": codec.high_action_repr(selected_high_template),
        "executed_high_template": codec.high_action_repr(executed_high_template),
        "selected_low_action": codec.low_action_repr(selected_low_action),
        "executed_low_action": codec.low_action_repr(executed_low_action),
        "num_legal_high_templates": int(num_legal_high_templates),
        "action_source": action_source,
    }


class HighLevelInferenceAdapter:
    """把高层策略模型包装成模板选择器。"""

    def __init__(self, model: Any, deterministic: bool = True, codec: HierarchicalActionCodec | None = None) -> None:
        self.model = model
        self.deterministic = deterministic
        self.codec = codec or HierarchicalActionCodec()
        self.last_action_source = "selected"
        self.last_requested_template: HighLevelAction | None = None
        self.last_resolved_template: HighLevelAction | None = None
        self.last_num_legal_high_templates = 0
        self.last_policy_high_template = ""

    def reset(self) -> None:
        self.last_action_source = "selected"
        self.last_requested_template = None
        self.last_resolved_template = None
        self.last_num_legal_high_templates = 0
        self.last_policy_high_template = ""
        return None

    def select(self, env: BlockchainGovEnv) -> tuple[int, HighLevelAction, np.ndarray]:
        mask = build_high_level_mask(env=env, codec=self.codec)
        self.last_num_legal_high_templates = int(mask.sum())
        requested_idx = None
        if mask.sum() > 0:
            obs = {"state": build_high_level_state(env), "action_mask": mask.astype(np.int8)}
            requested_idx = _predict_masked(self.model, obs=obs, mask=mask.astype(bool), deterministic=self.deterministic)
        action_idx, high_action, resolved_mask, source, requested_template, policy_high_template = _resolve_high_action(
            env=env,
            codec=self.codec,
            requested_idx=requested_idx,
        )
        self.last_action_source = source
        self.last_requested_template = requested_template
        self.last_resolved_template = high_action
        self.last_policy_high_template = policy_high_template
        return action_idx, high_action, resolved_mask


class LowLevelInferenceAdapter:
    """把低层策略模型包装成 `(b,tau)` 执行器。"""

    def __init__(self, model: Any, deterministic: bool = True, codec: HierarchicalActionCodec | None = None) -> None:
        self.model = model
        self.deterministic = deterministic
        self.codec = codec or HierarchicalActionCodec()
        self.last_action_source = "selected"
        self.last_requested_low_action: LowLevelAction | None = None
        self.last_resolved_low_action: LowLevelAction | None = None

    def reset(self) -> None:
        self.last_action_source = "selected"
        self.last_requested_low_action = None
        self.last_resolved_low_action = None
        return None

    def select(self, env: BlockchainGovEnv, high_action: HighLevelAction) -> tuple[int, LowLevelAction, np.ndarray]:
        mask = build_low_level_mask(env=env, codec=self.codec, high_action=high_action)
        if mask.sum() == 0:
            raise HierarchicalActionConstraintError(
                f"No legal low-level action is available for template {(high_action.m, high_action.theta)}."
            )
        obs = {"state": build_low_level_state(env, high_action=high_action), "action_mask": mask.astype(np.int8)}
        requested_idx = _predict_masked(self.model, obs=obs, mask=mask.astype(bool), deterministic=self.deterministic)
        self.last_requested_low_action = self.codec.decode_low(requested_idx) if 0 <= requested_idx < self.codec.low_dim else None
        action_idx, low_action, resolved_mask, source = _resolve_low_action(
            env=env,
            codec=self.codec,
            high_action=high_action,
            requested_idx=requested_idx,
        )
        self.last_action_source = source
        self.last_resolved_low_action = low_action
        return action_idx, low_action, resolved_mask


class EpisodeSampledTemplateSelector:
    """阶段 1 低层预训练：每个 episode 固定一个随机高层模板。"""

    def __init__(self, seed: int = 42, codec: HierarchicalActionCodec | None = None) -> None:
        self.codec = codec or HierarchicalActionCodec()
        self.rng = np.random.default_rng(seed)
        self.current_template = self.codec.high_actions[0]
        self.last_requested_template: HighLevelAction | None = None
        self.last_resolved_template: HighLevelAction | None = None
        self.last_num_legal_high_templates = 0
        self.last_policy_high_template = self.codec.high_action_repr(self.current_template)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        action_idx = int(self.rng.integers(0, self.codec.high_dim))
        self.current_template = self.codec.decode_high(action_idx)
        self.last_requested_template = self.current_template
        self.last_resolved_template = None
        self.last_num_legal_high_templates = 0
        self.last_policy_high_template = self.codec.high_action_repr(self.current_template)

    def template_for_step(self, env: BlockchainGovEnv) -> HighLevelAction:
        self.last_requested_template = self.current_template
        self.last_num_legal_high_templates = count_legal_high_templates(env=env, codec=self.codec)
        template, _, _, policy_high_template = resolve_template(env=env, codec=self.codec, requested=self.current_template)
        self.current_template = template
        self.last_resolved_template = template
        self.last_policy_high_template = policy_high_template
        return self.current_template

    def on_step_complete(self) -> None:
        return None


class PolicyTemplateSelector:
    """阶段 3 低层微调：由当前高层策略提供模板。"""

    def __init__(
        self,
        high_policy: HighLevelInferenceAdapter,
        update_interval: int = DEFAULT_HIGH_UPDATE_INTERVAL,
        codec: HierarchicalActionCodec | None = None,
    ) -> None:
        self.high_policy = high_policy
        self.update_interval = int(update_interval)
        self.codec = codec or HierarchicalActionCodec()
        self.current_template: HighLevelAction | None = None
        self.steps_since_refresh = 0
        self.last_selected_template: HighLevelAction | None = None
        self.last_executed_template: HighLevelAction | None = None
        self.last_num_legal_high_templates = 0
        self.last_policy_high_template = ""

    def reset(self, seed: int | None = None) -> None:
        del seed
        self.current_template = None
        self.steps_since_refresh = 0
        self.last_selected_template = None
        self.last_executed_template = None
        self.last_num_legal_high_templates = 0
        self.last_policy_high_template = ""
        if hasattr(self.high_policy, "reset"):
            self.high_policy.reset()

    def template_for_step(self, env: BlockchainGovEnv) -> HighLevelAction:
        self.last_num_legal_high_templates = count_legal_high_templates(env=env, codec=self.codec)
        needs_refresh = self.current_template is None or self.steps_since_refresh >= self.update_interval
        if self.current_template is not None and build_low_level_mask(env=env, codec=self.codec, high_action=self.current_template).sum() == 0:
            needs_refresh = True
        if needs_refresh:
            _, self.current_template, _ = self.high_policy.select(env)
            self.steps_since_refresh = 0
            self.last_selected_template = getattr(self.high_policy, "last_requested_template", self.current_template)
            self.last_policy_high_template = getattr(self.high_policy, "last_policy_high_template", "")
        if self.current_template is None:
            raise HierarchicalActionConstraintError("High-level template selector did not produce a template.")
        self.current_template, _, _, policy_high_template = resolve_template(env=env, codec=self.codec, requested=self.current_template)
        self.last_policy_high_template = policy_high_template
        if self.last_selected_template is None:
            self.last_selected_template = self.current_template
        self.last_executed_template = self.current_template
        return self.current_template

    def on_step_complete(self) -> None:
        self.steps_since_refresh += 1


class HierarchicalPolicyController:
    """评估阶段的分层执行控制器。"""

    def __init__(
        self,
        high_model: Any,
        low_model: Any,
        update_interval: int = DEFAULT_HIGH_UPDATE_INTERVAL,
        deterministic: bool = True,
    ) -> None:
        self.codec = HierarchicalActionCodec()
        self.high_adapter = HighLevelInferenceAdapter(model=high_model, deterministic=deterministic, codec=self.codec)
        self.low_adapter = LowLevelInferenceAdapter(model=low_model, deterministic=deterministic, codec=self.codec)
        self.update_interval = int(update_interval)
        self.current_template: HighLevelAction | None = None
        self.steps_since_refresh = 0
        self.last_selected_template: HighLevelAction | None = None
        self.last_executed_template: HighLevelAction | None = None
        self.last_num_legal_high_templates = 0
        self.last_policy_high_template = ""

    def reset(self) -> None:
        self.current_template = None
        self.steps_since_refresh = 0
        self.last_selected_template = None
        self.last_executed_template = None
        self.last_num_legal_high_templates = 0
        self.last_policy_high_template = ""
        self.high_adapter.reset()
        self.low_adapter.reset()

    def _refresh_template(self, env: BlockchainGovEnv) -> None:
        _, self.current_template, _ = self.high_adapter.select(env)
        self.last_selected_template = self.high_adapter.last_requested_template
        self.last_executed_template = self.current_template
        self.last_num_legal_high_templates = self.high_adapter.last_num_legal_high_templates
        self.last_policy_high_template = self.high_adapter.last_policy_high_template
        self.steps_since_refresh = 0

    def select_action(self, env: Any, obs: dict[str, np.ndarray]) -> int:
        del obs
        base_env = env.unwrapped if hasattr(env, "unwrapped") else env
        if not isinstance(base_env, BlockchainGovEnv):
            raise TypeError("HierarchicalPolicyController expects BlockchainGovEnv.")
        self.last_num_legal_high_templates = count_legal_high_templates(env=base_env, codec=self.codec)
        needs_refresh = self.current_template is None or self.steps_since_refresh >= self.update_interval
        if self.current_template is not None and build_low_level_mask(base_env, codec=self.codec, high_action=self.current_template).sum() == 0:
            needs_refresh = True
        if needs_refresh:
            self._refresh_template(base_env)
        if self.last_selected_template is None:
            self.last_selected_template = self.current_template
            self.last_executed_template = self.current_template
        low_mask = build_low_level_mask(base_env, codec=self.codec, high_action=self.current_template)
        if low_mask.sum() == 0:
            self._refresh_template(base_env)
            low_mask = build_low_level_mask(base_env, codec=self.codec, high_action=self.current_template)
        if low_mask.sum() == 0:
            raise HierarchicalActionConstraintError(
                f"No legal low-level action remains for template {(self.current_template.m, self.current_template.theta)}."
            )
        low_idx, low_action, _ = self.low_adapter.select(base_env, high_action=self.current_template)
        self.steps_since_refresh += 1
        governance_action = self.codec.to_governance_action(high_action=self.current_template, low_action=low_action)
        assert_hierarchical_action(self.codec, governance_action)
        action_source = "selected"
        if self.high_adapter.last_action_source != "selected" or self.low_adapter.last_action_source != "selected":
            action_source = "hierarchical_remap"
        if hasattr(base_env, "set_pending_action_source"):
            base_env.set_pending_action_source(action_source)
        if hasattr(base_env, "set_pending_hierarchical_metadata"):
            base_env.set_pending_hierarchical_metadata(
                build_hierarchical_action_metadata(
                    codec=self.codec,
                    selected_high_template=self.last_selected_template,
                    executed_high_template=self.current_template,
                    selected_low_action=self.low_adapter.last_requested_low_action,
                    executed_low_action=low_action,
                    num_legal_high_templates=self.last_num_legal_high_templates,
                    action_source=action_source,
                    policy_high_template=self.last_policy_high_template,
                )
            )
        del low_idx
        return self.codec.flat_index(high_action=self.current_template, low_action=low_action)
