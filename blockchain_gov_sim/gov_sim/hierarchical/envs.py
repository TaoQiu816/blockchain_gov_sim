"""分层训练用环境 wrapper。"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from gov_sim.env.gov_env import BlockchainGovEnv
from gov_sim.hierarchical.controller import (
    EpisodeSampledTemplateSelector,
    HierarchicalActionConstraintError,
    LowLevelInferenceAdapter,
    PolicyTemplateSelector,
    assert_hierarchical_action,
    build_hierarchical_action_metadata,
    build_high_level_mask,
    build_low_level_mask,
    count_legal_high_templates,
    resolve_template,
    summarize_low_level_usage,
)
from gov_sim.hierarchical.observation import HIGH_STATE_DIM, LOW_STATE_DIM, build_high_level_state, build_low_level_state
from gov_sim.hierarchical.spec import DEFAULT_HIGH_UPDATE_INTERVAL, HighLevelAction, HierarchicalActionCodec


class LowLevelGovEnv(gym.Env[dict[str, np.ndarray], int]):
    """只暴露 `(b,tau)` 的 fast-timescale 训练环境。"""

    metadata = {"render_modes": []}

    def __init__(self, config: dict[str, Any], template_selector: EpisodeSampledTemplateSelector | PolicyTemplateSelector) -> None:
        super().__init__()
        self.base_env = BlockchainGovEnv(config=deepcopy(config))
        self.base_env.set_invalid_action_mode("raise")
        self.template_selector = template_selector
        self.codec = HierarchicalActionCodec()
        self.action_space = spaces.Discrete(self.codec.low_dim)
        self.observation_space = spaces.Dict(
            {
                "state": spaces.Box(low=-1.0e6, high=1.0e6, shape=(LOW_STATE_DIM,), dtype=np.float32),
                "action_mask": spaces.Box(low=0, high=1, shape=(self.codec.low_dim,), dtype=np.int8),
            }
        )

    def _hierarchical_metadata(
        self,
        selected: HighLevelAction | None,
        executed: HighLevelAction | None,
        action_source: str,
        num_legal: int,
        selected_low: Any = None,
        executed_low: Any = None,
        policy_high_template: str | None = None,
    ) -> dict[str, int | str]:
        return build_hierarchical_action_metadata(
            codec=self.codec,
            selected_high_template=selected,
            executed_high_template=executed,
            selected_low_action=selected_low,
            executed_low_action=executed_low,
            num_legal_high_templates=num_legal,
            action_source=action_source,
            policy_high_template=policy_high_template,
        )

    def _infeasible_step(
        self,
        selected_high_template: HighLevelAction | None,
        executed_high_template: HighLevelAction | None,
        reason: str = "no_legal_high_template",
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        num_legal = count_legal_high_templates(env=self.base_env, codec=self.codec)
        metadata = self._hierarchical_metadata(
            selected=selected_high_template,
            executed=executed_high_template,
            action_source="infeasible_truncation",
            num_legal=num_legal,
        )
        scenario = self.base_env.current_scenario
        obs = {
            "state": build_low_level_state(self.base_env, high_action=selected_high_template or self.codec.high_actions[0]),
            "action_mask": np.zeros(self.codec.low_dim, dtype=np.int8),
        }
        info = {
            "epoch": int(self.base_env.epoch),
            "episode_seed": int(self.base_env.current_episode_seed),
            "scenario_type": str(scenario.scenario_type) if scenario is not None else "",
            "scenario_phase": str(scenario.scenario_phase) if scenario is not None else "",
            "reward": 0.0,
            "cost": 0.0,
            "tps": 0.0,
            "L_bar_e": 0.0,
            "unsafe": 0,
            "timeout_failure": 0,
            "mask_ratio": 0.0,
            "eligible_size": int(self.base_env._eligible_nodes(self.base_env.prev_action.theta).size),
            "Q_e": float(self.base_env.queue_size),
            "A_e": int(scenario.arrivals) if scenario is not None else 0,
            "infeasible": 1,
            "policy_invalid": 0,
            "structural_infeasible": 1,
            "infeasible_reason": reason,
            "invalid_action": 0,
            "attempted_action": "",
            "executed_action": "",
            "attempted_action_idx": -1,
            "executed_action_idx": -1,
            "m_e": 0,
            "b_e": 0,
            "tau_e": 0,
            "theta_e": -1.0,
            "committee_members": [],
            **metadata,
        }
        return obs, 0.0, False, True, info

    def _current_template(self) -> tuple[HighLevelAction | None, dict[str, int | str]]:
        try:
            requested = self.template_selector.template_for_step(self.base_env)
        except HierarchicalActionConstraintError:
            selected = getattr(self.template_selector, "last_selected_template", None)
            if selected is None:
                selected = getattr(self.template_selector, "last_requested_template", None)
            return None, self._hierarchical_metadata(
                selected=selected,
                executed=None,
                action_source="infeasible_truncation",
                num_legal=count_legal_high_templates(env=self.base_env, codec=self.codec),
            )
        template, _, action_source, policy_high_template = resolve_template(env=self.base_env, codec=self.codec, requested=requested)
        if hasattr(self.template_selector, "current_template"):
            self.template_selector.current_template = template
        selected = getattr(self.template_selector, "last_selected_template", None)
        if selected is None:
            selected = getattr(self.template_selector, "last_requested_template", requested)
        metadata = self._hierarchical_metadata(
            selected=selected,
            executed=template,
            action_source=action_source,
            num_legal=getattr(
                self.template_selector,
                "last_num_legal_high_templates",
                count_legal_high_templates(env=self.base_env, codec=self.codec),
            ),
            policy_high_template=getattr(self.template_selector, "last_policy_high_template", policy_high_template),
        )
        return template, metadata

    def _build_obs(self, template: HighLevelAction) -> dict[str, np.ndarray]:
        mask = build_low_level_mask(env=self.base_env, codec=self.codec, high_action=template)
        return {
            "state": build_low_level_state(self.base_env, high_action=template),
            "action_mask": mask.astype(np.int8),
        }

    def action_masks(self) -> np.ndarray:
        template, _ = self._current_template()
        if template is None:
            return np.zeros(self.codec.low_dim, dtype=bool)
        return self._build_obs(template)["action_mask"].astype(bool)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        _, info = self.base_env.reset(seed=seed, options=options)
        self.template_selector.reset(seed=seed)
        template, metadata = self._current_template()
        if template is None:
            obs = {
                "state": build_low_level_state(self.base_env, high_action=self.codec.high_actions[0]),
                "action_mask": np.zeros(self.codec.low_dim, dtype=np.int8),
            }
            info = info.copy()
            info.update(metadata)
            info["infeasible"] = 1
            info["policy_invalid"] = 0
            info["structural_infeasible"] = 1
            info["infeasible_reason"] = "no_legal_high_template"
            return obs, info
        info = info.copy()
        info["high_template_m"] = int(template.m)
        info["high_template_theta"] = float(template.theta)
        info.update(metadata)
        return self._build_obs(template), info

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        template, metadata = self._current_template()
        if template is None:
            return self._infeasible_step(
                selected_high_template=getattr(self.template_selector, "last_selected_template", None)
                or getattr(self.template_selector, "last_requested_template", None),
                executed_high_template=None,
            )
        low_action = self.codec.decode_low(int(action))
        governance_action = self.codec.to_governance_action(high_action=template, low_action=low_action)
        assert_hierarchical_action(self.codec, governance_action)
        flat_idx = self.codec.flat_index(high_action=template, low_action=low_action)
        metadata = metadata.copy()
        metadata["selected_low_action"] = self.codec.low_action_repr(low_action)
        metadata["executed_low_action"] = self.codec.low_action_repr(low_action)
        self.base_env.set_pending_action_source(str(metadata["action_source"]))
        self.base_env.set_pending_hierarchical_metadata(metadata)
        _, reward, terminated, truncated, info = self.base_env.step(flat_idx)
        self.template_selector.on_step_complete()
        next_template = template
        if not (terminated or truncated):
            resolved_next_template, _ = self._current_template()
            if resolved_next_template is None:
                infeasible_obs = {
                    "state": build_low_level_state(self.base_env, high_action=template),
                    "action_mask": np.zeros(self.codec.low_dim, dtype=np.int8),
                }
                info = info.copy()
                info["infeasible"] = 1
                info["policy_invalid"] = 0
                info["structural_infeasible"] = 1
                info["infeasible_reason"] = "no_legal_high_template"
                info["num_legal_high_templates"] = 0
                info["selected_high_template"] = self.codec.high_action_repr(
                    getattr(self.template_selector, "last_selected_template", None)
                    or getattr(self.template_selector, "last_requested_template", None)
                )
                info["executed_high_template"] = ""
                info["selected_low_action"] = ""
                info["executed_low_action"] = ""
                info["action_source"] = "infeasible_truncation"
                info["high_template_m"] = int(template.m)
                info["high_template_theta"] = float(template.theta)
                return infeasible_obs, float(reward), False, True, info
            next_template = resolved_next_template
        info = info.copy()
        info["high_template_m"] = int(template.m)
        info["high_template_theta"] = float(template.theta)
        return self._build_obs(next_template), float(reward), terminated, truncated, info


class HighLevelGovEnv(gym.Env[dict[str, np.ndarray], int]):
    """只暴露 `(m,theta)` 的 slow-timescale 训练环境。"""

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: dict[str, Any],
        low_policy: LowLevelInferenceAdapter,
        update_interval: int = DEFAULT_HIGH_UPDATE_INTERVAL,
    ) -> None:
        super().__init__()
        self.base_env = BlockchainGovEnv(config=deepcopy(config))
        self.base_env.set_invalid_action_mode("raise")
        self.low_policy = low_policy
        self.codec = HierarchicalActionCodec()
        self.update_interval = int(update_interval)
        self.horizon = int(getattr(self.base_env, "horizon", 0))
        self.high_gamma = float(config.get("agent", {}).get("gamma", 0.99))
        self.high_gamma_sum = float((1.0 - self.high_gamma**self.update_interval) / max(1.0 - self.high_gamma, 1.0e-8))
        self.action_space = spaces.Discrete(self.codec.high_dim)
        self.observation_space = spaces.Dict(
            {
                "state": spaces.Box(low=-1.0e6, high=1.0e6, shape=(HIGH_STATE_DIM,), dtype=np.float32),
                "action_mask": spaces.Box(low=0, high=1, shape=(self.codec.high_dim,), dtype=np.int8),
            }
        )

    def _build_obs(self) -> dict[str, np.ndarray]:
        mask = build_high_level_mask(env=self.base_env, codec=self.codec)
        return {
            "state": build_high_level_state(self.base_env),
            "action_mask": mask.astype(np.int8),
        }

    def _execute_template_chunk(
        self,
        requested_template: HighLevelAction | None,
        template: HighLevelAction,
        initial_mask: np.ndarray,
        policy_high_template: str,
        template_source: str,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any], bool]:
        rewards: list[float] = []
        costs: list[float] = []
        tps_values: list[float] = []
        latency_values: list[float] = []
        eligible_sizes: list[float] = []
        unsafe_values: list[float] = []
        timeout_values: list[float] = []
        low_actions = []
        last_info: dict[str, Any] | None = None
        terminated = False
        truncated = False
        trigger_refresh = False
        hit_infeasible_next_state = False

        for step_idx in range(self.update_interval):
            if count_legal_high_templates(env=self.base_env, codec=self.codec) == 0:
                hit_infeasible_next_state = True
                truncated = True
                break
            low_mask = build_low_level_mask(env=self.base_env, codec=self.codec, high_action=template)
            if low_mask.sum() == 0:
                if step_idx == 0:
                    try:
                        template, _, template_source, policy_high_template = resolve_template(
                            env=self.base_env,
                            codec=self.codec,
                            requested=template,
                        )
                    except HierarchicalActionConstraintError:
                        return self._infeasible_step(
                            selected_high_template=requested_template,
                            executed_high_template=None,
                            policy_high_template=policy_high_template,
                        ) + (False,)
                    low_mask = build_low_level_mask(env=self.base_env, codec=self.codec, high_action=template)
                if low_mask.sum() == 0:
                    remaining_high = count_legal_high_templates(env=self.base_env, codec=self.codec)
                    if remaining_high == 0:
                        if rewards:
                            hit_infeasible_next_state = True
                            truncated = True
                            break
                        return self._infeasible_step(
                            selected_high_template=requested_template,
                            executed_high_template=None,
                            policy_high_template=policy_high_template,
                        ) + (False,)
                    trigger_refresh = True
                    break
            low_result = self.low_policy.select(self.base_env, high_action=template)
            if len(low_result) == 4:
                _, low_action, _, low_source = low_result
            else:
                _, low_action, _ = low_result
                low_source = getattr(self.low_policy, "last_action_source", "selected")
            selected_low_action = getattr(self.low_policy, "last_requested_low_action", low_action)
            governance_action = self.codec.to_governance_action(high_action=template, low_action=low_action)
            assert_hierarchical_action(self.codec, governance_action)
            flat_idx = self.codec.flat_index(high_action=template, low_action=low_action)
            action_source = "selected"
            if template_source != "selected" or low_source != "selected":
                action_source = "hierarchical_remap"
            self.base_env.set_pending_action_source(action_source)
            self.base_env.set_pending_hierarchical_metadata(
                build_hierarchical_action_metadata(
                    codec=self.codec,
                    selected_high_template=requested_template,
                    executed_high_template=template,
                    selected_low_action=selected_low_action,
                    executed_low_action=low_action,
                    num_legal_high_templates=int(initial_mask.sum()),
                    action_source=action_source,
                    policy_high_template=policy_high_template,
                )
            )
            _, reward, terminated, truncated, info = self.base_env.step(flat_idx)
            rewards.append(float(reward))
            costs.append(float(info.get("cost", 0.0)))
            tps_values.append(float(info.get("tps", 0.0)))
            latency_values.append(float(info.get("L_bar_e", 0.0)))
            eligible_sizes.append(float(info.get("eligible_size", 0.0)))
            unsafe_values.append(float(info.get("unsafe", 0.0)))
            timeout_values.append(float(info.get("timeout_failure", 0.0)))
            low_actions.append(low_action)
            last_info = info
            if terminated or truncated:
                break

        if not rewards or last_info is None:
            return self._infeasible_step(
                selected_high_template=requested_template,
                executed_high_template=None,
                policy_high_template=policy_high_template,
            ) + (False,)

        discount_factors = np.asarray([self.high_gamma**idx for idx in range(len(rewards))], dtype=np.float32)
        discounted_reward = float(np.sum(discount_factors * np.asarray(rewards, dtype=np.float32)))
        discounted_cost = float(np.sum(discount_factors * np.asarray(costs, dtype=np.float32)))
        gamma_sum = float(self.high_gamma_sum)
        normalized_cost = float(discounted_cost / max(gamma_sum, 1.0e-8))
        dominant_b, dominant_tau = summarize_low_level_usage(low_actions)
        info = last_info.copy()
        info["reward"] = discounted_reward
        info["cost"] = discounted_cost
        info["tps"] = float(np.mean(tps_values))
        info["L_bar_e"] = float(np.mean(latency_values))
        info["eligible_size"] = float(np.mean(eligible_sizes))
        info["unsafe"] = float(np.mean(unsafe_values))
        info["timeout_failure"] = float(np.mean(timeout_values))
        info["m_e"] = int(template.m)
        info["theta_e"] = float(template.theta)
        info["b_e"] = int(dominant_b)
        info["tau_e"] = int(dominant_tau)
        info["mask_ratio"] = float(np.mean(initial_mask)) if initial_mask.size else 0.0
        info["high_chunk_len"] = int(len(rewards))
        info["high_trigger_refresh"] = int(trigger_refresh)
        info["high_chunk_discounted_reward"] = discounted_reward
        info["high_chunk_discounted_cost"] = discounted_cost
        info["high_chunk_normalized_cost"] = normalized_cost
        info["high_chunk_gamma"] = float(self.high_gamma)
        info["high_chunk_gamma_sum"] = gamma_sum
        next_obs = self._build_obs()
        if not (terminated or truncated) and int(next_obs["action_mask"].sum()) == 0:
            hit_infeasible_next_state = True
            truncated = True
        if hit_infeasible_next_state:
            info["infeasible"] = 1
            info["policy_invalid"] = 0
            info["structural_infeasible"] = 1
            info["infeasible_reason"] = "no_legal_high_template"
            info["num_legal_high_templates"] = 0
            info["policy_high_template"] = policy_high_template
            info["selected_high_template"] = self.codec.high_action_repr(requested_template)
            info["executed_high_template"] = ""
            info["selected_low_action"] = ""
            info["executed_low_action"] = ""
            info["action_source"] = "infeasible_truncation"
        return next_obs, discounted_reward, terminated, truncated, info

    def _infeasible_step(
        self,
        selected_high_template: HighLevelAction | None,
        executed_high_template: HighLevelAction | None,
        reason: str = "no_legal_high_template",
        policy_high_template: str | None = None,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        mask = build_high_level_mask(env=self.base_env, codec=self.codec)
        metadata = build_hierarchical_action_metadata(
            codec=self.codec,
            selected_high_template=selected_high_template,
            executed_high_template=executed_high_template,
            selected_low_action=None,
            executed_low_action=None,
            num_legal_high_templates=int(mask.sum()),
            action_source="infeasible_truncation",
            policy_high_template=policy_high_template,
        )
        scenario = self.base_env.current_scenario
        info = {
            "epoch": int(self.base_env.epoch),
            "episode_seed": int(self.base_env.current_episode_seed),
            "scenario_type": str(scenario.scenario_type) if scenario is not None else "",
            "scenario_phase": str(scenario.scenario_phase) if scenario is not None else "",
            "reward": 0.0,
            "cost": 0.0,
            "tps": 0.0,
            "L_bar_e": 0.0,
            "unsafe": 0,
            "timeout_failure": 0,
            "mask_ratio": float(np.mean(mask)) if mask.size else 0.0,
            "eligible_size": int(self.base_env._eligible_nodes(self.base_env.prev_action.theta).size),
            "Q_e": float(self.base_env.queue_size),
            "A_e": int(scenario.arrivals) if scenario is not None else 0,
            "infeasible": 1,
            "policy_invalid": 0,
            "structural_infeasible": 1,
            "infeasible_reason": reason,
            "invalid_action": 0,
            "attempted_action": "",
            "executed_action": "",
            "attempted_action_idx": -1,
            "executed_action_idx": -1,
            "m_e": 0,
            "b_e": 0,
            "tau_e": 0,
            "theta_e": -1.0,
            "committee_members": [],
            "high_chunk_len": 0,
            "high_trigger_refresh": 0,
            **metadata,
        }
        return self._build_obs(), 0.0, False, True, info

    def action_masks(self) -> np.ndarray:
        return self._build_obs()["action_mask"].astype(bool)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        _, info = self.base_env.reset(seed=seed, options=options)
        if hasattr(self.low_policy, "reset"):
            self.low_policy.reset()
        obs = self._build_obs()
        info = info.copy()
        info["num_legal_high_templates"] = int(obs["action_mask"].sum())
        return obs, info

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        initial_mask = build_high_level_mask(env=self.base_env, codec=self.codec)
        if initial_mask.sum() == 0:
            return self._infeasible_step(selected_high_template=None, executed_high_template=None, policy_high_template="")
        
        requested_template = self.codec.decode_high(int(action))
        template, _, template_source, policy_high_template = resolve_template(
            env=self.base_env,
            codec=self.codec,
            requested=requested_template,
        )

        next_obs, reward, terminated, truncated, info = self._execute_template_chunk(
            requested_template=requested_template,
            template=template,
            initial_mask=initial_mask,
            policy_high_template=policy_high_template,
            template_source=template_source,
        )
        return next_obs, float(reward), terminated, truncated, info
