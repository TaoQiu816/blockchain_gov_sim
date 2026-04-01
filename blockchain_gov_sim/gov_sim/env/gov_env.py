"""单层离散治理 CMDP 环境。"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from gov_sim.constants import ACTION_DIM, EPS, H_MIN, H_WARN, M_MIN, THETA_CHOICES, TR_MIN
from gov_sim.env.action_codec import ActionCodec, GovernanceAction
from gov_sim.env.action_mask import build_action_mask, resolve_action_with_fallback
from gov_sim.env.observation_builder import build_state_vector, state_vector_dim
from gov_sim.env.reward_cost import compute_cost, compute_reward
from gov_sim.modules.chain_model import ChainModel
from gov_sim.modules.evidence_generator import EvidenceBatch, EvidenceGenerator
from gov_sim.modules.reputation_model import ReputationModel, ReputationSnapshot
from gov_sim.modules.scenario_model import ScenarioModel, ScenarioStep


class BlockchainGovEnv(gym.Env[np.ndarray, int]):
    """联盟链治理单层环境。

    接口：
    - reset() -> obs, legal_mask, info
    - step(action_idx) -> next_obs, next_legal_mask, reward, cost, done, info
    """

    metadata = {"render_modes": []}

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        self.env_cfg = config["env"]
        self.eval_cfg = config.get("eval", {})
        self.base_seed = int(config["seed"])
        self.seed_value = self.base_seed
        self.episode_counter = 0
        self.current_episode_seed = self.base_seed
        self.num_rsus = int(self.env_cfg["num_rsus"])
        self.horizon = int(self.env_cfg["episode_length"])
        self.m_min = int(self.env_cfg.get("m_min", M_MIN))
        self.tr_min = float(TR_MIN)
        self.h_min = float(H_MIN)
        self.h_warn = float(H_WARN)
        self.codec = ActionCodec()
        self.default_action = GovernanceAction(rho_m=7.0 / 27.0, theta=0.50, b=384, tau=80)
        self.default_action_idx = self.codec.encode(self.default_action)
        self.prev_action = self.default_action
        self.prev_action_idx = self.default_action_idx

        self.action_space = spaces.Discrete(ACTION_DIM)
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(state_vector_dim(),), dtype=np.float32)

        self.scenario = ScenarioModel(
            config=config["scenario"],
            num_rsus=self.num_rsus,
            malicious_ratio=float(self.env_cfg["malicious_ratio"]),
            seed=self.seed_value,
            episode_length=self.horizon,
        )
        self.evidence_generator = EvidenceGenerator(config=config["scenario"], seed=self.seed_value)
        self.reputation_model = ReputationModel(config=config["reputation"], num_rsus=self.num_rsus)
        self.chain_model = ChainModel(config=config["chain"], h_min=self.h_min)

        self.epoch = 0
        self.queue_size = 0.0
        self.last_latency = 0.0
        self.prev_rtt = 0.0
        self.current_scenario: ScenarioStep | None = None
        self.current_snapshot: ReputationSnapshot | None = None
        self.current_evidence: EvidenceBatch | None = None
        self.current_obs = np.zeros(state_vector_dim(), dtype=np.float32)
        self.current_mask = np.ones(ACTION_DIM, dtype=np.int8)
        self.current_eligible_counts: dict[float, int] = {float(theta): self.num_rsus for theta in THETA_CHOICES}
        self.last_info: dict[str, Any] = {}

        self.arrival_max = self._estimate_arrival_max()
        self.queue_max = float(self.env_cfg.get("Q_max", max(self.arrival_max * self.horizon, 1.0)))
        self.rtt_max = float(self.env_cfg.get("RTT_max", self._estimate_rtt_max()))

    def _estimate_arrival_max(self) -> float:
        scenario_cfg = self.config["scenario"]
        base_values: list[float] = []
        for name in ("stable", "step"):
            if name in scenario_cfg and "lambda" in scenario_cfg[name]:
                lam = float(scenario_cfg[name]["lambda"])
                if name == "step":
                    lam *= float(scenario_cfg[name].get("k", 1.0))
                base_values.append(lam)
        if "mmpp" in scenario_cfg:
            base_values.extend(float(value) for value in scenario_cfg["mmpp"].get("lambdas", []))
        mix_cfg = scenario_cfg.get("training_mix", {}).get("profiles", {})
        for profile in mix_cfg.values():
            if "high_lambda_scale" in profile:
                base_values.append(max(base_values or [1.0]) * float(profile["high_lambda_scale"]))
        return max(base_values or [1.0])

    def _estimate_rtt_max(self) -> float:
        network_cfg = self.config["scenario"].get("network", {})
        rtt_max = float(network_cfg.get("rtt_max", 1.0))
        mix_cfg = self.config["scenario"].get("training_mix", {}).get("profiles", {})
        for profile in mix_cfg.values():
            if "rtt_max" in profile:
                rtt_max = max(rtt_max, float(profile["rtt_max"]))
        return rtt_max

    def _next_episode_seed(self, seed: int | None) -> int:
        if seed is not None:
            self.base_seed = int(seed)
            self.episode_counter = 1
            return self.base_seed
        episode_seed = self.base_seed + self.episode_counter
        self.episode_counter += 1
        return int(episode_seed)

    def _snapshot_trust(self) -> np.ndarray:
        if self.current_snapshot is None:
            return np.zeros(self.num_rsus, dtype=np.float32)
        return np.asarray(self.current_snapshot.final_scores, dtype=np.float32)

    def _eligible_nodes(self, theta: float) -> np.ndarray:
        if self.current_snapshot is None:
            return np.array([], dtype=np.int64)
        return np.asarray(self.current_snapshot.eligible_sets.get(float(theta), np.array([], dtype=np.int64)), dtype=np.int64)

    def _build_context(self, eligible_size_hint: int) -> np.ndarray:
        rtt = float(self.current_scenario.rtt if self.current_scenario is not None else 0.0)
        delta_rtt = rtt - float(self.prev_rtt)
        return np.asarray(
            [
                float(self.current_scenario.arrivals if self.current_scenario is not None else 0.0),
                float(self.queue_size),
                float(self.queue_size),
                rtt,
                delta_rtt,
                float(self.current_scenario.churn if self.current_scenario is not None else 0.0),
                float(eligible_size_hint),
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )

    def _build_observation(self) -> np.ndarray:
        trust_scores = self._snapshot_trust()
        if self.current_scenario is None:
            raise RuntimeError("Scenario is not initialized.")
        online_mask = self.current_scenario.online == 1
        online_scores = trust_scores[online_mask]
        mean_trust = float(np.mean(online_scores)) if online_scores.size > 0 else 0.0
        std_trust = float(np.std(online_scores)) if online_scores.size > 0 else 0.0
        return build_state_vector(
            A_e=float(self.current_scenario.arrivals),
            Q_e=float(self.queue_size),
            RTT_e=float(self.current_scenario.rtt),
            delta_RTT_e=float(self.current_scenario.rtt - self.prev_rtt),
            churn_ratio_e=float(self.current_scenario.churn),
            online_ratio_e=float(np.mean(self.current_scenario.online)),
            mean_trust_e=mean_trust,
            std_trust_e=std_trust,
            n_045=int(self.current_eligible_counts[0.45]),
            n_050=int(self.current_eligible_counts[0.50]),
            n_055=int(self.current_eligible_counts[0.55]),
            n_060=int(self.current_eligible_counts[0.60]),
            previous_action_idx=int(self.prev_action_idx),
            A_max=self.arrival_max,
            Q_max=self.queue_max,
            RTT_max=self.rtt_max,
            N_RSU=self.num_rsus,
        )

    def _prepare_epoch(self) -> None:
        base_eligible = int(self.current_eligible_counts.get(float(self.prev_action.theta), self.num_rsus))
        self.current_scenario = self.scenario.step(
            epoch=self.epoch,
            queue_size=self.queue_size,
            last_latency=self.last_latency,
            eligible_size=base_eligible,
        )
        self.current_evidence = self.evidence_generator.generate(
            epoch=self.epoch,
            base_probs=self.current_scenario.base_probs,
            malicious=self.current_scenario.malicious,
            online=self.current_scenario.online,
            uptime=self.current_scenario.uptime,
        )
        self.current_snapshot = self.reputation_model.update(
            context=self._build_context(base_eligible),
            evidence=self.current_evidence,
        )
        self.current_eligible_counts = dict(self.current_snapshot.eligible_counts)
        self.current_mask = build_action_mask(
            codec=self.codec,
            eligible_counts_by_theta=self.current_eligible_counts,
            m_min=self.m_min,
        )
        self.current_obs = self._build_observation()

    def _select_committee(self, action: GovernanceAction, eligible_nodes: np.ndarray) -> tuple[np.ndarray, int]:
        mapped_m = self.codec.mapped_committee_size(action.rho_m, int(eligible_nodes.size), m_min=self.m_min)
        if mapped_m < self.m_min or eligible_nodes.size < mapped_m:
            return np.array([], dtype=np.int64), 0
        trust_scores = self._snapshot_trust()[eligible_nodes]
        order = np.lexsort((eligible_nodes, -trust_scores))
        committee = eligible_nodes[order[:mapped_m]].astype(np.int64)
        return committee, int(mapped_m)

    def _mapped_target_committee_size(self, action: GovernanceAction, eligible_size: int) -> int:
        mapped_m = self.codec.mapped_committee_size(action.rho_m, int(eligible_size), m_min=self.m_min)
        if int(eligible_size) < self.m_min:
            return int(self.m_min)
        return int(mapped_m)

    @staticmethod
    def _action_tuple(action: GovernanceAction, committee_size: int) -> tuple[float, float, int, int, int]:
        return (float(action.rho_m), float(action.theta), int(action.b), int(action.tau), int(committee_size))

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        episode_seed = self._next_episode_seed(seed)
        self.seed_value = episode_seed
        self.current_episode_seed = episode_seed
        self.epoch = 0
        self.queue_size = 0.0
        self.last_latency = 0.0
        self.prev_rtt = 0.0
        self.prev_action = self.default_action
        self.prev_action_idx = self.default_action_idx
        self.current_obs = np.zeros(state_vector_dim(), dtype=np.float32)
        self.current_mask = np.ones(ACTION_DIM, dtype=np.int8)
        self.current_eligible_counts = {float(theta): self.num_rsus for theta in THETA_CHOICES}
        self.action_space.seed(episode_seed)
        self.scenario.reset(seed=episode_seed)
        self.evidence_generator.reset(seed=episode_seed)
        self.reputation_model.reset()
        self.chain_model.reset()
        self._prepare_epoch()
        self.last_info = {
            "reset": True,
            "episode_seed": int(episode_seed),
            "scenario_type": str(self.current_scenario.scenario_type if self.current_scenario is not None else "unknown"),
        }
        return self.current_obs.copy(), self.current_mask.copy(), self.last_info.copy()

    def step(self, action_idx: int) -> tuple[np.ndarray, np.ndarray, float, float, bool, dict[str, Any]]:
        if self.current_scenario is None or self.current_snapshot is None:
            raise RuntimeError("Environment not initialized; call reset() first.")

        requested_action, executed_action, was_fallback, structural_from_mask = resolve_action_with_fallback(
            codec=self.codec,
            requested_idx=int(action_idx),
            mask=self.current_mask,
            eligible_counts_by_theta=self.current_eligible_counts,
        )
        requested_idx = int(action_idx)
        executed_idx = self.codec.encode(executed_action)
        eligible_nodes = self._eligible_nodes(executed_action.theta)
        eligible_size = int(eligible_nodes.size)
        mapped_m_e = self._mapped_target_committee_size(executed_action, eligible_size)
        committee, committee_size = self._select_committee(executed_action, eligible_nodes)
        structural_infeasible = int(structural_from_mask or eligible_size < mapped_m_e)
        committee_mean_trust = float(np.mean(self._snapshot_trust()[committee])) if committee.size > 0 else 0.0

        chain_result = self.chain_model.step(
            queue_size=self.queue_size,
            arrivals=int(self.current_scenario.arrivals),
            eligible_size=eligible_size,
            committee=committee,
            committee_size=int(committee_size),
            batch_size=int(executed_action.b),
            tau_ms=int(executed_action.tau),
            rtt=float(self.current_scenario.rtt),
            churn=float(self.current_scenario.churn),
            committee_trust_scores=self._snapshot_trust()[committee] if committee.size > 0 else np.array([], dtype=np.float32),
            malicious=self.current_scenario.malicious,
        )

        unsafe = int(committee_mean_trust < self.h_min)
        timeout = int(float(chain_result.consensus_latency) > float(executed_action.tau))
        margin = (
            float(np.clip((self.h_warn - committee_mean_trust) / max(self.h_warn - self.h_min, EPS), 0.0, 1.0))
            if committee_size > 0
            else 1.0
        )
        reward, reward_terms = compute_reward(
            config=self.env_cfg,
            served=float(chain_result.service_capacity),
            latency=float(chain_result.total_latency),
            queue_next=float(chain_result.queue_next),
            batch_size=int(executed_action.b),
            action=executed_action,
            prev_action=self.prev_action,
        )
        cost, cost_terms = compute_cost(
            config=self.env_cfg,
            unsafe=unsafe,
            timeout_failure=timeout,
            margin_cost=margin,
        )

        info = {
            "epoch": int(self.epoch),
            "episode_seed": int(self.current_episode_seed),
            "scenario_type": str(self.current_scenario.scenario_type),
            "scenario_phase": str(self.current_scenario.scenario_phase),
            "unsafe": int(unsafe),
            "timeout": int(timeout),
            "timeout_failure": int(timeout),
            "structural_infeasible": int(structural_infeasible),
            "eligible_size": eligible_size,
            "committee_size": int(committee_size),
            "committee_formed": int(committee_size > 0),
            "committee_mean_trust": float(committee_mean_trust),
            "pollute_rate": float(self.current_evidence.pollute_rate if self.current_evidence is not None else 0.0),
            "queue": float(self.queue_size),
            "queue_next": float(chain_result.queue_next),
            "served": float(chain_result.service_capacity),
            "latency": float(chain_result.total_latency),
            "action_tuple": self._action_tuple(executed_action, committee_size),
            "was_fallback": bool(was_fallback),
            "requested_action": (
                float(requested_action.rho_m),
                float(requested_action.theta),
                int(requested_action.b),
                int(requested_action.tau),
            ),
            "executed_action": (
                float(executed_action.rho_m),
                float(executed_action.theta),
                int(executed_action.b),
                int(executed_action.tau),
            ),
            "reward": float(reward),
            "cost": float(cost),
            "reward_terms": reward_terms,
            "cost_terms": cost_terms,
            "A_e": int(self.current_scenario.arrivals),
            "Q_e": float(self.queue_size),
            "B_e": float(chain_result.effective_batch),
            "S_e": float(chain_result.service_capacity),
            "Q_next": float(chain_result.queue_next),
            "L_queue": float(chain_result.queue_latency),
            "L_batch": float(chain_result.batch_latency),
            "L_cons": float(chain_result.consensus_latency),
            "L_e": float(chain_result.total_latency),
            "L_queue_e": float(chain_result.queue_latency),
            "L_batch_e": float(chain_result.batch_latency),
            "L_cons_e": float(chain_result.consensus_latency),
            "L_bar_e": float(chain_result.total_latency),
            "RTT_e": float(self.current_scenario.rtt),
            "delta_RTT_e": float(self.current_scenario.rtt - self.prev_rtt),
            "churn_ratio_e": float(self.current_scenario.churn),
            "online_ratio_e": float(np.mean(self.current_scenario.online)),
            "margin_e": float(margin),
            "h_LCB": float(chain_result.h_lcb),
            "h_LCB_e": float(chain_result.h_lcb),
            "mapped_m_e": int(mapped_m_e),
            "m_e": int(mapped_m_e),
            "rho_m_e": float(executed_action.rho_m),
            "theta_e": float(executed_action.theta),
            "b_e": int(executed_action.b),
            "tau_e": int(executed_action.tau),
            "requested_action_idx": int(requested_idx),
            "executed_action_idx": int(executed_idx),
            "used_fallback_action": int(was_fallback),
            "trust_scores": self._snapshot_trust().astype(float).tolist(),
            "committee_members": committee.astype(int).tolist(),
            "qualified_node_count_mean": float(eligible_size),
            "qualified_node_count": int(eligible_size),
            "action_distribution_key": f"{float(executed_action.rho_m):.6f}|{float(executed_action.theta):.2f}|{int(executed_action.b)}|{int(executed_action.tau)}",
            "tps": float(chain_result.tps),
        }

        self.queue_size = float(chain_result.queue_next)
        self.last_latency = float(chain_result.total_latency)
        self.prev_rtt = float(self.current_scenario.rtt)
        self.prev_action = executed_action
        self.prev_action_idx = executed_idx
        done = bool(self.epoch + 1 >= self.horizon)
        self.epoch += 1
        if not done:
            self._prepare_epoch()
        self.last_info = info
        return self.current_obs.copy(), self.current_mask.copy(), float(reward), float(cost), done, info

    def action_masks(self) -> np.ndarray:
        return self.current_mask.astype(bool)

    def get_governance_state(self) -> dict[str, Any]:
        if self.current_snapshot is None or self.current_scenario is None:
            raise RuntimeError("Environment not initialized; call reset() first.")
        return {
            "epoch": int(self.epoch),
            "queue_size": float(self.queue_size),
            "snapshot": self.current_snapshot,
            "scenario": self.current_scenario,
            "mask": self.current_mask.copy(),
            "eligible_counts_by_theta": dict(self.current_eligible_counts),
            "previous_action_idx": int(self.prev_action_idx),
            "previous_action": self.prev_action,
        }
