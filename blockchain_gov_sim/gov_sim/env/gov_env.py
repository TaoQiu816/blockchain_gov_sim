"""Gymnasium 联盟链治理环境。

环境闭环：
`reset -> 场景生成 -> 证据生成 -> 信誉更新 -> action mask -> step(action)
 -> 委员会抽样 -> 链侧性能/安全 -> reward/cost/info -> 下一时刻观测`

该环境是第四章实验的统一入口，训练、评估、benchmark、ablation、baseline
都通过它执行，从而保证所有比较都在同一仿真语义下完成。
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from gov_sim.constants import ACTION_DIM, EPS
from gov_sim.env.action_codec import ActionCodec, GovernanceAction
from gov_sim.env.action_mask import build_action_mask, is_action_legal
from gov_sim.env.observation_builder import build_state_vector
from gov_sim.env.reward_cost import compute_cost, compute_reward
from gov_sim.modules.chain_model import ChainModel
from gov_sim.modules.committee_sampler import CommitteeSampler
from gov_sim.modules.evidence_generator import EvidenceGenerator
from gov_sim.modules.evidence_generator import EvidenceBatch
from gov_sim.modules.reputation_model import ReputationModel, ReputationSnapshot
from gov_sim.modules.scenario_model import ScenarioModel, ScenarioStep
from gov_sim.utils.seed import seed_everything


class BlockchainGovEnv(gym.Env[dict[str, np.ndarray], int]):
    """面向联盟链治理的 Gymnasium 环境。

    设计原则：
    - 观测采用 `Dict(state, action_mask)`，便于与 MaskablePPO 对接；
    - 动作为扁平 400 维离散动作；
    - `info` 尽量返回完整审计指标，便于论文作图与结果复核。
    """

    metadata = {"render_modes": []}

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        self.seed_value = int(config["seed"])
        self.env_cfg = config["env"]
        self.eval_cfg = config.get("eval", {})
        self.num_rsus = int(self.env_cfg["num_rsus"])
        self.horizon = int(self.env_cfg["episode_length"])
        self.codec = ActionCodec()
        self.default_action = GovernanceAction(m=11, b=256, tau=40, theta=0.6)
        self.prev_action = self.default_action
        self.action_space = spaces.Discrete(ACTION_DIM)
        # 状态向量维度 = 6 个全局统计 + 5 个信誉分位数 + bins 个信誉直方图
        # + 4 个维度 * (1 个均值 + 3 个分位数) + 4 个上一步动作归一化特征
        self.state_dim = 31 + int(self.env_cfg["observation_hist_bins"])
        self.observation_space = spaces.Dict(
            {
                "state": spaces.Box(low=-1.0e6, high=1.0e6, shape=(self.state_dim,), dtype=np.float32),
                "action_mask": spaces.Box(low=0, high=1, shape=(ACTION_DIM,), dtype=np.int8),
            }
        )

        self.scenario = ScenarioModel(
            config=config["scenario"],
            num_rsus=self.num_rsus,
            malicious_ratio=float(self.env_cfg["malicious_ratio"]),
            seed=self.seed_value,
        )
        self.evidence_generator = EvidenceGenerator(config=config["scenario"], seed=self.seed_value)
        self.reputation_model = ReputationModel(config=config["reputation"], num_rsus=self.num_rsus)
        self.committee_sampler = CommitteeSampler(seed=self.seed_value)
        self.chain_model = ChainModel(config=config["chain"], h_min=float(self.env_cfg["h_min"]))
        self.epoch = 0
        self.queue_size = 0.0
        self.last_latency = 0.0
        self.current_scenario: ScenarioStep | None = None
        self.current_snapshot: ReputationSnapshot | None = None
        self.current_obs: dict[str, np.ndarray] | None = None
        self.current_mask = np.ones(ACTION_DIM, dtype=np.int8)
        self.current_context = np.zeros(6, dtype=np.float32)
        self.last_info: dict[str, Any] = {}
        self.current_evidence: EvidenceBatch | None = None
        self.committee_override_method: str | None = None

    def _eligible_nodes(self, theta: float) -> np.ndarray:
        """按照当前信誉快照与动作门槛 `theta` 计算资格集。"""

        if self.current_snapshot is None or self.current_scenario is None:
            return np.array([], dtype=np.int64)
        mask = (
            (self.current_snapshot.final_scores >= theta)
            & (self.current_scenario.online == 1)
            & (self.current_scenario.uptime >= float(self.env_cfg["u_min"]))
        )
        return np.where(mask)[0].astype(np.int64)

    def _prepare_epoch(self) -> None:
        """生成当前周期观测所需的全部中间量。

        顺序不可随意改变：
        1. 先生成场景；
        2. 再基于场景生成证据；
        3. 再更新信誉；
        4. 最后根据新信誉构造 action mask 与观测。
        """

        base_eligible = self._eligible_nodes(self.prev_action.theta).size if self.current_snapshot is not None else self.num_rsus
        self.current_scenario = self.scenario.step(
            epoch=self.epoch,
            queue_size=self.queue_size,
            last_latency=self.last_latency,
            eligible_size=base_eligible,
        )
        context = np.asarray(
            [
                float(self.current_scenario.arrivals),
                float(self.queue_size),
                float(self.last_latency),
                float(self.current_scenario.rtt),
                float(self.current_scenario.churn),
                float(base_eligible),
            ],
            dtype=np.float32,
        )
        evidence = self.evidence_generator.generate(
            epoch=self.epoch,
            base_probs=self.current_scenario.base_probs,
            malicious=self.current_scenario.malicious,
            online=self.current_scenario.online,
            uptime=self.current_scenario.uptime,
        )
        self.current_evidence = evidence
        self.current_snapshot = self.reputation_model.update(context=context, evidence=evidence)
        self.current_context = context
        self.current_mask = build_action_mask(
            codec=self.codec,
            prev_action=self.prev_action,
            trust_scores=self.current_snapshot.final_scores,
            uptime=self.current_scenario.uptime,
            online=self.current_scenario.online,
            u_min=float(self.env_cfg["u_min"]),
            delta_m_max=int(self.env_cfg["delta_m_max"]),
            delta_b_max=int(self.env_cfg["delta_b_max"]),
            delta_tau_max=int(self.env_cfg["delta_tau_max"]),
            delta_theta_max=float(self.env_cfg["delta_theta_max"]),
            unsafe_guard=bool(self.env_cfg["unsafe_action_guard"]),
            h_min=float(self.env_cfg["h_min"]),
        )
        mask_to_show = self.current_mask if bool(self.env_cfg.get("mask_illegal_actions", True)) else np.ones_like(self.current_mask)
        summary = {
            "A_e": float(self.current_scenario.arrivals),
            "Q_e": float(self.queue_size),
            "L_bar_e": float(self.last_latency),
            "RTT_e": float(self.current_scenario.rtt),
            "chi_e": float(self.current_scenario.churn),
            "eligible_size": float(self._eligible_nodes(self.prev_action.theta).size),
        }
        state_vector = build_state_vector(
            summary=summary,
            snapshot=self.current_snapshot,
            prev_action=self.prev_action,
            bins=int(self.env_cfg["observation_hist_bins"]),
        )
        self.current_obs = {"state": state_vector, "action_mask": mask_to_show.astype(np.int8)}

    def _copy_obs(self) -> dict[str, np.ndarray]:
        """返回当前观测的深拷贝，避免外部持有环境内部数组引用。"""

        if self.current_obs is None:
            raise RuntimeError("Current observation is not initialized.")
        return {
            "state": np.copy(self.current_obs["state"]),
            "action_mask": np.copy(self.current_obs["action_mask"]),
        }

    def action_masks(self) -> np.ndarray:
        """兼容 MaskablePPO 常见接口，返回布尔 mask。"""
        return self.current_mask.astype(bool)

    def get_governance_state(self) -> dict[str, Any]:
        """向 baseline / 调试器暴露环境内部治理状态。"""
        if self.current_snapshot is None or self.current_scenario is None:
            raise RuntimeError("Environment not initialized; call reset() first.")
        return {
            "epoch": self.epoch,
            "queue_size": self.queue_size,
            "snapshot": self.current_snapshot,
            "scenario": self.current_scenario,
            "prev_action": self.prev_action,
            "mask": self.current_mask.copy(),
            "committee_method": self.committee_override_method or str(self.env_cfg.get("committee_method", "soft_sortition")),
        }

    def _resolve_action(self, action_idx: int) -> tuple[GovernanceAction, int]:
        """解析动作并在需要时处理非法动作。

        - 正常训练：非法动作通过 mask 预先屏蔽，几乎不会发生；
        - 消融 `no_action_mask`：允许策略选到非法动作，但会打 reward penalty；
        - 若启用 mask 且仍传入非法动作，则回退到当前 mask 的首个合法动作。
        """

        requested = self.codec.decode(int(action_idx))
        legal = is_action_legal(
            action=requested,
            prev_action=self.prev_action,
            trust_scores=self.current_snapshot.final_scores,
            uptime=self.current_scenario.uptime,
            online=self.current_scenario.online,
            u_min=float(self.env_cfg["u_min"]),
            delta_m_max=int(self.env_cfg["delta_m_max"]),
            delta_b_max=int(self.env_cfg["delta_b_max"]),
            delta_tau_max=int(self.env_cfg["delta_tau_max"]),
            delta_theta_max=float(self.env_cfg["delta_theta_max"]),
            unsafe_guard=bool(self.env_cfg["unsafe_action_guard"]),
            h_min=float(self.env_cfg["h_min"]),
        )
        if legal or not bool(self.env_cfg.get("mask_illegal_actions", True)):
            return requested, int(not legal)
        fallback_idx = int(np.flatnonzero(self.current_mask)[0])
        return self.codec.decode(fallback_idx), 1

    def _sample_committee(self, action: GovernanceAction, eligible_nodes: np.ndarray) -> np.ndarray:
        """根据当前策略/基线指定的委员会机制采样委员会。"""
        if eligible_nodes.size == 0:
            return np.array([], dtype=np.int64)
        method = self.committee_override_method or str(self.env_cfg.get("committee_method", "soft_sortition"))
        if method == "topk":
            scores = self.current_snapshot.final_scores[eligible_nodes]
            top_idx = np.argsort(scores)[-action.m :][::-1]
            return eligible_nodes[top_idx].astype(np.int64)
        beta_s = float(self.env_cfg.get("soft_sortition_beta", 6.0))
        logits = beta_s * self.current_snapshot.final_scores[eligible_nodes]
        logits = logits - np.max(logits)
        weights = np.exp(logits) + EPS
        weights = weights / np.sum(weights)
        return self.committee_sampler.sample(candidates=eligible_nodes, weights=weights, committee_size=action.m)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """重置环境并返回首个观测。

        这里会把 seed 传递给所有内部随机模块，确保：
        - 同一 seed 下整条轨迹可复现；
        - 不同 seed 下 episode 真正独立，而不是反复复制同一条样本路径。
        """

        if seed is not None:
            self.seed_value = int(seed)
        if options and "config_override" in options:
            raise ValueError("config_override is not supported at env.reset(); create a new env instead.")
        seed_everything(self.seed_value)
        self.epoch = 0
        self.queue_size = 0.0
        self.last_latency = 0.0
        self.prev_action = self.default_action
        self.committee_override_method = None
        self.scenario.reset(seed=self.seed_value)
        self.evidence_generator.reset(seed=self.seed_value)
        self.reputation_model.reset()
        self.committee_sampler.reset(seed=self.seed_value)
        self.chain_model.reset()
        self._prepare_epoch()
        self.last_info = {"reset": True}
        return self._copy_obs(), self.last_info.copy()

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """执行一个治理周期。

        流程：
        1. 解码并校验动作；
        2. 按 `theta_e` 计算资格集；
        3. 软抽签或 Top-K 生成委员会；
        4. 计算链侧性能、安全状态、reward 与 cost；
        5. 更新队列、延迟、上一步动作；
        6. 若未终止，则准备下一周期观测。
        """

        if self.current_obs is None or self.current_snapshot is None or self.current_scenario is None:
            raise RuntimeError("Environment not initialized; call reset() before step().")
        chosen_action, invalid_action = self._resolve_action(int(action))
        eligible_nodes = self._eligible_nodes(chosen_action.theta)
        effective_committee_size = min(chosen_action.m, int(eligible_nodes.size))
        committee = self._sample_committee(GovernanceAction(effective_committee_size, chosen_action.b, chosen_action.tau, chosen_action.theta), eligible_nodes)
        chain_result = self.chain_model.step(
            queue_size=self.queue_size,
            arrivals=self.current_scenario.arrivals,
            committee=committee,
            committee_size=effective_committee_size,
            batch_size=chosen_action.b,
            tau_ms=chosen_action.tau,
            rtt=self.current_scenario.rtt,
            churn=self.current_scenario.churn,
            uptime=self.current_scenario.uptime,
            malicious=self.current_scenario.malicious,
        )
        reward, reward_terms = compute_reward(
            config=self.env_cfg,
            service_capacity=chain_result.service_capacity,
            latency=chain_result.total_latency,
            queue_next=chain_result.queue_next,
            action=chosen_action,
            prev_action=self.prev_action,
        )
        cost, cost_terms = compute_cost(
            config=self.env_cfg,
            unsafe=chain_result.unsafe,
            honest_ratio=chain_result.honest_ratio,
        )
        reward -= invalid_action * float(self.env_cfg["invalid_action_penalty"])
        reward_terms["invalid_action"] = -invalid_action * float(self.env_cfg["invalid_action_penalty"])

        malicious_threshold = float(self.eval_cfg.get("malicious_threshold", 0.45))
        trust_scores = self.current_snapshot.final_scores.copy()
        malicious_pred = (trust_scores < malicious_threshold).astype(np.int8)
        # `info` 作为审计总线，尽量把论文中关心的链侧指标和信誉侧指标都带出去。
        info = {
            "epoch": self.epoch,
            "A_e": int(self.current_scenario.arrivals),
            "Q_e": float(self.queue_size),
            "S_e": float(chain_result.service_capacity),
            "L_queue_e": float(chain_result.queue_latency),
            "L_batch_e": float(chain_result.batch_latency),
            "L_cons_e": float(chain_result.consensus_latency),
            "L_bar_e": float(chain_result.total_latency),
            "RTT_e": float(self.current_scenario.rtt),
            "chi_e": float(self.current_scenario.churn),
            "h_e": float(chain_result.honest_ratio),
            "U_e": int(chain_result.unsafe),
            "Z_e": int(chain_result.success),
            "m_e": int(chosen_action.m),
            "b_e": int(chosen_action.b),
            "tau_e": int(chosen_action.tau),
            "theta_e": float(chosen_action.theta),
            "eligible_size": int(eligible_nodes.size),
            "executed_committee_size": int(effective_committee_size),
            "pollute_rate": float(self.current_evidence.pollute_rate if self.current_evidence is not None else 0.0),
            "unsafe": int(chain_result.unsafe),
            "reward_terms": reward_terms,
            "cost_terms": cost_terms,
            "reward": float(reward),
            "cost": float(cost),
            "timeout_failure": int(chain_result.timeout_failure),
            "leader_unstable": int(chain_result.leader_unstable),
            "tps": float(chain_result.tps),
            "mask_ratio": float(np.mean(self.current_mask)),
            "queue_next": float(chain_result.queue_next),
            "invalid_action": int(invalid_action),
            "committee_members": committee.astype(int).tolist(),
            "dim_weights": {key: float(value) for key, value in self.current_snapshot.dim_weights.items()},
            "context_vector": self.current_context.astype(float).tolist(),
            "trust_scores": trust_scores.tolist(),
            "malicious_true": self.current_scenario.malicious.astype(np.int8).tolist(),
            "malicious_pred": malicious_pred.tolist(),
        }
        self.queue_size = chain_result.queue_next
        self.last_latency = chain_result.total_latency
        self.prev_action = chosen_action
        terminated = self.epoch + 1 >= self.horizon
        truncated = False
        self.epoch += 1
        if not terminated:
            self._prepare_epoch()
            next_obs = self._copy_obs()
        else:
            next_obs = self._copy_obs()
        self.last_info = info
        return next_obs, float(reward), terminated, truncated, info
