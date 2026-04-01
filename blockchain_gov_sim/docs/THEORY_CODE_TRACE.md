# THEORY_CODE_TRACE

本文件把 [THEORY_SPEC.md](/Users/qiutao/研/毕设/毕设/blockchainSim/blockchain_gov_sim/docs/THEORY_SPEC.md) 中的核心理论项，逐条映射到当前代码。判定标准只有两类：

- 理论一致
- 理论不一致

“结果不好看”“某个 seed 不稳”“工作点还没校准”都不属于理论不一致。

## 1. 系统对象与动作语义

### [理论项] 高层动作 `(m, theta)` 与低层动作 `(b, tau)`

- 文件：
  - `/Users/qiutao/研/毕设/毕设/blockchainSim/blockchain_gov_sim/gov_sim/hierarchical/spec.py`
  - `/Users/qiutao/研/毕设/毕设/blockchainSim/blockchain_gov_sim/gov_sim/env/action_codec.py`
- 类 / 函数：
  - `HighLevelAction`
  - `LowLevelAction`
  - `HierarchicalActionCodec`
  - `ActionCodec`
- 变量：
  - `HIGH_LEVEL_TEMPLATES`
  - `LOW_LEVEL_ACTIONS`
- 日志项：
  - `m_e`, `theta_e`, `b_e`, `tau_e`
  - `executed_high_template`, `executed_low_action`
- 当前实现是否一致：一致
- 说明：
  - 高层只暴露 `12` 个 `(m,theta)` 模板。
  - 低层只暴露 `(b,tau)` 执行动作。

## 2. 多维信誉

### [理论项] 折扣伪计数更新

- 文件：
  - `/Users/qiutao/研/毕设/毕设/blockchainSim/blockchain_gov_sim/gov_sim/modules/reputation_model.py`
  - `/Users/qiutao/研/毕设/毕设/blockchainSim/blockchain_gov_sim/gov_sim/modules/evidence_generator.py`
- 类 / 函数：
  - `EvidenceGenerator.generate`
  - `ReputationModel.update`
- 变量：
  - `delta_s`, `delta_f`
  - `self.alpha`, `self.beta`
  - `rho_cfg`
- 日志项：
  - `trust_scores`
  - `dim_weights`
- 当前实现是否一致：一致
- 说明：
  - 代码以 `Δs/Δf` 聚合多条软证据，等价于规格中的 `omega*y` 与 `omega*(1-y)` 累加。

### [理论项] 均值、方差、冷启动门控、不确定性惩罚

- 文件：
  - `/Users/qiutao/研/毕设/毕设/blockchainSim/blockchain_gov_sim/gov_sim/modules/reputation_model.py`
- 类 / 函数：
  - `ReputationModel.update`
- 变量：
  - `mu`
  - `var`
  - `kappa`
  - `score_dim`
  - `lambda_dim_cfg`
  - `n0`
- 日志项：
  - `current_snapshot.mu`
  - `current_snapshot.var`
  - `current_snapshot.kappa`
  - `current_snapshot.score_dim`
- 当前实现是否一致：一致

### [理论项] 上下文感知融合

- 文件：
  - `/Users/qiutao/研/毕设/毕设/blockchainSim/blockchain_gov_sim/gov_sim/env/gov_env.py`
  - `/Users/qiutao/研/毕设/毕设/blockchainSim/blockchain_gov_sim/gov_sim/modules/reputation_model.py`
- 类 / 函数：
  - `BlockchainGovEnv._prepare_epoch`
  - `ReputationModel._normalize_context`
  - `ReputationModel.update`
- 变量：
  - `context`
  - `context_eta`
  - `dim_weights`
  - `normalized_context`
- 日志项：
  - `context_vector`
  - `dim_weights`
- 当前实现是否一致：一致
- 说明：
  - 当前上下文正好是 10 维：`arrival, queue, dqueue, rtt, drtt, churn, eligible_prev, honest_lcb_prev, unsafe_prev, timeout_prev`。

### [理论项] 攻击惩罚

- 文件：
  - `/Users/qiutao/研/毕设/毕设/blockchainSim/blockchain_gov_sim/gov_sim/modules/evidence_generator.py`
  - `/Users/qiutao/研/毕设/毕设/blockchainSim/blockchain_gov_sim/gov_sim/modules/reputation_model.py`
- 类 / 函数：
  - `EvidenceGenerator._generate_recommendations`
  - `EvidenceGenerator._malicious_prob_adjustment`
  - `ReputationModel._calc_onoff_penalty`
  - `ReputationModel._calc_zigzag_penalty`
  - `ReputationModel._calc_rec_penalty`
  - `ReputationModel._calc_collusion_penalty`
- 变量：
  - `gamma_onoff`, `gamma_zigzag`, `gamma_rec_bias`, `gamma_collusion`, `gamma_stability`
  - `penalties`
- 日志项：
  - `pollute_rate`
  - `current_snapshot.penalties`
- 当前实现是否一致：一致
- 说明：
  - bad-mouthing / ballot-stuffing 通过推荐矩阵污染进入 `rec_bias` / `collusion`，没有被单独重复记成第二套最终惩罚。

## 3. 资格集—委员会—安全性三段式

### [理论项] `E(theta)`

- 文件：
  - `/Users/qiutao/研/毕设/毕设/blockchainSim/blockchain_gov_sim/gov_sim/env/gov_env.py`
  - `/Users/qiutao/研/毕设/毕设/blockchainSim/blockchain_gov_sim/gov_sim/env/action_mask.py`
- 类 / 函数：
  - `BlockchainGovEnv._eligible_nodes`
  - `is_action_legal`
- 变量：
  - `final_scores`
  - `uptime`
  - `online`
  - `theta`
- 日志项：
  - `eligible_size`
- 当前实现是否一致：一致

### [理论项] `theta` 只控制 eligibility

- 文件：
  - `/Users/qiutao/研/毕设/毕设/blockchainSim/blockchain_gov_sim/gov_sim/env/gov_env.py`
  - `/Users/qiutao/研/毕设/毕设/blockchainSim/blockchain_gov_sim/gov_sim/env/action_mask.py`
  - `/Users/qiutao/研/毕设/毕设/blockchainSim/blockchain_gov_sim/gov_sim/modules/chain_model.py`
- 类 / 函数：
  - `_eligible_nodes`
  - `is_action_legal`
  - `ChainModel.step`
- 变量：
  - `action.theta`
  - `eligible_nodes`
- 日志项：
  - `theta_e`
  - `eligible_size`
- 当前实现是否一致：一致
- 说明：
  - `theta` 不直接进入 `unsafe`、`margin_cost`、`consensus_latency` 公式。

### [理论项] structural_infeasible

- 文件：
  - `/Users/qiutao/研/毕设/毕设/blockchainSim/blockchain_gov_sim/gov_sim/env/action_mask.py`
  - `/Users/qiutao/研/毕设/毕设/blockchainSim/blockchain_gov_sim/gov_sim/env/gov_env.py`
  - `/Users/qiutao/研/毕设/毕设/blockchainSim/blockchain_gov_sim/gov_sim/hierarchical/controller.py`
  - `/Users/qiutao/研/毕设/毕设/blockchainSim/blockchain_gov_sim/gov_sim/hierarchical/envs.py`
- 类 / 函数：
  - `is_action_legal`
  - `_protocol_invalid_breakdown`
  - `build_high_level_mask`
  - `HighLevelGovEnv._infeasible_step`
- 变量：
  - `policy_invalid`
  - `structural_infeasible`
  - `num_legal_nominal_high_templates`
- 日志项：
  - `structural_infeasible`
  - `num_legal_high_templates`
- 当前实现是否一致：一致
- 说明：
  - 当前代码把“无合法模板”与 `unsafe`、`timeout_failure` 分开记录。

### [理论项] 委员会采样

- 文件：
  - `/Users/qiutao/研/毕设/毕设/blockchainSim/blockchain_gov_sim/gov_sim/env/gov_env.py`
  - `/Users/qiutao/研/毕设/毕设/blockchainSim/blockchain_gov_sim/gov_sim/modules/committee_sampler.py`
- 类 / 函数：
  - `BlockchainGovEnv._sample_committee`
  - `CommitteeSampler.sample`
  - `weighted_sample_without_replacement`
- 变量：
  - `soft_sortition_beta`
  - `weights = exp(beta_s * final_score)`
- 日志项：
  - `committee_members`
  - `executed_committee_size`
- 当前实现是否一致：一致

### [理论项] `h_LCB`

- 文件：
  - `/Users/qiutao/研/毕设/毕设/blockchainSim/blockchain_gov_sim/gov_sim/env/gov_env.py`
  - `/Users/qiutao/研/毕设/毕设/blockchainSim/blockchain_gov_sim/gov_sim/modules/chain_model.py`
- 类 / 函数：
  - `BlockchainGovEnv.step`
  - `ChainModel.step`
- 变量：
  - `committee_con_scores`
  - `committee_con_vars`
  - `lambda_h`
  - `h_lcb`
- 日志项：
  - `h_LCB_e`
  - `h_LCB`
  - `committee_con_score_mean`
- 当前实现是否一致：一致
- 已修正的不一致：
  - 修正前 `committee_con_scores` 传的是 `kappa["con"] * mu["con"]`。
  - 规格要求 `con_score = score_dim["con"]`。
  - 现已改为 `self.current_snapshot.score_dim["con"][committee]`。

### [理论项] `unsafe`

- 文件：
  - `/Users/qiutao/研/毕设/毕设/blockchainSim/blockchain_gov_sim/gov_sim/modules/chain_model.py`
  - `/Users/qiutao/研/毕设/毕设/blockchainSim/blockchain_gov_sim/gov_sim/env/gov_env.py`
- 类 / 函数：
  - `ChainModel.step`
  - `BlockchainGovEnv.step`
- 变量：
  - `unsafe = int(structural_feasible and (h_lcb < h_min))`
- 日志项：
  - `unsafe`
  - `U_e`
- 当前实现是否一致：一致

### [理论项] `margin_cost`

- 文件：
  - `/Users/qiutao/研/毕设/毕设/blockchainSim/blockchain_gov_sim/gov_sim/modules/chain_model.py`
  - `/Users/qiutao/研/毕设/毕设/blockchainSim/blockchain_gov_sim/gov_sim/env/reward_cost.py`
- 类 / 函数：
  - `ChainModel.step`
  - `compute_cost`
- 变量：
  - `h_warn`
  - `margin_cost`
  - `cost_weights.margin`
- 日志项：
  - `margin_cost`
  - `cost_terms.margin_cost`
- 当前实现是否一致：一致
- 说明：
  - `h_warn`、`lambda_h` 的理论角色明确，但它们的具体数值仍属于工作点参数，不是文献常数。

## 4. 性能模型

### [理论项] `queue / arrivals / batch / served / consensus latency / timeout`

- 文件：
  - `/Users/qiutao/研/毕设/毕设/blockchainSim/blockchain_gov_sim/gov_sim/modules/chain_model.py`
  - `/Users/qiutao/研/毕设/毕设/blockchainSim/blockchain_gov_sim/gov_sim/env/gov_env.py`
- 类 / 函数：
  - `ChainModel.step`
  - `BlockchainGovEnv.step`
- 变量：
  - `x_total`
  - `wait_to_fill`
  - `effective_batch`
  - `mu_eff`
  - `served`
  - `queue_next`
  - `queue_latency`
  - `batch_latency`
  - `consensus_latency`
  - `tau_view`
  - `timeout_failure`
- 日志项：
  - `Q_e`, `A_e`, `S_e`, `B_e`
  - `L_queue_e`, `L_batch_e`, `L_cons_e`, `L_bar_e`
  - `timeout_failure`
- 当前实现是否一致：一致

### [理论项] `m / theta / b / tau` 的作用层次

- 文件：
  - `/Users/qiutao/研/毕设/毕设/blockchainSim/blockchain_gov_sim/gov_sim/modules/chain_model.py`
  - `/Users/qiutao/研/毕设/毕设/blockchainSim/blockchain_gov_sim/gov_sim/env/gov_env.py`
  - `/Users/qiutao/研/毕设/毕设/blockchainSim/blockchain_gov_sim/gov_sim/env/action_mask.py`
- 类 / 函数：
  - `ChainModel.step`
  - `_eligible_nodes`
  - `is_action_legal`
- 变量：
  - `committee_size`, `theta`, `batch_size`, `tau_ms`
- 日志项：
  - `m_e`, `theta_e`, `b_e`, `tau_e`
- 当前实现是否一致：一致
- 说明：
  - `b`、`tau` 没有被硬编码成“越大越好”或“越小越好”。

## 5. CMDP 与 chunk 语义

### [理论项] reward / cost 分离

- 文件：
  - `/Users/qiutao/研/毕设/毕设/blockchainSim/blockchain_gov_sim/gov_sim/env/reward_cost.py`
  - `/Users/qiutao/研/毕设/毕设/blockchainSim/blockchain_gov_sim/gov_sim/agent/masked_ppo_lagrangian.py`
- 类 / 函数：
  - `compute_reward`
  - `compute_cost`
  - `MaskablePPOLagrangian.train`
- 变量：
  - `reward_terms`
  - `cost_terms`
  - `advantages`
  - `cost_advantages`
- 日志项：
  - `reward_terms`
  - `cost_terms`
  - `train/constraint_violation`
- 当前实现是否一致：一致

### [理论项] high-level chunk transition

- 文件：
  - `/Users/qiutao/研/毕设/毕设/blockchainSim/blockchain_gov_sim/gov_sim/hierarchical/envs.py`
  - `/Users/qiutao/研/毕设/毕设/blockchainSim/blockchain_gov_sim/gov_sim/agent/masked_ppo_lagrangian.py`
- 类 / 函数：
  - `HighLevelGovEnv._execute_template_chunk`
  - `ConstrainedDictRolloutBuffer.add`
  - `ConstrainedDictRolloutBuffer.compute_returns_and_advantage`
- 变量：
  - `discounted_reward`
  - `discounted_cost`
  - `next_obs`
  - `rollout_buffer.costs`
- 日志项：
  - `high_chunk_len`
  - `high_chunk_discounted_reward`
  - `high_chunk_discounted_cost`
- 当前实现是否一致：一致
- 说明：
  - 每个高层动作只产生 1 条 chunk transition。
  - `next_obs` 取 chunk 结束后的高层状态。

### [理论项] high-level dual update 的约束尺度

- 文件：
  - `/Users/qiutao/研/毕设/毕设/blockchainSim/blockchain_gov_sim/gov_sim/hierarchical/envs.py`
  - `/Users/qiutao/研/毕设/毕设/blockchainSim/blockchain_gov_sim/gov_sim/agent/masked_ppo_lagrangian.py`
  - `/Users/qiutao/研/毕设/毕设/blockchainSim/blockchain_gov_sim/gov_sim/agent/callbacks.py`
- 类 / 函数：
  - `HighLevelGovEnv._execute_template_chunk`
  - `MaskablePPOLagrangian.collect_rollouts`
  - `MaskablePPOLagrangian.train`
  - `TrainLoggingCallback._on_step`
- 变量：
  - `high_chunk_normalized_cost`
  - `high_chunk_gamma_sum`
  - `constraint_costs`
  - `cost_limit`
- 日志项：
  - `high_chunk_discounted_cost`
  - `high_chunk_normalized_cost`
  - `high_chunk_gamma_sum`
  - `constraint_violation`
- 当前实现是否一致：一致
- 已修正的不一致：
  - 修正前 dual update 直接使用 discounted chunk cost，与旧 `cost_limit` 的原子周期平均语义失配。
  - 现已改为 `normalized_high_cost = discounted_chunk_cost / S_gamma(K)` 只用于 `lambda update`。

## 6. 日志可观测性

### [理论项] 理论语义必须能被日志观测

- 文件：
  - `/Users/qiutao/研/毕设/毕设/blockchainSim/blockchain_gov_sim/gov_sim/agent/callbacks.py`
  - `/Users/qiutao/研/毕设/毕设/blockchainSim/blockchain_gov_sim/gov_sim/utils/log_export.py`
  - `/Users/qiutao/研/毕设/毕设/blockchainSim/blockchain_gov_sim/gov_sim/utils/train_artifacts.py`
- 类 / 函数：
  - `TrainLoggingCallback._on_step`
  - `TrainLoggingCallback._on_training_end`
- 变量 / 日志项：
  - `unsafe`
  - `timeout_failure`
  - `structural_infeasible`
  - `eligible_size`
  - `h_LCB_e`
  - `committee_honest_ratio`
  - `L_queue_e`, `L_batch_e`, `L_cons_e`
  - `high_chunk_discounted_cost`, `high_chunk_normalized_cost`, `high_chunk_gamma_sum`
- 当前实现是否一致：一致

## 7. 当前结论

当前主线代码与理论规格的关系如下：

- 已一致：
  - 多维信誉
  - `E(theta)`
  - 委员会软抽签
  - `unsafe / timeout / structural_infeasible` 语义分离
  - 性能结构方程
  - high-level chunk transition 与 dual update 尺度
- 已发现并修正：
  - `h_LCB` 的输入应使用 `score_dim["con"]`，而不是 `kappa*mu`
- 当前仍存在的理论不一致：
  - 无新的主线理论不一致项；后续问题若出现，应先判定是工作点问题还是策略学习问题，不能再反向改理论。
