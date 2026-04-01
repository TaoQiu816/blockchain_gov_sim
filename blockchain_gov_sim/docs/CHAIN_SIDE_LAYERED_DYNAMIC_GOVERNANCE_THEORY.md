# 链侧分层动态治理理论设计

仓库路径：`/Users/qiutao/研/毕设/毕设/blockchainSim/blockchain_gov_sim`

本文档用于把当前 IoV/VEC 联盟链链侧分层动态治理模型，从“经验驱动的参数与测试”收口为“理论约束明确、实验校准支撑、训练可解释”的正式研究设计。本文档不触发训练，也不把“参数扫描是否单调”作为唯一有效性标准。

## 1. 变量与动作语义

### 1.1 时间、节点与状态

- 记治理周期为 `e = 0, 1, ..., H-1`。
- 记 RSU 节点集合为 `N = {1, ..., N_R}`。
- 对节点 `i`，定义：
  - `r_{i,e}`：周期 `e` 的最终信誉分数；
  - `o_{i,e} in {0,1}`：在线状态；
  - `u_{i,e} in [0,1]`：稳定性或 uptime；
  - `mal_i,e in {0,1}`：是否恶意。

环境在策略侧暴露的是压缩观测，而不是完整隐状态。理论上可把链侧治理状态写为：

`s_e = (A_e, Q_e, RTT_e, chi_e, r_e, o_e, u_e, a_{e-1}, feedback_{e-1})`

其中：

- `A_e`：本周期到达量；
- `Q_e`：周期开始时队列积压；
- `RTT_e`：网络往返时延；
- `chi_e`：churn 强度；
- `r_e = (r_{1,e}, ..., r_{N_R,e})`：信誉快照；
- `o_e, u_e`：在线与稳定性快照；
- `a_{e-1}`：上一步治理动作；
- `feedback_{e-1}`：上一步的 timeout / unsafe / latency 等反馈。

### 1.2 分层动作

高层动作：

- `m_e`：委员会规模目标；
- `theta_e`：资格门槛。

低层动作：

- `b_e`：区块批大小；
- `tau_e`：批等待超时。

语义区分：

- 高层 `(m, theta)` 决定“谁有资格进入候选池，以及最终委员会想要多大”。
- 低层 `(b, tau)` 决定“每轮准备装多少交易，以及等待多久再封块”。

### 1.3 资格集、委员会与核心指标

- 资格集 `E_e(theta_e)`：满足信誉、在线、稳定性门槛的候选节点集合。
- 委员会 `C_e`：从资格集中抽样出的执行委员会。
- 性能指标：
  - `L_queue_e`：排队时延；
  - `L_batch_e`：批处理等待时延；
  - `L_cons_e`：共识时延；
  - `L_bar_e`：总确认时延；
  - `S_e`：有效服务量；
  - `TPS_e`：吞吐率；
  - `Q_{e+1}`：下一周期队列。
- 安全指标：
  - `h_hat_e`：委员会实现诚实比例；
  - `h_LCB_e`：委员会诚实性下界；
  - `U_e`：unsafe 事件；
  - `T_e`：timeout 事件；
  - `M_e`：安全裕度成本 `margin_cost`。

## 2. 资格集 - 委员会 - 安全性三段式建模

### 2.1 候选资格集

定义：

`E_e(theta_e) = { i in N : r_{i,e} >= theta_e, o_{i,e} = 1, u_{i,e} >= u_min }`

这里 `theta` 只承担 admission control 语义，即只控制节点是否进入资格集。它不直接进入链侧时延公式，不直接进入 unsafe 判定公式，也不应被解释为“theta 越高，系统天然越安全”。

### 2.2 结构可行性

定义结构可行事件：

`F_e(m_e, theta_e) = 1[ |E_e(theta_e)| >= m_e ]`

解释：

- 若 `|E_e(theta_e)| < m_e`，则该高层模板在当前状态下不可执行；
- 这属于结构可行性问题，而不是 unsafe 事件。

### 2.3 委员会形成

当 `F_e = 1` 时，从资格集中形成委员会：

`C_e subset E_e(theta_e), |C_e| = m_e`

当前主线采用 soft sortition 加无放回加权抽样。若记资格集中节点权重为 `w_i(r_{i,e})`，则委员会由这些权重诱导的无放回抽样过程得到。

### 2.4 委员会诚实性下界

先定义实现诚实比例：

`h_hat_e = (1 / |C_e|) * sum_{i in C_e} (1 - mal_{i,e})`

再定义理论上的委员会诚实性下界：

`h_LCB_e = clip(h_hat_e - kappa_h / sqrt(max(|C_e|, 1)), 0, 1)`

其中：

- `kappa_h >= 0` 为诚实性下界收缩系数，是结构参数；
- 当 `kappa_h = 0` 时，`h_LCB_e = h_hat_e`，这对应当前代码的经验简化。

这个定义表达的是：委员会安全判断应基于“保守下界”而不是“只看样本均值”。这样 `m` 的作用不仅体现在委员会规模本身，还体现在下界估计的紧致度上。

### 2.5 unsafe 与 margin_cost

定义 unsafe：

`U_e = 1[ h_LCB_e < h_min ]`

定义安全裕度成本：

`M_e = clip((h_warn - h_LCB_e) / max(h_warn - h_min, eps), 0, 1)`

其中：

- `h_min` 是安全判定阈值；
- `h_warn > h_min` 是预警阈值；
- `M_e` 不等于 unsafe，而是“距离危险边界还有多近”的连续成本。

### 2.6 为什么 theta 与 unsafe 不是单调关系

`theta` 对 unsafe 的影响是联合效应，而不是单调关系。原因有三：

1. `theta` 提高时，`|E(theta)|` 通常下降，结构可行域收缩。
2. `theta` 提高时，候选池平均质量通常上升，委员会质量可能改善。
3. 在 soft sortition 下，委员会质量由“候选池大小、权重分布、抽样随机性、m 的大小”共同决定。

因此不能把 unsafe 口头简化成“theta 高就应该更安全”。理论上允许出现：

- `theta` 提高后 unsafe 下降；
- `theta` 提高后 unsafe 基本不变；
- `theta` 提高后由于资格集过小或可行域收缩，整体风险反而上升。

## 3. 性能结构方程

### 3.1 批前服务量

定义可被当前区块尝试处理的交易量：

`S_tilde_e = min(Q_e + A_e, b_e)`

这表示单周期内，最多只能从“当前积压 + 新到达”中装入 `b_e` 个交易。

### 3.2 时延分解

结构核心：

`L_bar_e^base = L_queue_e + L_batch_e + L_cons_e`

当前实现中的总时延写为：

`L_bar_e = L_queue_e + L_batch_e + L_cons_e + (1 - Z_e) * l_pen`

其中失败罚时 `l_pen` 是失败附加项，不改变三段式主体结构。

各项定义：

`L_queue_e = c_q * Q_e / max(S_tilde_e, 1)`

`L_batch_e = min(tau_e, [b_e - (Q_e + A_e)]_+ / max(lambda_hat_e, 1))`

`L_cons_e = c_0 + c_1 * RTT_e + c_2 * m_e * (m_e - 1) * l_ctrl / b_ctrl + c_3 * chi_e * m_e + c_4 * leader_unstable_e`

解释：

- 排队时延由积压与当前处理能力共同决定；
- 批等待时延由“区块还差多少装满”与“最多等多久 `tau`”共同决定；
- 共识时延由 RTT、委员会通信复杂度、churn 和 leader 不稳定共同决定。

### 3.3 吞吐与服务能力

定义共识成功事件：

`Z_e = 1[ h_LCB_e >= h_min ] * 1[ L_cons_e <= tau_view(tau_e) ]`

定义视图超时阈值：

`tau_view(tau_e) = max(tau_min, alpha_tau * tau_e)`

定义有效服务量：

`S_e = Z_e * S_tilde_e`

定义队列演化：

`Q_{e+1} = max(0, Q_e + A_e - S_e)`

定义吞吐：

`TPS_e = S_e / max(L_bar_e / 1000, eps)`

### 3.4 timeout 与 RTT / tau 的关系

定义 timeout：

`T_e = 1[ h_LCB_e >= h_min and L_cons_e > tau_view(tau_e) ]`

所以：

- `RTT` 上升会推高 `L_cons`；
- `tau` 上升会放宽 `tau_view`，从而可能降低 timeout；
- 但 `tau` 也会提高 `L_batch` 上界，因此存在时延 - timeout 折中，而不是全局单调优化。

### 3.5 block slack 与 b 的关系

定义区块松弛度：

`slack_e = [b_e - S_e]_+ / max(b_e, 1)`

解释：

- 低负载区间里，`b` 过大时更容易出现空块或半空块，`slack` 上升；
- 高负载区间里，`b` 增大可能减少排队并提高服务量，`slack` 反而下降。

因此 `b` 的作用必须按负载区间解释，而不能说“大区块一定更好”或“一定更差”。

## 4. 四个动作的理论影响方向

### 4.1 `m`：近似单调影响 latency / feasibility / security margin

在其他条件固定时：

- `m` 增大通常使 `L_cons` 上升，因为委员会通信复杂度增加；
- `m` 增大通常使结构可行性下降，因为需要更大的资格集；
- `m` 增大通常使安全下界估计更稳定，因为 `kappa_h / sqrt(m)` 收缩项减小。

因此 `m` 的理论角色是：用更高共识开销和更低可行性，去换取更强的委员会统计稳定性和潜在安全裕度。

### 4.2 `theta`：影响 eligible_size 与 committee quality，unsafe 不要求单调

在其他条件固定时：

- `theta` 主要影响 `|E(theta)|`；
- `theta` 也会改变候选池质量与权重分布，从而影响委员会质量；
- `theta` 对 unsafe 的影响是资格集规模与质量共同作用的结果，不要求单调。

因此 `theta` 的理论角色是 admission threshold，而不是“直接安全旋钮”。

### 4.3 `b`：负载区间折中，不是全局单调

在低负载区间：

- `b` 增大更容易带来更高 `L_batch` 和更高 `slack`。

在高负载区间：

- `b` 增大可能提高 `S_tilde`，缓解队列并改善吞吐。

因此 `b` 是负载相关的工作点动作，应通过拐点和区间折中来验证，而不是追求全局单调。

### 4.4 `tau`：RTT / timeout 区间折中，不是全局单调

在低 RTT 区间：

- 较大的 `tau` 主要增加等待时间，收益有限。

在高 RTT 区间：

- 较大的 `tau` 可能通过放宽 `tau_view` 降低 timeout，提升成功率。

因此 `tau` 是 RTT 相关的折中动作，不能按全局单调扫描裁决。

## 5. 参数分类

### 5.1 结构参数

结构参数由理论先定，不能通过“扫一圈看着顺眼”来拍脑袋决定。至少包括：

- 动作域：
  - `m in {5, 7, 9}`
  - `theta in {0.45, 0.50, 0.55, 0.60}`
  - `b in {256, 320, 384, 448, 512}`
  - `tau in {40, 60, 80, 100}`
- 资格规则：`u_min`
- 安全阈值：`h_min`
- 安全预警阈值：`h_warn`
- 诚实性下界系数：`kappa_h`
- 视图超时结构：`alpha_tau = tau_view_factor` 与 `tau_min = min_view_timeout`
- 高层更新时间隔：`update_interval`
- 委员会采样机制：soft sortition
- 动作平滑约束：`delta_m_max`, `delta_b_max`, `delta_tau_max`, `delta_theta_max`

### 5.2 工作点参数

工作点参数必须在理论可行域内，由实验校准，而不是反向决定理论。包括：

- 默认负载与扰动强度：
  - `lambda`
  - `RTT` 范围
  - `p_off / p_on`
  - `extra_malicious_ratio`
  - `high_lambda_scale`
- 各 hard scene 的具体强度；
- 训练用 reward / cost 权重；
- 训练 penalty 权重；
- 日志导出区间与绘图 ranges；
- smoke / full 审计采样规模。

## 6. 理论到代码映射表

| 理论项 | 正式定义 | 当前代码位置 | 现状判断 | 最小修正 |
|---|---|---|---|---|
| 高层动作 `(m, theta)` | 治理模板，控制委员会目标规模与 admission threshold | `gov_sim/hierarchical/spec.py`, `gov_sim/env/action_codec.py` | 已符合 | 无 |
| 低层动作 `(b, tau)` | 批大小与批等待超时 | `gov_sim/hierarchical/spec.py`, `gov_sim/env/action_codec.py` | 已符合 | 无 |
| 状态 `s_e` | 由负载、队列、RTT、churn、信誉、上一步反馈构成 | `gov_sim/env/gov_env.py::_build_observation_summary`, `gov_sim/env/observation_builder.py` | 已符合“压缩观测”设计 | 无 |
| `E_e(theta)` | `r >= theta and online and uptime >= u_min` | `gov_sim/env/gov_env.py::_eligible_nodes` | 已符合 | 无 |
| `theta` 只控制 eligibility | `theta` 只出现在 admission 规则中 | `_eligible_nodes`, `action_mask.is_action_legal`, `heuristic_unsafe_guard_audit` | 主流程已符合；启发式 guard 只做审计，不参与 legality / reward / cost | 无 |
| 结构可行性 `|E| >= m` | 高层模板是否可执行 | `gov_sim/env/action_mask.py::is_action_legal` | 已符合 | 无 |
| 委员会 `C subset E, |C| = m` | soft sortition 无放回加权抽样 | `gov_sim/env/gov_env.py::_sample_committee`, `gov_sim/modules/committee_sampler.py` | 已符合 | 无 |
| 委员会诚实质量 `h_hat` | 委员会实现诚实比例 | `gov_sim/modules/chain_model.py::step` 中 `honest_ratio` | 已实现，但名称上还是样本均值 | 建议显式命名为 `h_hat` 或至少增加 `h_hat` 审计字段 |
| 诚实性下界 `h_LCB` | `clip(h_hat - kappa_h/sqrt(m),0,1)` | 当前无独立实现 | 经验简化：当前等价于 `kappa_h = 0` | 增加 `kappa_h` 结构参数，并显式记录 `h_LCB` |
| `unsafe` | `1[h_LCB < h_min]` | `gov_sim/modules/chain_model.py::step` 中 `unsafe = int(honest_ratio < h_min)` | 方向正确，但仍基于 `h_hat` 而非显式 `h_LCB` | 把 unsafe 的理论名义切到 `h_LCB`，并在代码中显式实现 |
| `margin_cost` | `clip((h_warn-h_LCB)/(h_warn-h_min),0,1)` | `gov_sim/env/reward_cost.py::compute_cost` | 已有连续裕度成本，但基于 `honest_ratio` 且 `h_warn` 硬编码 | 将 `h_warn`、`kappa_h` 配置化，并用 `h_LCB` 计算 |
| 时延分解 | `queue + batch + consensus (+ penalty)` | `gov_sim/modules/chain_model.py::step` | 已符合 | 无 |
| 服务量 / 吞吐 / 队列 | `S_tilde`, `S`, `TPS`, `Q_{e+1}` | `gov_sim/modules/chain_model.py::step` | 已符合 | 无 |
| timeout 与 `RTT/tau` | `L_cons > tau_view(tau)` | `gov_sim/modules/chain_model.py::step` | 已符合 | 无 |
| block slack 与 `b` | `[b-S]_+/b` | `gov_sim/env/reward_cost.py::compute_reward` | 已进入 reward，但未稳定写入审计总线 | 在 `info` 中显式导出 `block_slack` |
| `structural_infeasible` | 当前状态下无合法高层模板 | `gov_sim/env/gov_env.py::_protocol_invalid_breakdown` | 已与 unsafe 语义分离 | 无 |
| 单动作不合法 | 某个 `(m,theta,b,tau)` 不满足结构或平滑约束 | `gov_sim/env/action_mask.py::is_action_legal` | 已与 `structural_infeasible` 区分 | 无 |
| 高层更新周期 | 每隔 `update_interval` 刷新高层模板 | `gov_sim/hierarchical/spec.py`, `gov_sim/hierarchical/controller.py`, `configs/train_hierarchical_formal_final.yaml` | 已符合 | 无 |
| 场景扰动强度 | 负载 / RTT / churn / malicious burst 的工作点 | `gov_sim/modules/scenario_model.py`, `configs/default.yaml` | 已有实现，但还需要按理论可行域解释 | 审计时改用“区间折中假设”而不是全局单调 |

## 7. 当前实现逐项判断

### 7.1 已经符合理论主线的部分

1. `theta` 在主流程中只控制 eligibility。
2. 委员会只从资格集产生，且 `|C| = m`。
3. `unsafe` 与 `structural_infeasible` 已经语义分离。
4. `b` 与 `tau` 在链侧模型中体现为折中，而不是硬编码单调关系。
5. 分层更新间隔 `update_interval` 已经存在，并且确实控制高层模板刷新频率。

### 7.2 仍属于经验实现的部分

1. 安全判定当前仍直接用 `honest_ratio`，尚未显式写成 `h_LCB`。
2. `margin_cost` 当前基于 `honest_ratio`，且 `h_warn = 0.78` 硬编码在 `compute_cost` 中。
3. `block_slack` 虽然进入 reward，但没有稳定进入 `info`，导致部分审计脚本读到的是空值或默认值。
4. 现有部分耦合审计把 `theta`、`b`、`tau` 默认按“单调扫描”评价，这与理论角色不一致。

### 7.3 需要的最小修正

1. 在链侧安全审计中显式引入 `h_LCB`，即使第一版先采用 `kappa_h = 0` 也要把语义显式化。
2. 将 `h_warn` 与 `kappa_h` 作为结构参数暴露到配置层，而不是留在函数体硬编码。
3. 在 `info` 中导出 `block_slack`，并建议同时导出 `h_hat`、`h_LCB`、`margin_cost`。
4. 重写动作审计判据，停止把“theta 单调降 unsafe”或“b / tau 全局单调”作为通过标准。

## 8. 基于理论重定义实验口径

下面的实验目标不再是“扫描曲线单调不单调”，而是验证动作是否符合其理论角色。

### 8.1 `m` 维度

理论假设：

- `H_m1`：在固定 `theta, b, tau` 下，`m` 增大会近似推高 `L_cons`；
- `H_m2`：`m` 增大会近似降低结构可行性；
- `H_m3`：在候选池质量不恶化的前提下，`m` 增大会提升安全下界稳定性。

适用场景：

- 主场景：`churn_burst`
- 辅场景：`stable`

核心指标：

- `L_cons_e`
- `eligible_size`
- `executed_committee_size`
- `num_legal_high_templates` 或高层可行率
- `h_hat` / `h_LCB`
- `unsafe_rate`

通过判据：

- `m` 增大时，`L_cons` 应近似上升；
- 可行率应近似下降或不升；
- 安全侧应至少表现出“更稳的质量下界或更低的波动”，不要求 `unsafe_rate` 每个点严格单调。

失败判据：

- `m` 与 `L_cons` 无明显关系；
- `m` 增大后可行率反而系统性上升且无法解释；
- 安全边界无法通过 `h_hat/h_LCB` 解释。

### 8.2 `theta` 维度

理论假设：

- `H_theta1`：`theta` 增大时，`eligible_size` 通常下降；
- `H_theta2`：`theta` 增大时，候选池和委员会质量通常提升；
- `H_theta3`：`unsafe` 由 eligibility - quality 联合效应决定，不要求单调。

适用场景：

- 主场景：`malicious_burst`
- 辅场景：`churn_burst`

核心指标：

- `eligible_size`
- `executed_committee_size`
- `h_hat` / `h_LCB`
- `unsafe_rate`
- `structural_infeasible_rate`

通过判据：

- `eligible_size` 随 `theta` 提高而收缩；
- 委员会质量指标出现可解释变化；
- `unsafe` 的变化能够由“质量提升 vs 可行域收缩”的联合效应解释。

失败判据：

- `theta` 变化既不影响 `eligible_size`，也不影响委员会质量；
- 审计仍然只能给出“theta 越大越安全”这类口头结论。

### 8.3 `b` 维度

理论假设：

- `H_b1`：在高负载区间，增大 `b` 应缓解队列、提高 `S_e` 或 TPS；
- `H_b2`：在低负载区间，增大 `b` 会推高 `slack` 与 `L_batch`；
- `H_b3`：`b` 是区间折中，不存在全局单调优解。

适用场景：

- 主场景：`load_shock`
- 辅场景：`stable`

核心指标：

- `Q_e`, `Q_{e+1}`
- `S_e`, `TPS`
- `L_batch_e`, `L_bar_e`
- `block_slack`

通过判据：

- 在高负载阶段观察到“大 `b` 有助于消队列/提升服务能力”；
- 在低负载阶段观察到“大 `b` 提高 `slack` 或等待时延”；
- 明确存在负载区间切换下的折中或拐点。

失败判据：

- 只能得到“b 越大越好”或“b 越大越差”的单结论；
- `block_slack` 不可观测，导致折中不可解释。

### 8.4 `tau` 维度

理论假设：

- `H_tau1`：高 RTT 场景下，增大 `tau` 应降低 timeout 风险或提升成功率；
- `H_tau2`：低 RTT 场景下，增大 `tau` 会提高等待时延；
- `H_tau3`：`tau` 是 RTT 区间折中，不存在全局单调优解。

适用场景：

- 主场景：`high_rtt_burst`
- 辅场景：`stable`

核心指标：

- `timeout_failure`
- `L_batch_e`
- `L_cons_e`
- `L_bar_e`
- `TPS`

通过判据：

- 在高 RTT 区间，较大 `tau` 应显著改善 timeout 或 success；
- 在普通 RTT 区间，较大 `tau` 应带来额外等待成本；
- 结论必须以“RTT 区间折中”表达，而不是“tau 单调好/坏”。

失败判据：

- 看不到 timeout 与 `tau_view(tau)` 的联动；
- 仍把 `tau` 当成全局单调控制杆。

## 9. 训练前理论准入条件

只有当以下条件同时满足，才允许进入训练 smoke：

1. 安全、活性、性能三套语义无冲突。
   - `unsafe`、`timeout`、`structural_infeasible` 必须可区分。
2. `m/theta/b/tau` 的理论角色明确。
   - `m` 是规模 - 可行性 - 安全裕度动作；
   - `theta` 是 admission threshold；
   - `b` 是负载区间折中动作；
   - `tau` 是 RTT / timeout 区间折中动作。
3. 环境实现与理论一致。
   - 至少需要把 `h_LCB` 语义显式化；
   - 至少需要把 `block_slack` 等关键审计量显式导出。
4. 场景扰动强度在理论上合理。
   - hard scene 必须仍位于“有可行域、但存在安全与性能压力”的区间；
   - 不能通过削弱场景或压缩动作空间来制造表面可学性。
5. 审计指标能解释。
   - 结论必须能回到 `eligible_size`、委员会质量、`L_queue/L_batch/L_cons`、`timeout` 这些机制量上；
   - 不能只看 reward 曲线或单个现象图。

## 10. 当前是否允许进入训练 smoke

结论：**当前不允许进入训练 smoke。**

原因不是主线结构错误，而是还存在三个训练前缺口：

1. 安全侧缺少显式 `h_LCB` 语义，unsafe / margin_cost 仍是经验近似；
2. 审计总线缺少 `block_slack` 等关键解释量；
3. 现有部分动作审计仍按单调扫描口径写结论，和理论角色不一致。

## 11. 训练前最小必要修正项

最小必要修正，不涉及压缩动作空间，也不涉及重写主线：

1. 在链侧安全模块中显式实现并记录 `h_LCB`，同时把 `h_warn`、`kappa_h` 变成结构参数。
2. 在环境 `info` 中显式导出 `block_slack`，并建议同步导出 `h_hat`、`h_LCB`、`margin_cost`。
3. 把动作审计脚本的通过条件从“单调性检查”改成“理论角色检查”。
4. 以理论可行域为前提重新校准 hard scene 扰动强度，而不是继续无约束扫参数。

