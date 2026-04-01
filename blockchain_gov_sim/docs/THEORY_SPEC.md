# THEORY_SPEC

本规格书只对应第四章“IoV/VEC 联盟链链侧分层动态治理”，不涉及 DAG 卸载执行层。其目的不是给工作点找借口，而是先固定理论对象、变量语义、约束关系与时间尺度，再要求代码逐条实现。

本规格综合三类文献中的稳定共识：

- IoV/IoT 信任管理：多维信誉、折扣证据、冷启动门控、不确定性惩罚、推荐攻击抑制。
- 联盟链 / BFT 委员会治理：资格筛选、委员会抽样、委员会质量下界、安全与时延的结构性权衡。
- Safe RL / CMDP / 两时标控制：主任务回报与约束成本分离，慢时标决策对应 chunk / option 级 transition。

## A. 系统对象

### A.1 网络对象

- 车辆：轻节点，请求与证据来源，不直接参与链上治理。
- RSU：联盟链全节点，也是候选治理节点。
- 一个治理周期记为 `e`。
- RSU 总集合记为 `N={1,...,n}`。

### A.2 分层动作

- 高层动作 `a_e^H = (m_e, theta_e)`。
  - `m_e`：目标委员会规模。
  - `theta_e`：资格阈值，只承担 admission control 语义。
- 低层动作 `a_e^L = (b_e, tau_e)`。
  - `b_e`：目标区块 / 批大小。
  - `tau_e`：批等待上限，也是 view timeout 的基准量。

### A.3 时间尺度

- 低层每一步对应一个原子治理周期。
- 高层每一步对应一个长度至多为 `K` 的 chunk。
- 一个高层动作在 chunk 内保持不变，低层动作在 chunk 内逐步执行。

## B. 多维信誉

### B.1 维度

对每个 RSU `j`，维护四维信誉：

- `svc`：服务执行可靠性。
- `con`：共识 / 投票可靠性。
- `rec`：推荐可信性。
- `stab`：在线稳定性。

### B.2 折扣伪计数

对维度 `d in {svc,con,rec,stab}`，设本周期软证据为 `y_{j,d} in [0,1]`、证据权重为 `omega_{j,d} > 0`，则

`alpha'_{j,d} = rho_d * alpha_{j,d} + omega_{j,d} * y_{j,d}`

`beta'_{j,d} = rho_d * beta_{j,d} + omega_{j,d} * (1 - y_{j,d})`

其中：

- `rho_d in (0,1]` 是历史折扣。
- `omega_{j,d}` 可以由多条同周期证据累加得到。

### B.3 后验统计量

`mu_{j,d} = alpha'_{j,d} / (alpha'_{j,d} + beta'_{j,d})`

`var_{j,d} = alpha'_{j,d} * beta'_{j,d} / ((alpha'_{j,d}+beta'_{j,d})^2 * (alpha'_{j,d}+beta'_{j,d}+1))`

`n_{j,d} = alpha'_{j,d} + beta'_{j,d}`

### B.4 冷启动门控与不确定性惩罚

IoV 信任管理中常见做法不是直接使用 `mu`，而是同时考虑冷启动与统计不确定性。故定义：

`kappa_{j,d} = 1 - exp(-n_{j,d} / N0)`

`score_dim_{j,d} = kappa_{j,d} * mu_{j,d} - lambda_d * sqrt(var_{j,d})`

含义：

- `kappa` 防止新节点因样本太少而被高估。
- `lambda_d * sqrt(var)` 抑制高方差节点。

### B.5 上下文感知融合

每个周期构造上下文向量：

`c_e = [arrival, queue, dqueue, rtt, drtt, churn, eligible_prev, honest_lcb_prev, unsafe_prev, timeout_prev]`

对标准化后的 `c_e`，定义四维权重：

`w_d(e) = softmax(eta_d^T c_e)`

最终基础信誉为：

`T_base(j,e) = sum_d w_d(e) * score_dim_{j,d}(e)`

### B.6 攻击惩罚

基于 IoV 信任文献中的常见攻击类型，本模型显式考虑：

- on-off
- zigzag
- bad-mouthing / ballot-stuffing 的推荐偏置后果
- collusion
- 稳定性不足

这些攻击不必都做成独立状态变量，但其效果必须通过惩罚项进入最终信誉：

`final_score_j(e) = clip01(T_base(j,e) - gamma_onoff * p_onoff - gamma_zigzag * p_zigzag - gamma_rec * p_rec_bias - gamma_col * p_collusion - gamma_stab * p_stability)`

其中 bad-mouthing、ballot-stuffing 可以作为推荐矩阵污染机制进入 `p_rec_bias` / `p_collusion`，不要求额外再定义一套独立最终分数。

## C. 资格集—委员会—安全性三段式

### C.1 资格集

`E_e(theta_e) = { j in N : final_score_j(e) >= theta_e, uptime_j(e) >= u_min, online_j(e)=1 }`

这里 `theta` 只控制 admission control。它不直接进入：

- `unsafe` 判定公式
- `margin_cost` 判定公式
- 链侧时延结构方程

因此不能把 `theta` 口头化为“直接安全旋钮”。

### C.2 结构可行性

对一个给定高层动作 `(m_e, theta_e)`，其结构可行性条件为：

`|E_e(theta_e)| >= m_e`

若对该动作不满足该条件，则该动作结构不可行。

对整个高层动作空间，若当前没有任何高层模板可执行，则记为当前状态 `structural_infeasible=1`。它表示“无合法模板”，不是安全事件。

### C.3 委员会抽样

在结构可行时，从 `E_e(theta_e)` 中抽样委员会 `C_e`：

`C_e subset E_e(theta_e), |C_e| = m_e`

抽样采用软抽签而非 Top-K：

`P(j selected | E_e) propto exp(beta_s * final_score_j(e))`

并且采用无放回加权采样。

### C.4 委员会质量下界

对被选中的委员会节点，定义其 `con` 维基础质量为 `con_score_j(e)`。本规格要求：

`con_score_j(e) = score_dim_{j,con}(e)`

则委员会诚实性下界定义为：

`h_LCB(e) = mean_{j in C_e} [ con_score_j(e) - lambda_h * sqrt(var_{j,con}(e)) ]`

最后裁剪到 `[0,1]`。

解释：

- `score_dim_con` 已经包含冷启动门控与维度级不确定性惩罚；
- `lambda_h` 额外表达“把节点级 `con` 质量映射为委员会级保守下界”时的安全保守度。

### C.5 unsafe 与 margin_cost

定义：

`unsafe_e = 1[h_LCB(e) < h_min]`

`margin_cost_e = clip((h_warn - h_LCB(e)) / (h_warn - h_min), 0, 1)`

因此：

- `unsafe` 必须由 selected committee 的 `h_LCB` 决定；
- `theta` 只能通过改变资格集和委员会组成，间接影响 `h_LCB`；
- `theta -> unsafe` 是联合效应，不要求单调。

### C.6 BFT 安全边界的抽象语义

在 BFT 文献中，安全性依赖于委员会内诚实份额能否支撑法定人数 / 超多数阈值。仿真中的 `h_min` 不是某篇文献直接给出的物理常数，而是把“满足 BFT 安全边界所需的最小诚实份额”抽象为一个可校准阈值。因此：

- `h_min` 具有明确理论角色；
- 但其数值不是文献常数，而是工作点校准参数。

## D. 性能模型

### D.1 基础量

`X_e = queue_e + arrivals_e`

### D.2 等待成批

`wait_to_fill_e = min(tau_e, max(0, (b_e - X_e) / (arrival_hat_e + eps)))`

`B_e = min(b_e, X_e + arrival_hat_e * wait_to_fill_e)`

### D.3 有效服务率

`mu_eff_e = mu0`

`          * exp(-a_r * RTT_e - a_chi * churn_e)`

`          * exp(-a_m * m_e * (m_e-1) / 2)`

`          * 1 / (1 + a_b * B_e / B_ref)`

`          * 1[|E_e(theta_e)| >= m_e]`

### D.4 服务量与队列

`S_e = min(X_e, B_e, mu_eff_e * delta_t)`

`queue_{e+1} = max(0, X_e - S_e)`

### D.5 时延分解

`L_queue_e = queue_{e+1} / (mu_eff_e + eps)`

`L_batch_e = wait_to_fill_e / 2`

`L_cons_e = RTT_e * (c0 + c1*m_e + c2*m_e*(m_e-1)/2) + c3 * B_e / B_ref`

`tau_view_e = max(tau_min, tau_view_factor * tau_e)`

`timeout_e = 1[L_cons_e > tau_view_e]`

`success_e = 1[|E_e(theta_e)| >= m_e and h_LCB(e) >= h_min and L_cons_e <= tau_view_e]`

`total_latency_e = L_queue_e + L_batch_e + L_cons_e + (1-success_e) * L_pen`

### D.6 四个动作的作用层次

- `m`：
  - 直接进入委员会规模和共识复杂度；
  - 影响 BFT 代表性 / capture 风险；
  - 近似单调地提高 `L_cons`；
  - 同时改变结构可行域。
- `theta`：
  - 只进入 `E(theta)`；
  - 通过资格集规模与质量共同影响委员会质量；
  - 对 `unsafe` 不要求单调。
- `b`：
  - 主要影响 `wait_to_fill`、`B_e`、`block_slack`、`served`；
  - 是负载区间折中变量，不是全局单调变量。
- `tau`：
  - 主要影响 `wait_to_fill` 与 `tau_view`；
  - 是 RTT / timeout 折中变量，不是全局单调变量。

## E. CMDP

### E.1 原子周期 reward / cost

`r_e = lambda1 * log(1 + S_e) - lambda2 * total_latency_e - lambda3 * queue_{e+1} - lambda4 * smooth(a_e, a_{e-1}) - lambda5 * block_slack_e`

`c_e = unsafe_e + w_timeout * timeout_e + w_margin * margin_cost_e`

约束成本不能被并入 reward；必须保留显式 CMDP 结构。

### E.2 高层 chunk 语义

高层一次动作对应一个 chunk `g={e, e+1, ..., e+K'-1}`，其中 `K'<=K`。

高层 reward / cost 训练目标为折扣 chunk return：

`R_g^H = sum_{i=0}^{K'-1} gamma^i * r_{e+i}`

`C_g^H = sum_{i=0}^{K'-1} gamma^i * c_{e+i}`

高层 `next_obs` 必须取 chunk 结束后的高层状态。

### E.3 高层约束尺度

若 `cost_limit` 的理论语义定义在“原子治理周期平均成本”上，而高层训练用的是 chunk 折扣和，则 dual update 必须使用归一化后的 chunk 成本：

`S_gamma(K) = sum_{i=0}^{K-1} gamma^i = (1-gamma^K)/(1-gamma)`

`C_norm^H = C_g^H / S_gamma(K)`

于是：

- high critic / cost advantage 可以继续学习 `C_g^H`；
- dual update / constraint violation 使用 `C_norm^H - cost_limit`；
- 这样 `cost_limit` 保持“原子周期平均约束”的旧语义。

## F. 参数分类

### F.1 理论语义参数

下列符号有明确模型意义，必须先定义语义，后谈数值：

- `rho_d`：信誉历史折扣。
- `N0`：冷启动证据门控尺度。
- `lambda_d`：维度级不确定性惩罚强度。
- `eta_d`：上下文权重映射。
- `beta_s`：软抽签温度。
- `u_min`：资格稳定性门槛。
- `mu0, a_r, a_chi, a_m, a_b, c0, c1, c2, c3`：链侧服务 / 共识结构参数。
- `K`：高层 chunk 长度。
- `gamma`：高层折扣因子。

### F.2 工作点校准参数

下列量不能被当成文献常数，只能解释为仿真工作点：

- `h_min`
- `h_warn`
- `lambda_h`
- `cost_weights.timeout`
- `cost_weights.margin`
- hard scene 的具体扰动强度
- `cost_limit`
- reward / cost 权重的具体数值

它们可以具有理论角色，但其具体数值只能来自工作点校准，不能冒充文献定值。

## G. 本规格的硬约束

后续任何代码与实验都必须满足：

1. `theta` 只控制 eligibility，不直接定义 `unsafe`。
2. `unsafe` 只能由 selected committee 的 `h_LCB` 判定。
3. `structural_infeasible` 与 `unsafe`、`timeout` 语义分离。
4. `b`、`tau` 只作为性能折中变量，不预设全局单调。
5. 高层一次动作必须对应一个 chunk transition。
6. high-level dual update 的成本尺度必须与 `cost_limit` 语义一致。
