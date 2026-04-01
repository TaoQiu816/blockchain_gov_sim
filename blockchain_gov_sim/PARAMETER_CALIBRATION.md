# 第四章参数体系标定

> 注：当前链侧分层动态治理的正式理论口径，以
> `docs/CHAIN_SIDE_LAYERED_DYNAMIC_GOVERNANCE_THEORY.md`
> 为准。本文档保留历史标定记录，不再作为“是否可以进入训练”的最终依据。

仓库路径：`/Users/qiutao/研/毕设/毕设/blockchainSim/blockchain_gov_sim`

## 1. 标定目标

本次标定围绕第四章“联盟链治理仿真”进行，目标不是追求某篇文献的逐参数复刻，而是基于：

1. 参考文献中的典型 IoV / PBFT / 信誉共识设置；
2. 当前实现中的状态、动作、时延、安全与信誉耦合方式；
3. `Static-Param`、`Heuristic-AIMD`、`Ours-short` 的 pilot 运行结果；

为主实验确定一个“非退化默认工作点”。

## 2. 参考文献依据

本次标定主要参考以下文献方向：

- [Consensus on the Internet of Vehicles: A Systematic Literature Review](https://www.mdpi.com/2032-6653/16/11/616)
  结论：IoV 场景下联盟链 / PBFT 及其变体是低时延可信治理的主流路线，小规模、受信 RSU 集合较常见。
- [SG-PBFT: a Secure and Highly Efficient Blockchain PBFT Consensus Algorithm for Internet of Vehicles](https://arxiv.org/abs/2101.01306)
  结论：IoV 中 PBFT 需要通过分组、信誉或节点筛选控制通信开销，说明“全体节点直接共识”并不是合理默认工作点。
- [A Scalable and Trust-Value-Based Consensus Algorithm for Internet of Vehicles](https://www.mdpi.com/2076-3417/13/19/10663)
  结论：RSU 在每周期收集行为反馈并更新信誉，再将结果打包上链，这与当前实现的“证据 -> 信誉 -> 资格集 -> 共识”闭环一致。
- [A Blockchain-Enabled Incentive Trust Management with Threshold Ring Signature Scheme for Traffic Event Validation in VANETs](https://pmc.ncbi.nlm.nih.gov/articles/PMC9460646/)
  结论：RSU 同时承担消息验证、信任值计算与 PBFT 出块职责，证明“车辆轻节点 + RSU 全节点治理”建模是合理的。
- [BARS: a Blockchain-based Anonymous Reputation System for Trust Management in VANETs](https://arxiv.org/abs/1807.06159)
  结论：信誉与区块链治理在 VANET/IoV 中应联合设计，而不是把信誉仅作为离线统计量。

文献并不会直接给出本仿真所需的每一个数值，因此正式参数采用“文献范围 + pilot 校准”的组合方式确定。

## 3. 参数分类

### 3.1 固定结构参数

这些参数由研究方案定义，不属于默认工作点标定：

- 信誉维度：`svc / con / rec / stab`
- 动作空间：`a=(m,b,tau,theta)`，共 `400` 个联合离散动作
- 委员会大小候选：`{7, 9, 11, 13, 15}`
- 区块大小候选：`{128, 256, 384, 512}`
- 批等待超时候选：`{20, 40, 60, 80} ms`
- 门槛候选：`{0.40, 0.50, 0.60, 0.70, 0.80}`
- 主算法：`Maskable PPO-Lagrangian`
- 主方案委员会机制：`soft sortition`

### 3.2 默认工作点参数

这些参数需要在“既不退化、又不过载”的范围内标定：

- `N_R = num_rsus`
- `p_m = malicious_ratio`
- `RTT = [rtt_min, rtt_max]`
- `p_off / p_on`
- `lambda`（平稳负载工作点）
- `rho_svc / rho_con / rho_rec / rho_stab`
- `N0`
- `u_min / h_min`
- `gamma_sc / gamma_on / gamma_rec / gamma_col / gamma_stab`
- `Static-Param` 的固定默认动作

### 3.3 实验横轴扫描参数

这些参数保留给 benchmark / ablation：

- `malicious_ratio ∈ {0, 0.1, 0.2, 0.3, 0.4}`
- `on_off_period ∈ {4, 8, 12}`
- `zigzag_freq ∈ {0.08, 0.18, 0.32}`
- `collusion_group_size ∈ {2, 3, 5}`
- `step load k ∈ {1.5, 2.0, 2.5}`
- `RTT_max ∈ {55, 70, 90} ms`
- `p_off ∈ {0.06, 0.10, 0.14}`
- `p_on ∈ {0.30, 0.25, 0.18}`

## 4. Pilot 结果

### 4.1 第一轮粗扫

候选点：

- `c1_mild`: `N_R=25, p_m=0.20, lambda=180, RTT=[8,45], p_off=0.05, p_on=0.35`
- `c2_balanced`: `N_R=27, p_m=0.25, lambda=260, RTT=[12,35], p_off=0.08, p_on=0.24`
- `c3_stress`: `N_R=29, p_m=0.30, lambda=320, RTT=[18,50], p_off=0.10, p_on=0.20`

结论：

- `c1` 太轻：队列几乎长期为空，unsafe 恒 0，不适合作为正式工作点。
- `c3` 太重：baseline 队列普遍爆炸，资格规模逼近下边界。
- `c2` 最接近平衡，但 unsafe 仍偏低，需要进一步微调。

### 4.2 第二轮窄范围微调

最终选中的 pilot 工作点：

- `N_R = 27`
- `p_m = 0.32`
- `u_min = 0.58`
- `h_min = 0.68`
- `lambda = 260`
- `RTT = [14, 42] ms`
- `p_off = 0.08`
- `p_on = 0.22`
- `rho = {svc:0.95, con:0.96, rec:0.94, stab:0.97}`
- `N0 = 8`
- `Static-Param = (11, 256, 40, 0.55)`

在该工作点上的 pilot 结果：

| Controller | unsafe_rate | queue_peak | eligible_size_mean | TPS | mean_latency |
|---|---:|---:|---:|---:|---:|
| Static-Param | 0.0000 | 7635 | 11.98 | 6551.89 | 30.19 |
| Heuristic-AIMD | 0.0083 | 8403 | 12.42 | 5592.53 | 31.28 |
| Ours-short | 0.0083 | 256 | 12.56 | 12455.23 | 24.75 |

解释：

- 队列不再长期为 0，说明负载已经进入“治理参数会影响系统状态”的区域。
- 队列也没有在 `Ours-short` 下长期爆炸，说明环境没有进入纯不可控区。
- `unsafe_rate` 变成“低但非零”，适合作为正式工作点。
- `eligible_size_mean ≈ 12.5`，对动作集合中的 `m ∈ {7,9,11,13,15}` 来说，已经能稳定出现合法/非法动作混合。
- `TPS / latency` 在不同策略下明显拉开，说明该工作点有足够辨识度。

### 4.3 固定动作可分辨性检查

在同一工作点下，使用三组固定动作额外检查：

| Static Action | TPS | mean_latency | queue_peak | unsafe_rate |
|---|---:|---:|---:|---:|
| conservative `(15,128,80,0.7)` | 5916.80 | 30.91 | 5698 | 0.0125 |
| mid `(11,256,40,0.55)` | 7350.48 | 27.92 | 4802 | 0.0000 |
| aggressive `(7,512,20,0.4)` | 13600.91 | 32.23 | 497 | 0.0125 |

结论：该工作点下，不同动作能够显著拉开 TPS / latency / queue / unsafe，满足后续 RL 学习与对比需求。

## 5. 最终推荐参数表

### 5.1 默认工作点

| 参数 | 推荐值 | 说明 |
|---|---:|---|
| `N_R` | `27` | 小规模联盟链，兼顾 Top-K/软抽签资格规模 |
| `p_m` | `0.32` | 使安全事件低但非零 |
| `RTT` | `[14, 42] ms` | 对应较真实的 V2I / RSU 间低到中等延迟 |
| `p_off / p_on` | `0.08 / 0.22` | 维持适度 churn，不至于完全稳定 |
| `lambda` | `260` | 使队列进入非平凡区间 |
| `rho_svc` | `0.95` | 服务维证据衰减略快 |
| `rho_con` | `0.96` | 共识维更稳定 |
| `rho_rec` | `0.94` | 推荐维易受污染，衰减略快 |
| `rho_stab` | `0.97` | 稳定性维惯性最大 |
| `N0` | `8.0` | 冷启动不应过快放开 |
| `gamma_sc` | `0.35` | 跨维伪装抑制最强 |
| `gamma_on` | `0.30` | on-off/zigzag 次强 |
| `gamma_rec` | `0.25` | 推荐偏差中等抑制 |
| `gamma_col` | `0.22` | 合谋抑制中等 |
| `gamma_stab` | `0.18` | 稳定性惩罚较轻，避免过度筛空资格集 |

### 5.2 关于“svc/con 内部权重”

当前实现中**不存在独立的 `svc/con` 二级内部子权重超参数**。这是有意设计：

- 维度间权重由 `w_d(e)` 动态产生；
- 它依赖 `kappa / sigma^2` 与上下文门控，而不是一个人工写死的固定 `svc:con` 比例。

因此本次标定**不新增一个代码中无效的固定权重参数**。如果后续论文需要“二级内部权重”，那属于模型扩展，不属于本轮标定。

## 6. 正式训练建议

推荐正式训练配置：

- `learning_rate = 2.5e-4`
- `total_timesteps = 100000`
- `n_steps = 512`
- `batch_size = 128`
- `n_epochs = 8`
- 随机种子数：`3` 个起步，建议 `[42, 52, 62]`

原因：

- pilot 下 `3e-4` 能跑通，但正式实验更建议略降到 `2.5e-4` 以减小动作震荡；
- `100k` 步对当前环境规模已能明显超过 smoke / short-train 区间；
- `3` 个种子是硕士论文阶段较常见的最低可接受重复次数。

## 7. 配置落地

本次已回写：

- `configs/default.yaml`
  写入默认工作点、扫描参数分类与注释。
- `configs/train.yaml`
  写入正式训练超参数建议与 formal seed list。

辅助验证配置：

- `configs/audit_train.yaml`
- `configs/audit_eval.yaml`

## 8. 是否可以开始正式实验

结论：**可以开始正式实验。**

建议顺序：

1. 用 `configs/train.yaml` 跑 3 个正式种子。
2. 固化主模型后再跑 benchmark / ablation。
3. 把 `outputs/train/train_formal_*`、`outputs/benchmark/*` 和 `outputs/ablation/*` 作为论文作图源目录冻结。
