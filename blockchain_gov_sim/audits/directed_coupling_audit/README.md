# 定向耦合审计 - 按机制对应场景验证

## 背景

之前在 stable 场景下的耦合检查可能误判了低层动作机制。本次审计在每个动作的主导场景中验证其折中关系。

## 审计设计

### 核心原则

1. **场景匹配**: 每个动作在其最相关的场景中测试
2. **多 seed 验证**: 每个配置运行 3 个 seeds，每个 seed 10 episodes
3. **统计显著性**: 导出 mean/std，确保结果可靠
4. **真实运行**: 基于实际环境运行，不简化模型

### 审计内容

#### 1. b 扫描 —— load_shock 场景

**固定参数**: m=7, theta=0.50, tau=60  
**扫描范围**: b ∈ {256, 320, 384, 448, 512}  
**关键指标**: queue_size, TPS, latency, block_slack, batch_fill_ratio

**预期折中关系**:
- b ↑ → queue_size ↓ (更大区块处理更多交易)
- b ↑ → TPS ↑ (吞吐量提升)
- b ↑ → latency ↑ (但增幅较小，因为批处理效率提升)

#### 2. tau 扫描 —— high_rtt_burst 场景

**固定参数**: m=7, theta=0.50, b=384  
**扫描范围**: tau ∈ {40, 60, 80, 100} ms  
**关键指标**: timeout_rate, latency, confirm_latency, queue_size, served_count

**预期折中关系**:
- tau ↑ → timeout_rate ↓ (更长超时减少失败)
- tau ↑ → latency ↑ (等待时间增加)
- 在高 RTT 场景下，tau 的作用更明显

#### 3. theta 扫描 —— malicious_burst 场景

**固定参数**: m=7, b=384, tau=60  
**扫描范围**: theta ∈ {0.45, 0.50, 0.55, 0.60}  
**关键指标**: unsafe_rate, committee_honest_ratio, eligible_size, structural_infeasible_rate

**预期折中关系**:
- theta ↑ → unsafe_rate ↓ (更严格筛选提高安全性)
- theta ↑ → committee_size ↓ (合格节点减少)
- theta ↑ → eligible_size ↓ (资格集缩小)

#### 4. m 扫描 —— churn_burst 场景

**固定参数**: theta=0.50, b=384, tau=60  
**扫描范围**: m ∈ {5, 7, 9}  
**关键指标**: committee_size, structural_infeasible_rate, latency, unsafe_rate

**预期折中关系**:
- m ↑ → committee_size ↑ (直接决定)
- m ↑ → structural_infeasible_rate ↓ (更容易满足)
- 在 churn 场景下，m 的稳定性作用更明显

## 审计结果

### 通过情况

| 审计项 | 场景 | 结果 | 相关系数 | 判据 |
|--------|------|------|----------|------|
| b → queue_size | load_shock | ✅ 通过 | -1.000 | 完美负相关 |
| b → TPS | load_shock | ✅ 通过 | +1.000 | 完美正相关 |
| tau → timeout_rate | high_rtt_burst | ✅ 通过 | -0.775 | 强负相关 |
| **theta → unsafe_rate** | **malicious_burst** | **❌ 未通过** | **+0.536** | **正相关（预期负）** |
| theta → committee_size | malicious_burst | ✅ 符合 | -0.929 | 强负相关 |
| m → committee_size | churn_burst | ✅ 通过 | +0.989 | 近乎完美 |

### 关键发现

#### ✅ 成功验证的机制

1. **b (区块大小) 在 load_shock 场景**
   - queue_size: 7055 → 2995 (降低 57.6%)
   - TPS: 10049 → 16158 (提升 60.8%)
   - 完美的吞吐/延迟权衡

2. **tau (批次超时) 在 high_rtt_burst 场景**
   - timeout_rate: 0.36% → 0% (tau=40 → 60)
   - 在高 RTT 环境下有效降低超时

3. **m (委员会大小) 在 churn_burst 场景**
   - committee_size: 3.63 → 5.29 (m=5 → 9)
   - 直接决定委员会规模，符合预期

#### ⚠️ 需要关注的问题

**theta (信任阈值) 在 malicious_burst 场景**

- **问题**: unsafe_rate 随 theta 增加而增加（相关系数 +0.536）
- **预期**: theta ↑ → unsafe_rate ↓
- **观察数据**:
  - theta=0.45: unsafe_rate=0.33%
  - theta=0.50: unsafe_rate=0.14%
  - theta=0.55: unsafe_rate=0.22%
  - theta=0.60: unsafe_rate=0.56%

- **可能原因**:
  1. theta 过高导致 eligible_size 过小（10.8 → 9.1）
  2. 委员会规模不足（6.94 → 6.26），增加结构不可行性
  3. 在 malicious_burst 场景下，过度筛选反而降低了系统活性
  4. 安全机制可能需要与委员会规模联合考虑

- **committee_size 趋势正常**: 6.94 → 6.26（负相关 -0.929）
- **honest_ratio 保持高位**: 0.987 → 0.992

## 判据与建议

### 总体判据

**所有审计通过**: ❌ 否  
**允许模型修改**: ⚠️ 是（针对 theta-unsafe 机制）

### 具体建议

#### 1. 对于 b, tau, m 机制

✅ **不建议修改模型**
- 这些机制在对应场景中表现符合预期
- 折中关系清晰且可控
- 可以继续使用当前实现

#### 2. 对于 theta 机制

⚠️ **建议检查或允许调整**

**选项 A: 检查安全机制实现**
- 验证 unsafe 判定逻辑是否正确
- 检查 h_min 阈值设置（当前 0.68）
- 确认 malicious_burst 场景的恶意节点注入是否正确

**选项 B: 调整 theta 与 m 的联合约束**
- 考虑 theta 过高时自动降低 m 的要求
- 或者在 theta 过高时放宽 structural_infeasible 判定
- 引入动态安全阈值

**选项 C: 接受当前行为**
- 如果认为"过度筛选降低活性"是合理的系统特性
- 将其作为 theta 的上界约束
- 在训练中让策略学习避免极端 theta 值

### 不允许的修改

根据任务约束，以下修改**不被允许**:
- ❌ 简化动作空间
- ❌ 弱化 hard 场景（如 malicious_burst）
- ❌ 大幅修改 chain_model 核心逻辑
- ❌ 进入正式训练前不验证

### 允许的调整

如果选择修改，以下调整**可以考虑**:
- ✅ 调整 h_min 阈值（当前 0.68）
- ✅ 优化 unsafe 判定逻辑
- ✅ 引入 theta-m 联合约束
- ✅ 调整 malicious_burst 场景参数
- ✅ 增加中间指标监控

## 数据文件

- [`directed_coupling_b_scan.csv`](directed_coupling_b_scan.csv): b 扫描详细数据
- [`directed_coupling_tau_scan.csv`](directed_coupling_tau_scan.csv): tau 扫描详细数据
- [`directed_coupling_theta_scan.csv`](directed_coupling_theta_scan.csv): theta 扫描详细数据
- [`directed_coupling_m_scan.csv`](directed_coupling_m_scan.csv): m 扫描详细数据
- [`directed_coupling_full_results.json`](directed_coupling_full_results.json): 完整结果（含 raw 数据）
- [`judgment.json`](judgment.json): 判据详情
- [`DIRECTED_COUPLING_REPORT.md`](DIRECTED_COUPLING_REPORT.md): 格式化报告

## 运行方式

```bash
# 从项目根目录运行
PYTHONPATH=. python audits/directed_coupling_audit/run_directed_coupling_audit.py
```

## 结论

本次定向耦合审计基于真实运行证据，在每个动作的主导场景中验证了折中关系：

1. **b, tau, m 机制**: 验证通过，不建议修改
2. **theta 机制**: 在 malicious_burst 场景下表现异常，建议检查或调整
3. **总体建议**: 允许针对 theta-unsafe 机制进行有限调整，但需保持其他机制不变

这为后续"是否允许改模型"提供了明确的判据依据。
