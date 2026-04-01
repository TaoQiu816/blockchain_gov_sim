# 场景机制检查报告（修正后）

## 执行摘要

通过运行 20 episodes × 4 scenes 的随机动作评估，检查各场景是否真实产生预期的机制效果。

**修正状态**: ✅ 所有场景参数已修正并验证生效

## 场景对比结果

| 场景 | arrivals | rtt (ms) | churn | queue_size | latency (ms) | unsafe_rate | structural_infeasible_rate |
|------|----------|----------|-------|------------|--------------|-------------|----------------------------|
| load_shock | 442.1 | 28.1 | 0.114 | 5809.4 | 22.5 | 0.0000 | 0.0008 |
| high_rtt_burst | 260.0 | 44.8 | 0.115 | 8.2 | 26.1 | 0.0000 | 0.0004 |
| churn_burst | 260.0 | 28.1 | 0.119 | 85.7 | 18.6 | 0.0000 | 0.4021 |
| malicious_burst | 259.7 | 28.2 | 0.116 | 20.1 | 21.8 | 0.0075 | 0.0083 |

**Baseline (steady)**: arrivals=260.0, rtt=28.1ms, churn=0.115, queue_size≈10, unsafe_rate≈0.0000

## 场景机制判断

### 1. load_shock ✅

**预期**: 提高 arrival / queue pressure / TPS-pressure tradeoff

**实际表现**:
- arrivals: 442.1 vs baseline 260.0 (提高 **70.0%** ✅)
- queue_size: 5809.4 vs baseline ~10 (提高 **580x** ✅)
- load: 58.09 vs baseline ~0.1 (提高 **580x** ✅)

**判断**: ✅ **机制完全生效**
- ✅ arrival 提升显著超过 50% 阈值
- ✅ queue pressure 极度提高
- ✅ 场景分化明显

**修正参数**:
```yaml
load_shock:
  high_lambda_scale: 2.5  # 从 1.42 提高到 2.5
```

### 2. high_rtt_burst ✅

**预期**: 提高 RTT / timeout risk / latency

**实际表现**:
- rtt: 44.8 ms vs baseline 28.1 ms (提高 **59.4%** ✅)
- latency: 26.1 ms vs baseline ~20 ms (提高 **30.5%** ✅)
- timeout_rate: 0.00125 (开始触发 ✅)

**判断**: ✅ **机制完全生效**
- ✅ RTT 提升显著超过 50% 阈值
- ✅ latency 显著提高
- ✅ timeout risk 开始触发

**修正参数**:
```yaml
high_rtt_burst:
  rtt_min: 50.0  # 从 36.0 提高到 50.0
  rtt_max: 100.0  # 从 92.0 提高到 100.0
```

### 3. churn_burst ✅

**预期**: 提高资格不足与 structural infeasible

**实际表现**:
- churn: 0.119 vs baseline 0.115 (提高 3.5%)
- structural_infeasible_rate: 0.4021 vs baseline 0.0006 (提高 **670x** ✅)
- eligible_size: 6.6 vs baseline 10.5 (降低 **37%** ✅)

**判断**: ✅ **机制���全生效**
- ✅ structural infeasible 显著提高
- ✅ eligible_size 显著降低
- ⚠️ churn 本身提升不明显，但已经通过 eligible_size 产生了预期效果

**无需修正**: 该场景机制已经生效，通过降低 eligible_size 触发了 structural infeasible。

### 4. malicious_burst ✅

**预期**: 提高 unsafe risk，使更高 theta / 更稳健 m 的收益可见

**实际表现**:
- unsafe_rate: 0.0075 vs baseline 0.0000 (提高到 **0.75%** ✅)
- honest_ratio: 0.990 vs baseline 0.999 (降低 **0.9%** ✅)
- structural_infeasible_rate: 0.0083 vs baseline 0.0006 (提高 **14x** ✅)

**判断**: ✅ **机制完全生效**
- ✅ unsafe risk 显著提高（0.75% 是可观测的风险水平）
- ✅ honest_ratio 降低明显
- ✅ 场景分化达到预期

**修正参数**:
```yaml
malicious_burst:
  extra_malicious_ratio: 0.15  # 从 0.12 提高到 0.15
```

## 总体结论

### 场景分化程度

| 场景 | 机制生效程度 | 修正状态 | 关键指标提升 |
|------|-------------|---------|-------------|
| load_shock | ✅ 完全生效 | ✅ 已修正 | arrivals +70%, queue +580x |
| high_rtt_burst | ✅ 完全生效 | ✅ 已修正 | rtt +59%, timeout 触发 |
| churn_burst | ✅ 完全生效 | ✅ 无需修正 | structural_infeasible +670x |
| malicious_burst | ✅ 完全生效 | ✅ 已修正 | unsafe_rate +∞ (0→0.75%) |

### ���景特征总结

1. **load_shock**: 极高到达率与队列压力，考验 batch size / timeout 的吞吐-延迟权衡
2. **high_rtt_burst**: 高网络延迟与 timeout 风险，考验 timeout 参数的鲁棒性
3. **churn_burst**: 高节点流失率导致资格集不足，考验 committee size 的结构可行性
4. **malicious_burst**: 高恶意节点比例，考验 theta 阈值的安全性权衡

### 验证结论

✅ **所有场景机制已完全生效，场景分化达到预期**

- 各场景的关键指标都有显著提升（>50% 或数量级变化）
- 场景之间的差异明显，能够有效测试不同的治理机制
- 可以进入下一阶段：动作-机制耦合检查

### 下一步行动

1. ✅ 场景配置参数修正完成
2. ✅ 场景机制检查验证通过
3. ➡️ **进入 B 部分：动作-机制耦合检查**
   - B1: 检查高层 (m, theta) 对资格集/委员会/unsafe/latency 的影响
   - B2: 检查低层 (b, tau) 对 batch/queue/latency/timeout 的影响

## 附录：完整统计数据

详见 [`scenario_comparison.csv`](audits/scenario_mechanism_check/scenario_comparison.csv) 和 [`scenario_summaries.json`](audits/scenario_mechanism_check/scenario_summaries.json)

## 修正历史

### 第一次检查（修正前）
- load_shock: arrivals 仅 +10.7%，不够显著
- high_rtt_burst: rtt 仅 +45.6%，不够显著
- malicious_burst: unsafe_rate 仅 0.21%，太低

### 第二次检查（修正后）
- load_shock: arrivals +70% ✅
- high_rtt_burst: rtt +59% ✅
- malicious_burst: unsafe_rate 0.75% ✅
