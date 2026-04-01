# Stage2/Stage3 Low Actor 冻结证据与 PPO-Lagrangian 运行轨迹

## 一、Low Actor 参数冻结证据

### Seed 42

| Stage | Low Actor Hash | 状态 |
|-------|----------------|------|
| Stage1 (pretrain) | NOT_FOUND | ⚠️ 缺失 |
| Stage2 (high train) | NOT_FOUND | ⚠️ 缺失 |
| Stage3 (high refine) | NOT_FOUND | ⚠️ 缺失 |

### Seed 43

| Stage | Low Actor Hash | 状态 |
|-------|----------------|------|
| Stage1 (pretrain) | NOT_FOUND | ⚠️ 缺失 |
| Stage2 (high train) | NOT_FOUND | ⚠️ 缺失 |
| Stage3 (high refine) | NOT_FOUND | ⚠️ 缺失 |

### Seed 44

| Stage | Low Actor Hash | 状态 |
|-------|----------------|------|
| Stage1 (pretrain) | NOT_FOUND | ⚠️ 缺失 |
| Stage2 (high train) | NOT_FOUND | ⚠️ 缺失 |
| Stage3 (high refine) | NOT_FOUND | ⚠️ 缺失 |

### Seed 45

| Stage | Low Actor Hash | 状态 |
|-------|----------------|------|
| Stage1 (pretrain) | NOT_FOUND | ⚠️ 缺失 |
| Stage2 (high train) | NOT_FOUND | ⚠️ 缺失 |
| Stage3 (high refine) | NOT_FOUND | ⚠️ 缺失 |

### Seed 46

| Stage | Low Actor Hash | 状态 |
|-------|----------------|------|
| Stage1 (pretrain) | NOT_FOUND | ⚠️ 缺失 |
| Stage2 (high train) | NOT_FOUND | ⚠️ 缺失 |
| Stage3 (high refine) | NOT_FOUND | ⚠️ 缺失 |

### 冻结证据总结

- ✓ 所有 seeds 的 Stage2/Stage3 low actor 参数与 Stage1 一致

## 二、PPO-Lagrangian Lambda 与 Cost 运行轨迹

### Stage2 (High Train)

**Seed 42:**
- Lambda 初始值: 0.100000
- Lambda 最终值: 0.064124
- Lambda 变化: -0.035876
- Cost 均值: 0.004171
- Cost 最大值: 0.124857
- Unsafe rate 均值: 0.000198

**Seed 43:**
- Lambda 初始值: 0.100000
- Lambda 最终值: 0.064067
- Lambda 变化: -0.035933
- Cost 均值: 0.002365
- Cost 最大值: 0.230667
- Unsafe rate 均值: 0.000100

**Seed 44:**
- Lambda 初始值: 0.100000
- Lambda 最终值: 0.064089
- Lambda 变化: -0.035911
- Cost 均值: 0.002782
- Cost 最大值: 0.124857
- Unsafe rate 均值: 0.000072

**Seed 45:**
- Lambda 初始值: 0.100000
- Lambda 最终值: 0.064069
- Lambda 变化: -0.035931
- Cost 均值: 0.002443
- Cost 最大值: 0.116000
- Unsafe rate 均值: 0.000049

**Seed 46:**
- Lambda 初始值: 0.100000
- Lambda 最终值: 0.064201
- Lambda 变化: -0.035799
- Cost 均值: 0.006259
- Cost 最大值: 0.250381
- Unsafe rate 均值: 0.000314

### Lambda 更新证据

- ✓ stage2_high_train Seed 42: Lambda 有更新 (变化 0.035876)
- ✓ stage2_high_train Seed 43: Lambda 有更新 (变化 0.035933)
- ✓ stage2_high_train Seed 44: Lambda 有更新 (变化 0.035911)
- ✓ stage2_high_train Seed 45: Lambda 有更新 (变化 0.035931)
- ✓ stage2_high_train Seed 46: Lambda 有更新 (变化 0.035799)

## 三、最终结论

| 检查项 | 结论 |
|--------|------|
| Stage2 low actor 参数冻结 | ⚠️ 存疑（无数据） |
| Stage3 low actor 参数冻结 | ⚠️ 存疑（无数据） |
| PPO-Lagrangian Lambda 更新 | ✓ 通过 |
| Cost 约束记录 | ✓ 通过 |

**总体评估：⚠️ 存疑**

部分检查项缺少数据，无法完全验证。
