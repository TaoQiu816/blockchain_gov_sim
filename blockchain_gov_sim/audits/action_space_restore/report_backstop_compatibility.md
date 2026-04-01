# Backstop 兼容字段污染审计报告

**审计时间**: 2026-03-18

## 审计目标

在 nominal 已包含全部 12 个高层模板后，确认 `backstop_high_template_legal` 和 `used_backstop_template` 这两个兼容字段是否继续影响：
- structural_infeasible_rate
- policy_invalid_rate
- template usage 统计
- evaluator / logger / analyzer 的结论

## 审计方法

1. 全仓库搜索 `backstop_high_template_legal` 和 `backstop` 相关字段
2. 分析每个读取位置的用途和影响
3. 判断是否污染统计口径
4. 给出最小必要修复建议

## 所有读取位置

### 1. [`gov_sim/env/gov_env.py`](gov_sim/env/gov_env.py)

#### 位置 1: 第 150 行 - 初始化为 0
```python
return {
    "num_legal_nominal_high_templates": int(legal_nominal),
    "backstop_high_template_legal": 0,  # 硬编码为 0
    "policy_invalid": int(has_protocol_legal_template),
    "structural_infeasible": int(not has_protocol_legal_template),
}
```

**用途**: 在 `_get_invalid_action_breakdown()` 中返回 invalid action 的详细信息。

**影响**: 
- ✅ **不污染统计**：该字段硬编码为 0，不参与任何逻辑判断
- ✅ `structural_infeasible` 的计算只基于 `legal_nominal`，不依赖 backstop

#### 位置 2: 第 560 行 - 记录到 info
```python
"num_legal_nominal_high_templates": int(invalid_breakdown["num_legal_nominal_high_templates"]),
"backstop_high_template_legal": int(invalid_breakdown["backstop_high_template_legal"]),
```

**用途**: 在 invalid action 时记录到 step info 中。

**影响**:
- ✅ **不污染统计**：仅作为记录字段，值始终为 0
- ⚠️ 但会出现在日志和审计文件中

#### 位置 3: 第 684 行 - 记录到 info
```python
if invalid_breakdown is not None:
    info["num_legal_nominal_high_templates"] = int(invalid_breakdown["num_legal_nominal_high_templates"])
    info["backstop_high_template_legal"] = int(invalid_breakdown["backstop_high_template_legal"])
```

**用途**: 在 hierarchical 模式下记录到 step info 中。

**影响**:
- ✅ **不污染统计**：仅作为记录字段，值始终为 0
- ⚠️ 但会出现在日志和审计文件中

### 2. [`gov_sim/modules/metrics_tracker.py`](gov_sim/modules/metrics_tracker.py)

#### 位置: 第 55 行 - 计算 used_backstop_template_rate
```python
"used_backstop_template_rate": float(np.mean([row.get("used_backstop_template", 0) for row in self.rows])),
```

**用途**: 在 `get_summary()` 中计算 backstop 模板使用率。

**影响**:
- ✅ **不污染核心统计**：该字段独立统计，不影响 structural_infeasible_rate 或 policy_invalid_rate
- ⚠️ 但会出现在汇总统计中，值始终为 0

### 3. [`scripts/aggregate_formal_hierarchical.py`](scripts/aggregate_formal_hierarchical.py)

#### 位置: 第 21 行 - 作为汇总字段
```python
"structural_infeasible_rate",
"used_backstop_template_rate",
"template_dominant_ratio",
```

**用途**: 在正式实验汇总中提取该字段。

**影响**:
- ✅ **不污染核心统计**：仅作为独立字段提取
- ⚠️ 但会出现在汇总 CSV 中，值始终为 0

### 4. 审计脚本（不影响训练和评估）

以下文件仅用于审计分析，不影响训练和评估：
- [`audits/round2_runtime_checks/action_space_compression/analyze_existing_results.py`](audits/round2_runtime_checks/action_space_compression/analyze_existing_results.py)
- [`audits/round2_runtime_checks/action_space_compression/analyze_template_distribution.py`](audits/round2_runtime_checks/action_space_compression/analyze_template_distribution.py)
- [`audits/round2_runtime_checks/action_space_compression/export_legality_frequency.py`](audits/round2_runtime_checks/action_space_compression/export_legality_frequency.py)

### 5. 测试脚本（验证移除）

[`scripts/tests/test_high_action_space_restored.py`](scripts/tests/test_high_action_space_restored.py) 中有测试验证 backstop 相关属性已被移除。

## 是否污染统计？

### 核心结论：✅ **不污染核心统计口径**

**理由**：

1. **structural_infeasible_rate 计算不受影响**
   - 计算逻辑：`structural_infeasible = int(not has_protocol_legal_template)`
   - 只依赖 `legal_nominal`（12 个 nominal 模板的合法性）
   - 不依赖 `backstop_high_template_legal`（始终为 0）

2. **policy_invalid_rate 计算不受影响**
   - 计算逻辑：`policy_invalid = int(has_protocol_legal_template)`
   - 只依赖 `legal_nominal > 0`
   - 不依赖 backstop 字段

3. **template usage 统计不受影响**
   - `used_backstop_template` 始终为 0
   - 不影响 nominal 模板的使用统计
   - 不影响 `top_template` 和 `template_dominant_ratio`

4. **evaluator / logger / analyzer 结论不受影响**
   - 所有核心指标（TPS、latency、unsafe_rate、timeout_rate）不依赖 backstop 字段
   - 汇总统计中 `used_backstop_template_rate` 始终为 0，不影响结论

### 潜在问题：⚠️ **字段冗余和混淆**

虽然不污染统计，但存在以下问题：

1. **日志冗余**：`backstop_high_template_legal` 和 `used_backstop_template` 始终为 0，但仍出现在所有日志和审计文件中
2. **代码混淆**：保留这些字段可能让未来维护者误以为 backstop 机制仍然存在
3. **存储浪费**：每个 step 都记录这些无用字段

## 是否需要修复？

### 建议：✅ **需要清理，但优先级较低**

**理由**：
- 不影响当前训练和评估的正确性
- 不影响论文实验结果
- 但为了代码清洁和可维护性，建议清理

## 最小必要修复建议

### 方案 1：完全移除（推荐）

**修改文件**：
1. [`gov_sim/env/gov_env.py`](gov_sim/env/gov_env.py)
   - 移除 `backstop_high_template_legal` 字段的所有引用
   - 移除 `used_backstop_template` 字段的所有引用

2. [`gov_sim/modules/metrics_tracker.py`](gov_sim/modules/metrics_tracker.py)
   - 移除 `used_backstop_template_rate` 的计算

3. [`scripts/aggregate_formal_hierarchical.py`](scripts/aggregate_formal_hierarchical.py)
   - 移除 `used_backstop_template_rate` 字段

**优点**：
- 代码更清洁
- 减少混淆
- 减少存储开销

**缺点**：
- 需要修改多个文件
- 可能影响旧结果的兼容性（如果需要重新分析旧数据）

### 方案 2：保留但添加注释（临时方案）

**修改文件**：
1. [`gov_sim/env/gov_env.py`](gov_sim/env/gov_env.py)
   - 在相关位置添加注释说明这些字段已废弃，仅为兼容性保留

**优点**：
- 最小改动
- 保持向后兼容

**缺点**：
- 仍然存在冗余
- 未来仍需清理

### 方案 3：不修复（当前状态）

**理由**：
- 不影响正确性
- 论文截稿前不引入新风险

**缺点**：
- 代码不够清洁
- 可能混淆未来维护者

## 推荐行动

### 短期（论文截稿前）：
✅ **不修复**，保持当前状态
- 已验证不污染统计
- 不影响论文实验结果
- 避免引入新风险

### 中期（论文提交后）：
✅ **执行方案 1**，完全移除 backstop 相关字段
- 清理代码
- 提高可维护性
- 为未来工作打好基础

## 最终结论

1. ✅ **backstop 兼容字段不污染当前统计口径**
2. ✅ **structural_infeasible_rate 和 policy_invalid_rate 计算正确**
3. ✅ **template usage 统计不受影响**
4. ✅ **evaluator / logger / analyzer 结论不受影响**
5. ⚠️ **存在字段冗余，但不影响正确性**
6. 📋 **建议论文提交后清理，当前不修复**
