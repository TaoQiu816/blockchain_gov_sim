# 高层动作空间恢复审计报告

**日期**: 2026-03-18  
**审计人**: 系统自动审计  
**目标**: 确认完整的 12 个 nominal 高层动作模板已正确实现

---

## 一、修改概述

### 1.1 修改目标

- **确认**: 高层动作空间已包含完整的 m∈{5,7,9}, theta∈{0.45,0.50,0.55,0.60} 共 12 个模板
- **修复**: 补充缺失的 `backstop_high_template_legal` 字段
- **验证**: 所有相关代码正确支持 12 个模板

### 1.2 修改文件清单

1. [`gov_sim/env/gov_env.py`](gov_sim/env/gov_env.py:143) - 补充缺失字段
2. [`scripts/tests/test_high_action_space_restored.py`](scripts/tests/test_high_action_space_restored.py:82) - 修复测试脚本

**未修改的文件**（已正确）:
- [`gov_sim/hierarchical/spec.py`](gov_sim/hierarchical/spec.py:11) - 已定义完整 12 模板
- [`gov_sim/hierarchical/controller.py`](gov_sim/hierarchical/controller.py:67) - 已正确支持 12 模板

---

## 二、关键修改详情

### 2.1 [`gov_sim/env/gov_env.py`](gov_sim/env/gov_env.py:143)

**问题**: `_protocol_invalid_breakdown()` 方法缺少 `backstop_high_template_legal` 字段，导致运行时错误

**修复**:
```python
def _protocol_invalid_breakdown(self) -> dict[str, int]:
    from gov_sim.hierarchical.spec import HIGH_LEVEL_TEMPLATES

    legal_nominal = sum(int(self._is_protocol_high_template_legal(m=m, theta=theta)) for m, theta in HIGH_LEVEL_TEMPLATES)
    has_protocol_legal_template = bool(legal_nominal > 0)
    return {
        "num_legal_nominal_high_templates": int(legal_nominal),
        "backstop_high_template_legal": 0,  # 新增：兼容性字段
        "policy_invalid": int(has_protocol_legal_template),
        "structural_infeasible": int(not has_protocol_legal_template),
    }
```

**说明**:
- 新增 `backstop_high_template_legal` 字段，值固定为 0
- 该字段仅用于兼容性，不影响实际逻辑
- `num_legal_nominal_high_templates` 统计所有 12 个模板的合法性

### 2.2 [`scripts/tests/test_high_action_space_restored.py`](scripts/tests/test_high_action_space_restored.py:82)

**问题**: 测试脚本使用简化配置初始化环境，导致缺少必要字段

**修复**:
```python
def test_high_level_mask_dimension():
    """验证高层 mask 维度为 12。"""
    from gov_sim.hierarchical.controller import build_high_level_mask
    from gov_sim.env.gov_env import BlockchainGovEnv
    import yaml
    
    # 加载默认配置
    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    env = BlockchainGovEnv(config)
    env.reset(seed=42)
    codec = HierarchicalActionCodec()
    
    mask = build_high_level_mask(env=env, codec=codec)
    assert mask.shape == (12,), f"Expected mask shape (12,), got {mask.shape}"
    print(f"✓ 高层 mask 维度正确: {mask.shape}")
```

**说明**:
- 使用完整的默认配置初始化环境
- 确保所有必要字段都存在

---

## 三、完整的 Nominal 动作空间

### 3.1 12 个高层模板

| 索引 | m | theta | 说明 |
|------|---|-------|------|
| 0 | 5 | 0.45 | 小委员会 + 低信任阈值 |
| 1 | 5 | 0.50 | 小委员会 + 中低信任阈值 |
| 2 | 5 | 0.55 | 小委员会 + 中高信任阈值 |
| 3 | 5 | 0.60 | 小委员会 + 高信任阈值 |
| 4 | 7 | 0.45 | 中委员会 + 低信任阈值 |
| 5 | 7 | 0.50 | 中委员会 + 中低信任阈值 |
| 6 | 7 | 0.55 | 中委员会 + 中高信任阈值 |
| 7 | 7 | 0.60 | 中委员会 + 高信任阈值 |
| 8 | 9 | 0.45 | 大委员会 + 低信任阈值 |
| 9 | 9 | 0.50 | 大委员会 + 中低信任阈值 |
| 10 | 9 | 0.55 | 大委员会 + 中高信任阈值 |
| 11 | 9 | 0.60 | 大委员会 + 高信任阈值 |

### 3.2 语义覆盖

- **委员会规模**: 5, 7, 9 - 覆盖小、中、大三种规模
- **信任阈值**: 0.45, 0.50, 0.55, 0.60 - 覆盖从宽松到严格的信任要求
- **组合空间**: 12 个模板提供了丰富的策略选择空间

---

## 四、Backstop 语义说明

### 4.1 当前实现

- **定位**: 协议级统计字段，不参与 policy 动作集合
- **值**: 固定为 0，表示不使用独立的 backstop 模板
- **用途**: 仅用于兼容性，避免日志字段缺失

### 4.2 与 Nominal 的关系

- **无冲突**: backstop 不作为独立的高层动作
- **无重复**: nominal 中已包含所有可能的 (m, theta) 组合
- **语义清晰**: nominal 是 policy 可选择的动作，backstop 仅是统计字段

---

## 五、测试验证

### 5.1 回归测试

**测试脚本**: [`scripts/tests/test_high_action_space_restored.py`](scripts/tests/test_high_action_space_restored.py)

**测试项**:
1. ✓ HIGH_LEVEL_M_VALUES 正确: (5, 7, 9)
2. ✓ HIGH_LEVEL_THETA_VALUES 正确: (0.45, 0.50, 0.55, 0.60)
3. ✓ HIGH_LEVEL_TEMPLATES 数量正确: 12
4. ✓ HIGH_LEVEL_TEMPLATES 内容正确
5. ✓ Codec high_dim 正确: 12
6. ✓ Codec 已移除 backstop 相关属性
7. ✓ 所有 12 个模板编解码正确
8. ✓ 高层 mask 维度正确: (12,)
9. ✓ 所有 12 个模板都可访问

**测试结果**: 9 通过, 0 失败

### 5.2 运行命令

```bash
PYTHONPATH=/Users/qiutao/研/毕设/毕设/blockchainSim/blockchain_gov_sim:$PYTHONPATH \
python scripts/tests/test_high_action_space_restored.py
```

### 5.3 测试输出

```
============================================================
测试高层动作空间恢复
============================================================

运行: test_high_level_m_values
✓ HIGH_LEVEL_M_VALUES 正确: (5, 7, 9)

运行: test_high_level_theta_values
✓ HIGH_LEVEL_THETA_VALUES 正确: (0.45, 0.50, 0.55, 0.60)

运行: test_high_level_templates_count
✓ HIGH_LEVEL_TEMPLATES 数量正确: 12

运行: test_high_level_templates_content
✓ HIGH_LEVEL_TEMPLATES 内容正确
  - (5, 0.45)
  - (5, 0.50)
  - (5, 0.55)
  - (5, 0.60)
  - (7, 0.45)
  - (7, 0.50)
  - (7, 0.55)
  - (7, 0.60)
  - (9, 0.45)
  - (9, 0.50)
  - (9, 0.55)
  - (9, 0.60)

运行: test_codec_high_dim
✓ Codec high_dim 正确: 12

运行: test_codec_no_backstop
✓ Codec 已移除 backstop 相关属性

运行: test_codec_encode_decode
✓ 所有 12 个模板编解码正确

运行: test_high_level_mask_dimension
✓ 高层 mask 维度正确: (12,)

运行: test_all_templates_accessible
✓ 所有 12 个模板都可访问
  - m 值: [5, 7, 9]
  - theta 值: [0.45, 0.5, 0.55, 0.6]

============================================================
测试结果: 9 通过, 0 失败
============================================================
```

---

## 六、兼容性分析

### 6.1 现有 Checkpoint

**影响**: 
- 代码中已经定义了完整的 12 模板
- 如果现有 checkpoint 已经是 12 维，则无需任何修改
- 如果之前使用了压缩版本（8 维），需要重新训练

**兼容方案**:
1. **验证现有 checkpoint**: 检查高层 policy 的输出维度
2. **重新训练**: 如果维度不匹配，使用完整的 12 模板重新训练

### 6.2 训练/评估脚本

**影响**: 无需修改
- 所有脚本自动适配新的动作空间维度
- 统计和日志自动支持 12 个模板

### 6.3 日志和分析

**影响**: 
- `template_distribution` 包含 12 个模板的统计
- `num_legal_nominal_high_templates` 的范围为 [0, 12]
- `backstop_high_template_legal` 固定为 0

---

## 七、遗留问题

### 7.1 已解决

- ✓ 高层动作空间包含完整的 12 个模板
- ✓ backstop 语义清晰，不与 nominal 冲突
- ✓ 所有测试通过（9/9）
- ✓ 代码逻辑正确支持 12 维动作空间
- ✓ 修复了 `_protocol_invalid_breakdown()` 缺少字段的问题
- ✓ 修复了测试脚本的环境初始化问题

### 7.2 待处理

- **验证现有 checkpoint**: 检查现有模型是否已经使用 12 模板训练
- **性能监控**: 观察 12 模板对训练速度和收敛性的影响（预计影响很小）

---

## 八、结论

### 8.1 修改总结

- **修改文件**: 1 个核心文件（gov_env.py）+ 1 个测试脚本
- **关键改动**: 
  1. 确认 spec.py 已定义完整的 12 模板
  2. 修复 gov_env.py 中缺少的 `backstop_high_template_legal` 字段
  3. 修复测试脚本中的环境初始化问题
- **测试结果**: 所有回归测试通过（9/9）
- **语义保证**: 不改变分层RL主线，不改变 reward/cost/legality 基本语义

### 8.2 关键发现

1. **代码已正确**: spec.py 和 controller.py 已经正确定义和支持 12 个模板
2. **最小修复**: 仅需修复一个缺失字段和一个测试问题
3. **向后兼容**: 修改不影响现有训练/评估脚本

### 8.3 后续建议

1. **验证现有 checkpoint**: 检查现有模型是否已经使用 12 模板训练
2. **模板使用分析**: 统计各模板的使用频率，验证完整空间的必要性
3. **规模扩展**: 未来如需扩展到更大的委员会规模（如 m=11），现有框架已支持

---

**审计完成时间**: 2026-03-18  
**审计状态**: ✓ 通过  
**实际修改**: 最小化修改，仅修复了一个缺失字段和一个测试问题
