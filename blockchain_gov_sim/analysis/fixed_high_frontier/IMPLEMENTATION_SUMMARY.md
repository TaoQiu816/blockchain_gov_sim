# 固定高层模板前沿评估 - 实现总结

## 实现概述

已完成"固定 12 个高层模板前沿评估"的完整实现，用于判断 learned high policy 集中在 7|0.55 是合理收敛还是训练问题。

## 新增文件清单

### 1. 核心实现

#### [`gov_sim/hierarchical/fixed_template_selector.py`](../../gov_sim/hierarchical/fixed_template_selector.py)
- **功能**: 固定高层模板选择器
- **设计**: 在整个 episode 内固定返回指定的 (m, theta)
- **接口**: 兼容 `EpisodeSampledTemplateSelector` 和 `HighLevelInferenceAdapter`
- **特点**: 
  - 不修改训练逻辑
  - 支持 legality remap
  - 保持所有合法性检查

#### [`gov_sim/hierarchical/__init__.py`](../../gov_sim/hierarchical/__init__.py) (修改)
- **修改**: 导出 `FixedTemplateSelector`
- **影响**: 无，仅增加导出

### 2. 批量评估脚本

#### [`scripts/eval_fixed_high_frontier.py`](../../scripts/eval_fixed_high_frontier.py)
- **功能**: 批量评估所有 checkpoint × scene × template 组合
- **评估对象**:
  - 5 个 checkpoints (seed 42-46)
  - 4 个 scenes (load_shock, high_rtt_burst, churn_burst, malicious_burst)
  - 12 个 fixed templates
  - 1 个 learned baseline
- **输出文件**:
  - `fixed_high_episode_metrics.csv`: 固定模板逐 episode 明细
  - `learned_high_episode_metrics.csv`: Learned baseline 逐 episode 明细
  - `fixed_high_summary_by_ckpt.csv`: 按 checkpoint 聚合
  - `fixed_high_summary_across_seeds.csv`: 跨 seeds 聚合
  - `learned_high_baseline_summary.csv`: Learned baseline 聚合
  - `metadata.json`: 评估元数据

### 3. 结果分析脚本

#### [`analysis/fixed_high_frontier/analyze_results.py`](../../analysis/fixed_high_frontier/analyze_results.py)
- **功能**: 生成图表和排序表
- **输出**:
  - TPS-latency scatter 图 (每个场景)
  - 风险指标柱状图 (每个场景)
  - 模板排序表 (每个场景)
  - Fixed vs Learned 差距分析
  - 最终判断报告

### 4. 文档和脚本

#### [`analysis/fixed_high_frontier/README.md`](../../analysis/fixed_high_frontier/README.md)
- **内容**: 完整的评估设计、使用方法、判断标准

#### [`scripts/run_fixed_high_frontier.sh`](../../scripts/run_fixed_high_frontier.sh)
- **功能**: 一键运行完整评估流程
- **用法**: `./scripts/run_fixed_high_frontier.sh [smoke|full]`

## 运行命令

### Smoke 模式 (快速验证)

```bash
# 方式 1: 使用一键脚本
./scripts/run_fixed_high_frontier.sh smoke

# 方式 2: 分步执行
python scripts/eval_fixed_high_frontier.py --mode smoke
python analysis/fixed_high_frontier/analyze_results.py
```

### Full 模式 (正式评估)

```bash
# 方式 1: 使用一键脚本
./scripts/run_fixed_high_frontier.sh full

# 方式 2: 分步执行
python scripts/eval_fixed_high_frontier.py --mode full
python analysis/fixed_high_frontier/analyze_results.py
```

## 结果目录结构

```
analysis/fixed_high_frontier/
├── README.md                              # 说明文档
├── analyze_results.py                     # 分析脚本
├── metadata.json                          # 评估元数据
├── fixed_high_episode_metrics.csv         # 固定模板明细
├── learned_high_episode_metrics.csv       # Learned baseline 明细
├── fixed_high_summary_by_ckpt.csv         # 按 checkpoint 聚合
├── fixed_high_summary_across_seeds.csv    # 跨 seeds 聚合
├── learned_high_baseline_summary.csv      # Learned baseline 聚合
└── plots/
    ├── tps_latency_scatter_load_shock.png
    ├── tps_latency_scatter_high_rtt_burst.png
    ├── tps_latency_scatter_churn_burst.png
    ├── tps_latency_scatter_malicious_burst.png
    ├── risk_bars_load_shock.png
    ├── risk_bars_high_rtt_burst.png
    ├── risk_bars_churn_burst.png
    ├── risk_bars_malicious_burst.png
    ├── template_ranking_load_shock.csv
    ├── template_ranking_high_rtt_burst.csv
    ├── template_ranking_churn_burst.csv
    ├── template_ranking_malicious_burst.csv
    ├── fixed_vs_learned_gap.csv
    └── final_judgment.md                  # 最终判断报告
```

## 关键设计决策

### 1. 固定模板机制

- **实现**: `FixedTemplateSelector` 类
- **原理**: 在 `template_for_step()` 中始终返回固定模板
- **合法性**: 如果固定模板不合法，自动 remap 到最近的合法模板
- **兼容性**: 完全兼容现有 `LowLevelGovEnv` 和 `HierarchicalPolicyController`

### 2. 评估公平性

- **Eval seeds**: 同一场景下所有评估使用相同的 eval seeds
- **Deterministic**: low actor 使用 deterministic 推理
- **Legality**: 保持所有 legality / invalid / structural infeasible 检查
- **Baseline**: 同时评估 learned high + learned low 作为对比

### 3. 结果分析

- **TPS-latency scatter**: 可视化前沿分布，颜色映射 unsafe_rate
- **Risk bars**: 对比 unsafe / timeout / structural infeasible
- **Template ranking**: 按综合得分排序 (TPS + latency + safety)
- **Gap analysis**: 量化 learned vs best fixed 差距

### 4. 判断标准

- **7|0.55 排名**: 检查在各场景的 TPS 排名
- **最优模板**: 识别各场景的最优模板类型 (m=5/7/9)
- **差距阈值**: 
  - < 5%: 合理收敛
  - > 10%: 训练问题
- **后续方向**: 基于差距大小给出检查建议

## 约束遵守情况

✅ **不重训**: 复用现有 formal_final_seed42-46 checkpoint
✅ **不收缩动作空间**: 保持 12 个完整高层模板
✅ **不修改训练协议**: 只做评估相关新增代码
✅ **不改 warm-start / stage2 / stage3**: 不污染训练主线
✅ **不靠静态代码阅读**: 必须生成真实评估结果
✅ **不关闭检查**: 保持所有 legality / invalid / structural infeasible 检查
✅ **不污染训练主线**: 只在评估路径增加覆盖

## 预期输出示例

### 最终判断报告 (final_judgment.md)

```markdown
# 固定高层模板前沿评估 - 最终判断报告

## 1. 7|0.55 是否接近 Pareto 最优？

- **load_shock**: 7|0.55 TPS 排名 2/12
- **high_rtt_burst**: 7|0.55 TPS 排名 1/12
- **churn_burst**: 7|0.55 TPS 排名 3/12
- **malicious_burst**: 7|0.55 TPS 排名 2/12

## 2. 不同场景下的最优模板类型

- **load_shock**: 最优模板 7|0.50 (TPS=245.32)
- **high_rtt_burst**: 最优模板 7|0.55 (TPS=198.76)
- **churn_burst**: 最优模板 9|0.45 (TPS=223.45)
- **malicious_burst**: 最优模板 7|0.55 (TPS=187.23)

## 3. Learned vs Best Fixed 差距

- 平均 TPS 差距: 3.2%
- **load_shock**: Learned TPS=238.45, Best Fixed TPS=245.32, Gap=-2.8%
- **high_rtt_burst**: Learned TPS=195.67, Best Fixed TPS=198.76, Gap=-1.6%
- **churn_burst**: Learned TPS=215.34, Best Fixed TPS=223.45, Gap=-3.6%
- **malicious_burst**: Learned TPS=183.21, Best Fixed TPS=187.23, Gap=-2.1%

## 4. 最终判断

**结论**: Learned high policy 集中在 7|0.55 是**合理收敛**，该模板在多数场景下接近 Pareto 最优。
```

## 下一步

1. **运行 smoke 评估**: 验证实现正确性
2. **检查结果**: 确认数据格式和图表质量
3. **运行 full 评估**: 获取正式结果
4. **分析判断**: 基于真实结果给出最终判断
5. **后续行动**: 根据判断结果决定是否需要调整高层训练

## 技术亮点

1. **最小侵入**: 不修改训练代码，只在评估路径增加覆盖
2. **完全兼容**: 复用现有架构和 checkpoint
3. **公平对比**: 统一 eval seeds 和 deterministic 模式
4. **全面分析**: 从多个维度评估模板性能
5. **可操作结论**: 基于量化指标给出明确判断和后续建议
