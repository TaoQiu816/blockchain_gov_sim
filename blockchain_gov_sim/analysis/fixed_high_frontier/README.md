# 固定高层模板前沿评估

## 目标

基于真实评估证据判断 learned high policy 在 4 个 hard 场景下高度集中到 7|0.55，到底是合理收敛还是高层训练/状态表达存在问题。

## 评估设计

### 评估对象

- **Checkpoints**: formal_final_seed42-46 (5 个训练 seeds)
- **Scenes**: load_shock, high_rtt_burst, churn_burst, malicious_burst
- **Fixed Templates**: 12 个完整高层模板
  - (5,0.45), (5,0.50), (5,0.55), (5,0.60)
  - (7,0.45), (7,0.50), (7,0.55), (7,0.60)
  - (9,0.45), (9,0.50), (9,0.55), (9,0.60)

### 评估协议

1. **固定模板评估**: 每个 checkpoint 的 low actor + 固定高层模板
2. **Learned baseline**: 每个 checkpoint 的 high actor + low actor
3. **公平性保证**: 同一场景下所有评估使用相同的 eval seeds
4. **Deterministic 模式**: low actor 使用 deterministic 推理
5. **合法性检查**: 保持所有 legality / invalid / structural infeasible 检查

### 评估模式

- **Smoke 模式**: N_eval=20，快速验证
- **Full 模式**: N_eval=64，正式评估

## 使用方法

### 1. 运行评估

```bash
# Smoke 模式 (约 30 分钟)
python scripts/eval_fixed_high_frontier.py --mode smoke

# Full 模式 (约 2-3 小时)
python scripts/eval_fixed_high_frontier.py --mode full
```

### 2. 分析结果

```bash
python analysis/fixed_high_frontier/analyze_results.py \
    --data-dir analysis/fixed_high_frontier \
    --output-dir analysis/fixed_high_frontier/plots
```

## 输出文件

### 数据文件

- `fixed_high_episode_metrics.csv`: 固定模板逐 episode 明细
- `learned_high_episode_metrics.csv`: Learned baseline 逐 episode 明细
- `fixed_high_summary_by_ckpt.csv`: 固定模板按 checkpoint 聚合
- `fixed_high_summary_across_seeds.csv`: 固定模板跨 seeds 聚合
- `learned_high_baseline_summary.csv`: Learned baseline 聚合
- `fixed_vs_learned_gap.csv`: Fixed vs Learned 差距分析
- `metadata.json`: 评估元数据

### 图表文件 (plots/)

- `tps_latency_scatter_{scene}.png`: TPS-latency 前沿散点图
- `risk_bars_{scene}.png`: 风险指标柱状图
- `template_ranking_{scene}.csv`: 模板排序表
- `final_judgment.md`: 最终判断报告

## 关键指标

### 性能指标

- **TPS**: 吞吐量
- **mean_latency**: 平均延迟
- **queue_mean / queue_p95**: 队列长度

### 风险指标

- **unsafe_rate**: 不安全率
- **timeout_rate**: 超时率
- **policy_invalid_rate**: 策略非法率
- **structural_infeasible_rate**: 结构不可行率

## 判断标准

### 1. 7|0.55 是否接近 Pareto 最优？

- 检查 7|0.55 在各场景的 TPS 排名
- 如果排名在前 3，则认为接近最优

### 2. 不同场景下的最优模板类型

- 按场景分析 m=5/7/9 的性能差异
- 识别场景特异性模板偏好

### 3. Learned vs Best Fixed 差距

- 计算 TPS 差距百分比
- 如果差距 < 5%，认为 learned 合理收敛
- 如果差距 > 10%，认为存在训练问题

### 4. 后续检查方向

如果 learned 与 best fixed 差距显著：
- 检查高层状态表达是否充分捕捉场景差异
- 检查高层 chunk update 机制是否合理
- 检查高层目标/约束权衡是否需要调整

**不建议**:
- 收缩高层动作空间
- 重新训练 low actor
- 修改 warm-start / stage2 / stage3 逻辑

## 实现细节

### 固定模板机制

- `FixedTemplateSelector`: 在整个 episode 内固定返回指定的 (m, theta)
- 复用现有 `LowLevelInferenceAdapter` 和 `HierarchicalPolicyController` 架构
- 不修改训练代码，只在评估路径增加覆盖

### 批量评估

- 遍历所有 checkpoint × scene × template 组合
- 每个场景使用相同的 eval seeds (base_seed + episode_idx)
- 同时评估 learned baseline 作为对比

### 结果分析

- TPS-latency scatter: 可视化前沿分布
- Risk bars: 对比风险指标
- Template ranking: 按综合得分排序
- Gap analysis: 量化 learned vs fixed 差距

## 注意事项

1. **不要重训**: 复用现有 checkpoint
2. **不要收缩动作空间**: 保持 12 个完整模板
3. **不要修改训练协议**: 只做评估
4. **不要关闭检查**: 保持所有合法性检查
5. **基于真实结果**: 不要靠静态代码阅读下结论

## 预期结果

### 场景 1: 7|0.55 接近最优

- 各场景 TPS 排名前 3
- Learned vs Best Fixed 差距 < 5%
- **结论**: 合理收敛到稳健模板

### 场景 2: 7|0.55 非最优

- 某些场景 TPS 排名靠后
- Learned vs Best Fixed 差距 > 10%
- **结论**: 高层训练存在问题，需检查状态表达/更新机制

## 参考

- 高层模板定义: [`gov_sim/hierarchical/spec.py`](../../gov_sim/hierarchical/spec.py)
- 固定模板选择器: [`gov_sim/hierarchical/fixed_template_selector.py`](../../gov_sim/hierarchical/fixed_template_selector.py)
- 批量评估脚本: [`scripts/eval_fixed_high_frontier.py`](../../scripts/eval_fixed_high_frontier.py)
- 结果分析脚本: [`analyze_results.py`](analyze_results.py)
