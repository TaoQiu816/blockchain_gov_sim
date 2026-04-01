# Stage2/Stage3 冻结与 PPO-Lagrangian 运行证据

## 快速开始

### 1. 导出证据

```bash
python audits/round2_runtime_checks/training_protocol/export_freeze_and_lambda_evidence.py
```

**输出：**
- `low_actor_hash_trace.csv`: 模型 hash 提取记录
- `lambda_cost_trace.csv`: Lambda 和 cost 完整轨迹 (975 条记录)
- `freeze_evidence.md`: 初步证据报告

### 2. 生成最终报告

```bash
python audits/round2_runtime_checks/training_protocol/generate_final_report.py
```

**输出：**
- `report_training_protocol.md`: 综合证据报告

## 主要发现

### ✓ PPO-Lagrangian Lambda 更新

从 5 个 seeds 的训练日志中提取到 **975 条记录**：

| Seed | 初始 Lambda | 最终 Lambda | 变化量 | Episodes |
|------|-------------|-------------|--------|----------|
| 42 | 0.100000 | 0.064124 | -0.035876 | 197 |
| 43 | 0.100000 | 0.064067 | -0.035933 | 182 |
| 44 | 0.100000 | 0.064089 | -0.035911 | 205 |
| 45 | 0.100000 | 0.064069 | -0.035931 | 185 |
| 46 | 0.100000 | 0.064201 | -0.035799 | 206 |

**结论：** Lambda 在所有 seeds 中均有显著更新（约 -0.036），证明 PPO-Lagrangian 约束机制真实生效。

### ✓ Cost 约束记录

| Seed | Cost 均值 | Cost 最大值 | Unsafe Rate 均值 |
|------|-----------|-------------|------------------|
| 42 | 0.004171 | 0.124857 | 0.000198 |
| 43 | 0.002365 | 0.230667 | 0.000100 |
| 44 | 0.002782 | 0.124857 | 0.000072 |
| 45 | 0.002443 | 0.116000 | 0.000049 |
| 46 | 0.006259 | 0.250381 | 0.000314 |

**结论：** Episode cost 和 unsafe rate 均被正确记录和计算，进入约束链路。

### ✓ Low Actor 冻结机制

通过代码架构分析确认（[`gov_sim/hierarchical/trainer.py:246-248`](../../../gov_sim/hierarchical/trainer.py)）：

1. `LowLevelInferenceAdapter` 将 low_model 封装为推理接口
2. Low policy 仅用于环境内部的 action 解码，不参与 high-level agent 的训练
3. High-level PPO agent 的 optimizer 只包含 high-level policy 参数
4. Low policy 参数不在任何 optimizer 的 param_groups 中

**结论：** Low actor 在 Stage2/Stage3 中通过架构设计实现冻结，不参与梯度更新。

## 文件清单

- [`export_freeze_and_lambda_evidence.py`](export_freeze_and_lambda_evidence.py): 证据导出脚本
- [`generate_final_report.py`](generate_final_report.py): 最终报告生成脚本
- [`run_minimal_freeze_audit.py`](run_minimal_freeze_audit.py): 最小化审计运行脚本（可选）
- [`low_actor_hash_trace.csv`](low_actor_hash_trace.csv): 模型 hash 提取记录
- [`lambda_cost_trace.csv`](lambda_cost_trace.csv): Lambda 和 cost 完整轨迹
- [`freeze_evidence.md`](freeze_evidence.md): 初步证据报告
- [`report_training_protocol.md`](report_training_protocol.md): 综合证据报告（主报告）

## 总体评估

**✓ 通过**

所有检查项均通过验证：
1. PPO-Lagrangian Lambda 更新 ✓
2. Cost 约束记录 ✓
3. Stage2 Low Actor 冻结 ✓
4. Stage3 Low Actor 冻结 ✓
5. Low Actor 不在 Optimizer 中 ✓
