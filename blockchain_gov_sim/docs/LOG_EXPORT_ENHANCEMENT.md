# 训练/评估日志导出增强

## 概述

本次更新为 IoV/VEC 联盟链链侧分层动态治理仿真环��补齐了训练/评估日志导出功能，添加了缺失的日志项，并统一了日志格式为 CSV/JSON。

## 完成的工作

### D1. 检查当前训练日志导出情况

已检查现有的训练日志导出机制：
- [`gov_sim/agent/callbacks.py`](../gov_sim/agent/callbacks.py) 中的 [`TrainLoggingCallback`](../gov_sim/agent/callbacks.py:21) 负责 episode 级日志
- 已有基础的 reward、cost、unsafe_rate、TPS、latency 等指标
- 缺少 high template usage、low action usage、entropy trajectory、reward/cost decomposition 等细分指标

### D2. 添加缺失的日志项

#### 扩展 TrainLoggingCallback

在 [`gov_sim/agent/callbacks.py`](../gov_sim/agent/callbacks.py) 中添加：

1. **Step 级详细日志**：
   - 新增 `step_rows` 列表，记录每个 step 的详细信息
   - 包含：reward、cost、entropy、lambda、executed_high_template、executed_low_action、TPS、latency 等

2. **Entropy 收集**：
   - 在 `_on_step` 中尝���从 policy 获取 entropy
   - 计算 episode 级平均 entropy

3. **Template/Action 分布统计**：
   - 新增 `_ep_high_template_counts` 和 `_ep_low_action_counts`
   - 计算 episode 级 template/action 使用分布
   - 计算 template/action 的 entropy（多样性指标）

4. **Episode 级新增字段**：
   - `mean_entropy`: episode 平均 entropy
   - `top_high_template`: 最常用的 high template
   - `high_template_entropy`: high template 分布的 entropy
   - `top_low_action`: 最常用的 low action
   - `low_action_entropy`: low action 分布的 entropy
   - `num_unique_high_templates`: 使用的不同 high template 数量
   - `num_unique_low_actions`: 使用的不同 low action 数量

5. **Step 级日志导出**：
   - 在 `_on_training_end` 中导出 `train_log_steps.csv`
   - 包含完整的 step 级轨迹数据

### D3. 统一日志格式为 CSV/JSON

#### 新增日志导出模块

创建 [`gov_sim/utils/log_export.py`](../gov_sim/utils/log_export.py)，提供：

1. **`export_training_summary`**：
   - 读取 episode 级日志（`train_log.csv`）
   - 读取 step 级日志（`train_log_steps.csv`）
   - 读取 audit JSON（`train_audit.json`）
   - 生成统一的 `training_export_summary.json`
   - 包含：
     - Episode 统计（mean reward、cost、TPS、latency、final lambda）
     - Step 统计（mean reward、cost、entropy）
     - Lambda 轨迹（min、max、mean、std、final）
     - High template 使用频率分布
     - Low action 使用频率分布

2. **`export_evaluation_summary`**：
   - 读取多个评估 CSV 文件
   - 按 scenario 和 controller 分组
   - 计算关键指标（TPS、latency、unsafe_rate、infeasible_rate）
   - 提取 template/action 分布
   - 生成统一的 `evaluation_export_summary.json`

3. **`generate_training_artifacts_with_export`**：
   - 集成到现有的 [`generate_train_artifacts`](../gov_sim/utils/train_artifacts.py:79) 函数
   - 自动调用 `export_training_summary`

## 导出的日志文件

### 训练阶段

每个训练阶段目录（如 `stage2_high_train`）会生成：

1. **`train_log.csv`** (Episode 级)：
   - 原有字段：timesteps、episode_reward、episode_cost、unsafe_rate、TPS、latency、lagrangian_lambda 等
   - 新增字段：mean_entropy、top_high_template、high_template_entropy、top_low_action、low_action_entropy、num_unique_high_templates、num_unique_low_actions

2. **`train_log_steps.csv`** (Step 级，新增)：
   - timesteps、episode_seed、scenario_type、step_in_episode
   - reward、cost、entropy、lagrangian_lambda
   - executed_high_template、executed_low_action
   - tps、latency、unsafe、mask_ratio、eligible_size

3. **`train_audit.json`** (Audit 信息)：
   - 原有：episodes、episode_repeat_ratio、action_frequency 等
   - 新增：executed_high_template_distribution、executed_low_action_distribution

4. **`training_export_summary.json`** (统一摘要，新增)：
   - 整合 episode、step、audit 信息
   - 提供结构化的统计摘要
   - 包含 lambda 轨迹、template/action 使用分布

### 评估阶段

每个评估目录（如 `hard_eval`）会生成：

1. **`{scenario}_{controller}.csv`** (原有)：
   - 每个 scenario-controller 组合的详细 episode 数据

2. **`evaluation_export_summary.json`** (新增)：
   - 按 scenario 和 controller 分组的统计摘要
   - 包含 TPS、latency、unsafe_rate、template/action 分布

## 使用方法

### 测试日志导出

```bash
python scripts/test_log_export.py
```

### 在训练中自动导出

日志导出已集成到训练流程中，无需额外操作。训练完成后会自动生成所有日志文件。

### 手动导出

```python
from gov_sim.utils.log_export import export_training_summary, export_evaluation_summary
from pathlib import Path

# 导出训练摘要
export_training_summary(
    stage_dir=Path("outputs/formal_multiseed/hierarchical/formal_final_seed43/stage2_high_train"),
    episode_log_path=Path("outputs/.../train_log.csv"),
    step_log_path=Path("outputs/.../train_log_steps.csv"),
    audit_path=Path("outputs/.../train_audit.json"),
)

# 导出评估摘要
export_evaluation_summary(
    eval_dir=Path("outputs/.../hard_eval"),
    episode_csv_paths=[Path("outputs/.../scenario_controller.csv"), ...],
    metadata={"description": "Hard scenario evaluation"},
)
```

## 关键改进

1. **完整的轨迹记录**：
   - Step 级日志提供完整的训练轨迹
   - 可用于分析 lambda 演化、entropy 变化、template/action 切换模式

2. **分层动作分布**：
   - 明确区分 high template 和 low action 的使用情况
   - 计算分布 entropy 以评估策略多样性

3. **统一的导出格式**：
   - JSON 格式便于程序化分析
   - CSV 格式便于人工检查和可视化
   - 结构化的摘要便于快速对比不同实验

4. **向后兼容**：
   - 保留原有的 `train_log.csv` 格式
   - 新增字段不影响现有分析脚本
   - 导出功能失败不影响训练主流程

## 后续工作

- E1: learned baseline eval
- E2: fixed high frontier eval (已部分完成)
- E3: scalability eval
- F: 生成最终交付报告
