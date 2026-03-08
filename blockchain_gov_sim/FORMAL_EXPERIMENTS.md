# 第四章正式实验执行说明

本文档固定当前仓库的第四章正式实验方案，目标是把“默认工作点 + 正式训练 + benchmark + ablation + 单 GPU 调度”落成可执行流程。

## 0. 环境准备

如果你使用的是仓库中新增的 `base_requirements.txt` 对应 conda 环境，请不要重建环境，
而是在该环境上直接补项目所需增量依赖：

```bash
cd /Users/qiutao/研/毕设/毕设/blockchainSim/blockchain_gov_sim
conda activate <你的现有环境>
pip install -r base_env_delta_requirements.txt
```

或直接执行：

```bash
bash scripts/install_on_base_env.sh
```

补丁依赖的目的只有两个：
- 补齐 `stable-baselines3` 与 `sb3-contrib`
- 把 `gymnasium` 从 `1.2.3` 调整到与 SB3 2.4.x 兼容的版本

## 1. 默认工作点

### 1.1 hard/default

- `N_R = 27`
- `p_m = 0.32`
- `RTT = [14, 42] ms`
- `p_off = 0.08`
- `p_on = 0.22`
- `lambda = 260`
- `rho = {svc:0.95, con:0.96, rec:0.94, stab:0.97}`
- `N0 = 8.0`
- `u_min = 0.58`
- `h_min = 0.68`
- `gamma = {sc:0.35, on:0.30, rec:0.25, col:0.22, stab:0.18}`
- `Static-Param = (11,256,40,0.55)`

对应覆盖文件：

- `configs/scenario_default_hard.yaml`

### 1.2 moderate

除恶意比例外，与 hard/default 保持一致：

- `p_m = 0.20`

对应覆盖文件：

- `configs/scenario_default_moderate.yaml`

## 2. 正式训练

正式训练配置：

- `configs/train_final.yaml`

默认训练参数：

- `learning_rate = 2.5e-4`
- `total_timesteps = 300000`
- `n_steps = 512`
- `batch_size = 128`
- `n_epochs = 8`

种子：

- hard: `[42, 52, 62, 72, 82]`
- moderate: `[42, 52, 62]`

训练输出目录：

```text
outputs/formal/train/
├─ hard_seed42/
├─ hard_seed52/
├─ ...
├─ moderate_seed42/
├─ ...
├─ hard_best/
└─ moderate_best/
```

每个种子目录至少生成：

- `model.zip`
- `train_log.csv`
- `train_summary.json`
- `reward_curve.png/.pdf`
- `cost_curve.png/.pdf`
- `unsafe_curve.png/.pdf`
- `lambda_curve.png/.pdf`
- `post_train_eval.csv`

## 3. benchmark / ablation

正式 benchmark 与消融统一由以下配置描述：

- `configs/benchmark_final.yaml`
- `configs/ablation_final.yaml`

### 3.1 main compare

方法：

- `Static-Param`
- `Heuristic-AIMD`
- `AE-PBFT-like`
- `DVRC-like`
- `TwoLayer-LPBFT-like`
- `Ours`

指标：

- `R_unsafe`
- `R_pollute`
- `TPS`
- `mean_latency`
- `queue_peak`
- `eligible_size_mean`

### 3.2 malicious scan

- `p_m ∈ {0, 0.1, 0.2, 0.3, 0.4}`

### 3.3 dynamic attacks

- on-off period: `{4, 8, 12}`
- zigzag freq: `{0.08, 0.18, 0.32}`
- collusion group size: `{2, 4, 6, 8}`

### 3.4 load shock

- `lambda1 = 160`
- `lambda2 ∈ {240, 320, 400}`
- `e0 = 40`

### 3.5 high RTT

- `RTT_max ∈ {60, 80, 100, 120}`

### 3.6 high churn

- `(p_off, p_on)`:
  - `(0.02, 0.20)`
  - `(0.05, 0.20)`
  - `(0.08, 0.22)`
  - `(0.10, 0.20)`

### 3.7 ablation

- `w/o context gate`
- `w/o consistency & anti-manipulation`
- `Top-K instead of soft sortition`
- `Plain PPO instead of PPO-Lagrangian`
- `w/o action mask`
- `single-dimension reputation`

## 4. 单 GPU 调度建议

### 必须串行的任务

- `Ours` 的正式训练
- 正式消融中需要重新训练的 RL 变体

### 可与 GPU 训练并行的 CPU 任务

- baseline 纯评估
- benchmark 汇总
- csv/json 二次统计
- 绘图与论文制表

### 推荐顺序

1. 先跑 hard/default 的 5 个种子训练；
2. 选择 `hard_best`；
3. 用 `hard_best` 跑正式 benchmark；
4. 再跑 moderate 的 3 个种子训练；
5. 选择 `moderate_best`；
6. 最后跑正式消融。

原因：

- benchmark 与主对比都依赖稳定的 `hard_best`；
- moderate 主要是补充工作点，不阻塞主结果；
- ablation 最耗时，适合主结果稳定后再跑。

## 5. 直接可运行命令

### 5.1 hard/default 正式训练

```bash
python scripts/train_gov.py \
  --config configs/default.yaml \
  --train-config configs/train_final.yaml \
  --override configs/scenario_default_hard.yaml \
  --seed 42 \
  --run-name hard_seed42 \
  --output-root outputs/formal 2>&1 | tee outputs/formal/logs/train_hard_seed42.log
```

### 5.2 moderate 正式训练

```bash
python scripts/train_gov.py \
  --config configs/default.yaml \
  --train-config configs/train_final.yaml \
  --override configs/scenario_default_moderate.yaml \
  --seed 42 \
  --run-name moderate_seed42 \
  --output-root outputs/formal 2>&1 | tee outputs/formal/logs/train_moderate_seed42.log
```

### 5.3 选最佳模型

```bash
python scripts/select_best_model.py --train-root outputs/formal/train --prefix hard
python scripts/select_best_model.py --train-root outputs/formal/train --prefix moderate
```

### 5.4 正式 benchmark + ablation

```bash
python scripts/run_formal_suite.py \
  --config configs/default.yaml \
  --benchmark-config configs/benchmark_final.yaml \
  --override configs/scenario_default_hard.yaml \
  --model-path outputs/formal/train/hard_best/model.zip 2>&1 | tee outputs/formal/logs/formal_suite.log
```

### 5.5 一键脚本

```bash
bash scripts/run_formal_experiments.sh all
```

## 6. manifest 与日志

`scripts/run_formal_experiments.sh` 会生成：

- `outputs/formal/manifest.jsonl`
- `outputs/formal/logs/*.log`

其中每条 manifest 至少记录：

- 实验名
- 命令
- 日志路径

## 7. 当前是否可以正式开跑

可以。

建议在服务器上先单独跑一个 `hard_seed42` 验证 GPU 占用、日志输出和目录权限，然后按本文档顺序执行整套实验。
