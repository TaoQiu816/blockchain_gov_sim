# blockchain_gov_sim

面向硕士论文第四章的独立联盟链治理仿真系统。研究对象是“联盟链链侧治理”，不是 DAG 卸载本身；本工程不依赖第三章代码、日志或输出。

说明：
- 主算法固定为 `Maskable PPO-Lagrangian`。
- `Joint-Trust-Consensus-like`、`AE-PBFT-like`、`DVRC-like`、`TwoLayer-LPBFT-like` 等 baseline 为统一仿真框架下的机制抽象版，不是逐篇论文 1:1 复刻。

## 研究闭环

环境单周期执行流程：

1. 更新在线/离线状态与负载到达。
2. 生成服务/共识/推荐/稳定性证据。
3. 更新四维信誉 `svc/con/rec/stab`。
4. 根据信誉与稳定性构造资格集。
5. 策略输出扁平离散动作 `a=(m,b,tau,theta)`。
6. 动态 action mask 屏蔽非法动作。
7. 按 soft sortition 或 baseline 指定机制生成委员会。
8. 计算链侧性能、安全状态、reward 与 cost。
9. 返回 `obs/reward/cost/info`，进入下一周期。

## 目录

```text
blockchain_gov_sim/
├─ README.md
├─ requirements.txt
├─ pytest.ini
├─ configs/
├─ gov_sim/
├─ scripts/
└─ tests/
```

## 安装

```bash
cd /Users/qiutao/研/毕设/毕设/blockchainSim/blockchain_gov_sim
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

服务器部署可直接使用：

```bash
conda env create -f environment.server.yml
conda activate blockchain_gov_sim
```

如果你已经有一套现成 conda 环境，并且它的包清单与 `base_requirements.txt` 基本一致，
不要重新建环境，直接在该环境里补项目所需增量依赖即可：

```bash
cd /Users/qiutao/研/毕设/毕设/blockchainSim/blockchain_gov_sim
conda activate <你的现有环境>
pip install -r base_env_delta_requirements.txt
```

或者直接执行：

```bash
bash scripts/install_on_base_env.sh
```

说明：
- 当前 `base_requirements.txt` 已包含 `python=3.10`、`torch=2.1.2+cu118`、`pandas`、`matplotlib`、`scikit-learn` 等基础依赖；
- 但未包含本项目必需的 `stable-baselines3` 与 `sb3-contrib`；
- 同时其中的 `gymnasium=1.2.3` 与 `stable-baselines3 2.4.x` 不兼容，因此 `base_env_delta_requirements.txt` 会把它调整到 `1.0.0`。

如果你只想用 pip 锁定依赖：

```bash
pip install -r requirements.lock.txt
```

当前本机激活的 conda base 环境快照也已导出到：

- `environment.current-conda-base.yml`

## 配置文件说明

- `configs/default.yaml`
  作用：全局默认配置，含环境、信誉、链模型、场景、攻击、agent、eval、baseline。
- `configs/train.yaml`
  作用：覆盖训练超参数，例如 `total_timesteps`、`batch_size`。
- `configs/eval.yaml`
  作用：覆盖评估 episode 数与默认场景。
- `configs/scenarios.yaml`
  作用：预定义场景片段，便于后续扩展脚本做场景批量运行。
- `PARAMETER_CALIBRATION.md`
  作用：记录第四章参数分类、pilot 结果、默认工作点和正式训练建议。
- `FORMAL_EXPERIMENTS.md`
  作用：固定第四章正式实验矩阵、单 GPU 调度顺序、运行命令与输出规范。

## 训练

主算法为 `Maskable PPO-Lagrangian`：
- 使用 `sb3-contrib` 的 `MaskablePPO` 处理离散动作与 action mask；
- 额外维护独立 `cost critic`；
- 额外维护拉格朗日乘子 `lambda >= 0`；
- actor 更新使用 `A_r - lambda * A_c`；
- cost 不会被简单并入 reward。
- 设备选择遵循：`CUDA -> MPS -> CPU`。
- 也就是说：服务器有 NVIDIA GPU 时优先用 GPU；Apple Silicon 本地优先用 MPS；没有 GPU 则自动回退 CPU。

```bash
python scripts/train_gov.py --config configs/default.yaml --train-config configs/train.yaml
```

训练输出目录：

- 常规运行：`outputs/train/<run_name>/...`
- 第四章正式实验：`outputs/formal/train/<run_name>/...`

正式实验训练目录至少包含：

- `train_log.csv`
- `train_metrics.csv`
- `train_summary.json`
- `reward_curve.png/.pdf`
- `cost_curve.png/.pdf`
- `unsafe_curve.png/.pdf`
- `lambda_curve.png/.pdf`
- `convergence_panel.png/.pdf`
- `training_diagnostics_panel.png/.pdf`
- `post_train_eval.csv`
- `post_train_eval_panel.png/.pdf`
- `post_train_action_trajectory.png/.pdf`
- `model.zip`

## 评估

```bash
python scripts/eval_gov.py \
  --config configs/default.yaml \
  --eval-config configs/eval.yaml \
  --model-path outputs/train/train_run/model.zip
```

也可以直接评估 baseline：

```bash
python scripts/eval_gov.py \
  --config configs/default.yaml \
  --eval-config configs/eval.yaml \
  --baseline "MultiRep-TopK-Static"
```

评估输出目录：

- `outputs/eval/<run_name>/eval_log.csv`
- `outputs/eval/<run_name>/eval_summary.json`
- `outputs/eval/<run_name>/latency_curve.png/.pdf`
- `outputs/eval/<run_name>/action_trajectory.png/.pdf`

## Benchmark

```bash
python scripts/run_benchmarks.py \
  --config configs/default.yaml \
  --eval-config configs/eval.yaml \
  --model-path outputs/train/train_run/model.zip
```

默认 benchmark 包含：

- 恶意节点渗透实验
- on-off / zigzag / collusion 攻击实验
- 阶跃负载实验
- 高 RTT / 高 churn 实验

benchmark 主要输出：

- `benchmark_log.csv`
- `benchmark_summary.json`
- `tps_vs_malicious_ratio.png/.pdf`
- `latency_vs_malicious_ratio.png/.pdf`
- `queue_recovery.png/.pdf`
- `action_trajectory.png/.pdf`
- `unsafe_vs_high_rtt.png/.pdf`
- `eligible_size_distribution.png/.pdf`
- 若干代表性 step 级日志 `*_steps.csv`

## Ablation

```bash
python scripts/run_ablation.py \
  --config configs/default.yaml \
  --train-config configs/train.yaml
```

支持的消融：

- 去掉上下文门控
- 去掉一致性/反合谋惩罚
- `Top-K` 替代 soft sortition
- 普通 PPO 替代 PPO-Lagrangian
- 去掉 `action mask`，仅保留非法动作惩罚

ablation 输出：

- `outputs/ablation/<run_name>/ablation_log.csv`
- `outputs/ablation/<run_name>/ablation_summary.json`
- `outputs/ablation/<run_name>/ablation_unsafe_bar.png/.pdf`
- `outputs/ablation/<run_name>/ablation_tps_bar.png/.pdf`

## Smoke Test 与 Pytest

快速跑通训练闭环：

```bash
python scripts/smoke_test.py
```

运行测试：

```bash
pytest -q
```

## Baseline 含义

- `Static-Param`
  固定治理参数，作为最简单参考线。
- `Heuristic-AIMD`
  基于队列积压与整体信誉的加性增/乘性减控制。
- `SingleRep-TopK-Static`
  只看 `svc` 单维信誉，并使用 Top-K 委员会。
- `MultiRep-TopK-Static`
  使用多维融合信誉，但委员会采用 Top-K 而不是软抽签。
- `Joint-Trust-Consensus-like`
  让总体信任风险与共识参数联动。
- `AE-PBFT-like`
  在高 RTT / 高 churn / 低信誉环境下采用更保守的信用驱动组建。
- `DVRC-like`
  根据推荐污染与合谋风险动态提高门槛。
- `TwoLayer-LPBFT-like`
  近似两层轻量 PBFT，偏向高门槛、小委员会。

## 输出目录说明

- `outputs/train/*`
  训练过程日志、曲线、模型、训练后评估结果。
- `outputs/eval/*`
  评估 step 日志、汇总指标、动作轨迹图。
- `outputs/benchmark/*`
  benchmark 汇总表、代表性 step 日志和对比图。
- `outputs/ablation/*`
  消融汇总表和对比柱状图。

当前仓库对“正式实验结果”的推荐同步路径已经归一化为：

- `outputs/formal/`

也就是说：
- 后续正式训练统一保存到 `outputs/formal/train/<run_name>/`
- 正式 benchmark / ablation 统一保存到 `outputs/formal/<experiment_group>/`
- Git 同步时直接提交 `blockchain_gov_sim/outputs/formal/`

同步前可执行：

```bash
bash scripts/sync_formal_results.sh
```

该脚本会：
- 清理 `.DS_Store`、`.ipynb_checkpoints`
- 给 `outputs/formal/train/*` 自动补齐综合收敛图和评估图

## 常见问题

- `ModuleNotFoundError: gov_sim`
  请在项目根目录执行脚本，或先 `cd blockchain_gov_sim`。
- `tensorboard is not installed`
  当前工程会自动关闭 tensorboard，不会阻塞训练。
- benchmark 或 ablation 太慢
  可临时把 `configs/eval.yaml` 中 `episodes` 调小，或在 `default.yaml` 中降低 `episode_length`。
- 想切换场景
  可以直接覆盖 `scenario.default_name`，或在脚本层加载 `configs/scenarios.yaml` 中的片段。
