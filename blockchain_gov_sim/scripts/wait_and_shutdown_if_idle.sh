#!/usr/bin/env bash
set -euo pipefail

# 在“当前命令已经结束”之后调用本脚本。
# 如果仓库下没有其他实验进程在跑，则等待两分钟后关机；
# 若等待期间检测到新的实验启动，则取消关机。
#
# 设计目标：
# 1. 不要求训练脚本本身知道系统电源状态；
# 2. 不误伤同仓库内仍在运行的其他实验；
# 3. 只在“整个实验集群空闲”时才触发关机。

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WAIT_SECONDS="${1:-120}"

is_any_experiment_running() {
  pgrep -af "python .*${ROOT_DIR}/scripts/(train_gov|eval_gov|run_benchmarks|run_ablation|run_formal_suite|smoke_test|select_best_model)\.py" >/dev/null
}

echo "[auto-shutdown] checking for running experiments under ${ROOT_DIR}" >&2
if is_any_experiment_running; then
  echo "[auto-shutdown] another experiment is still running; skip shutdown." >&2
  exit 0
fi

echo "[auto-shutdown] no experiment detected, waiting ${WAIT_SECONDS}s before shutdown." >&2
sleep "${WAIT_SECONDS}"

if is_any_experiment_running; then
  echo "[auto-shutdown] new experiment detected during wait window; cancel shutdown." >&2
  exit 0
fi

echo "[auto-shutdown] still idle after ${WAIT_SECONDS}s, shutting down now." >&2
shutdown -h now
