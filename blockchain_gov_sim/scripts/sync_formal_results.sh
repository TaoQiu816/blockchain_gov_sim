#!/usr/bin/env bash
set -euo pipefail

# 当前仓库已经直接跟踪 `outputs/formal/`。
# 该脚本保留为“正式结果清理与补图入口”，不再复制到其它目录。

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="${ROOT_DIR}/outputs/formal"

if [[ ! -d "${SRC_DIR}" ]]; then
  echo "formal output directory not found: ${SRC_DIR}" >&2
  echo "Run formal experiments first, then sync results." >&2
  exit 1
fi

find "${SRC_DIR}" -name '.DS_Store' -delete
find "${SRC_DIR}" -name '.ipynb_checkpoints' -type d -prune -exec rm -rf {} +

if [[ -d "${SRC_DIR}/train" ]]; then
  while IFS= read -r run_dir; do
    if [[ -f "${run_dir}/train_log.csv" ]]; then
      python "${ROOT_DIR}/scripts/postprocess_train_run.py" --run-dir "${run_dir}"
    fi
  done < <(find "${SRC_DIR}/train" -mindepth 1 -maxdepth 1 -type d | sort)
fi

echo "formal outputs cleaned and postprocessed in-place: ${SRC_DIR}" >&2
