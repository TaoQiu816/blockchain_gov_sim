#!/usr/bin/env bash
set -euo pipefail

# 将 `outputs/formal` 中的正式实验结果整理到可提交到 Git 的 `results/formal_release/`。
# 这样既保留正式结果的同步能力，又避免把 smoke / audit / 临时输出全部放进仓库。

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="${ROOT_DIR}/outputs/formal"
DST_DIR="${ROOT_DIR}/results/formal_release"

if [[ ! -d "${SRC_DIR}" ]]; then
  echo "formal output directory not found: ${SRC_DIR}" >&2
  echo "Run formal experiments first, then sync results." >&2
  exit 1
fi

mkdir -p "${DST_DIR}"

copy_dir_if_exists() {
  local rel="$1"
  if [[ -d "${SRC_DIR}/${rel}" ]]; then
    mkdir -p "${DST_DIR}"
    rsync -a --delete "${SRC_DIR}/${rel}/" "${DST_DIR}/${rel}/"
  fi
}

copy_file_if_exists() {
  local rel="$1"
  if [[ -f "${SRC_DIR}/${rel}" ]]; then
    mkdir -p "$(dirname "${DST_DIR}/${rel}")"
    cp -f "${SRC_DIR}/${rel}" "${DST_DIR}/${rel}"
  fi
}

# 同步正式实验主目录
copy_dir_if_exists "train"
copy_dir_if_exists "main_compare"
copy_dir_if_exists "malicious_scan"
copy_dir_if_exists "dynamic_attacks"
copy_dir_if_exists "load_shock"
copy_dir_if_exists "high_rtt"
copy_dir_if_exists "high_churn"
copy_dir_if_exists "ablation"
copy_dir_if_exists "logs"

# 同步 manifest
copy_file_if_exists "manifest.json"
copy_file_if_exists "manifest.jsonl"

cat > "${DST_DIR}/README.md" <<'EOF'
# Formal Results Release

该目录是 `outputs/formal/` 的可提交归档版本，用于：

- 同步正式实验结果到 Git 仓库
- 保留 `csv/json/png/pdf/model.zip/log` 等论文主结果
- 避免把 smoke、临时检查和其它非正式输出一起提交

更新方式：

```bash
bash scripts/sync_formal_results.sh
```
EOF

echo "synced formal results to: ${DST_DIR}" >&2
