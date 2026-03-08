#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

mkdir -p outputs/formal/logs
MANIFEST="outputs/formal/manifest.jsonl"
: > "${MANIFEST}"

run_and_log() {
  local name="$1"
  shift
  local logfile="outputs/formal/logs/${name}.log"
  printf '{"name":"%s","command":"%s","log":"%s"}\n' "${name}" "$*" "${logfile}" >> "${MANIFEST}"
  "$@" 2>&1 | tee "${logfile}"
}

train_ours_hard() {
  for seed in 42 52 62 72 82; do
    run_and_log "train_hard_seed${seed}" \
      python scripts/train_gov.py \
      --config configs/default.yaml \
      --train-config configs/train_final.yaml \
      --override configs/scenario_default_hard.yaml \
      --seed "${seed}" \
      --run-name "hard_seed${seed}" \
      --output-root outputs/formal
  done
  run_and_log "select_best_hard" \
    python scripts/select_best_model.py --train-root outputs/formal/train --prefix hard
}

train_ours_moderate() {
  for seed in 42 52 62; do
    run_and_log "train_moderate_seed${seed}" \
      python scripts/train_gov.py \
      --config configs/default.yaml \
      --train-config configs/train_final.yaml \
      --override configs/scenario_default_moderate.yaml \
      --seed "${seed}" \
      --run-name "moderate_seed${seed}" \
      --output-root outputs/formal
  done
  run_and_log "select_best_moderate" \
    python scripts/select_best_model.py --train-root outputs/formal/train --prefix moderate
}

run_formal_benchmarks() {
  local model_path="${1:-outputs/formal/train/hard_best/model.zip}"
  run_and_log "formal_suite" \
    python scripts/run_formal_suite.py \
    --config configs/default.yaml \
    --benchmark-config configs/benchmark_final.yaml \
    --override configs/scenario_default_hard.yaml \
    --model-path "${model_path}"
}

case "${1:-all}" in
  train_hard)
    train_ours_hard
    ;;
  train_moderate)
    train_ours_moderate
    ;;
  benchmarks)
    run_formal_benchmarks "${2:-outputs/formal/train/hard_best/model.zip}"
    ;;
  all)
    train_ours_hard
    train_ours_moderate
    run_formal_benchmarks "outputs/formal/train/hard_best/model.zip"
    ;;
  *)
    echo "usage: $0 [train_hard|train_moderate|benchmarks|all] [optional_model_path]" >&2
    exit 1
    ;;
esac
