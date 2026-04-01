#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/formal_multiseed}"
mkdir -p "${OUTPUT_ROOT}/logs"

SEEDS=(42 43 44 45 46)
VARIANTS=(final no_dynamic_theta single_dim_trust no_context_fusion)

run_one() {
  local variant="$1"
  local seed="$2"
  local run_name="formal_${variant}_seed${seed}"
  local logfile="${OUTPUT_ROOT}/logs/${run_name}.log"
  shift 2
  python scripts/train_hierarchical_formal.py \
    --config configs/default.yaml \
    --train-config configs/train_hierarchical_formal_final.yaml \
    --variant "${variant}" \
    --seed "${seed}" \
    --run-name "${run_name}" \
    --output-root "${OUTPUT_ROOT}" \
    "$@" 2>&1 | tee "${logfile}"
}

run_variant() {
  local variant="$1"
  shift
  for seed in "${SEEDS[@]}"; do
    case "${variant}" in
      final)
        run_one "${variant}" "${seed}"
        ;;
      no_dynamic_theta)
        run_one "${variant}" "${seed}"
        ;;
      single_dim_trust)
        run_one "${variant}" "${seed}" --override configs/formal_ablation_single_dim_trust.yaml
        ;;
      no_context_fusion)
        run_one "${variant}" "${seed}" --override configs/formal_ablation_no_context_fusion.yaml
        ;;
      *)
        echo "unknown variant: ${variant}" >&2
        exit 1
        ;;
    esac
  done
}

aggregate() {
  python scripts/aggregate_formal_hierarchical.py \
    --output-root "${OUTPUT_ROOT}" \
    --variants "${VARIANTS[@]}" \
    --seeds "${SEEDS[@]}"
}

case "${1:-all}" in
  final|no_dynamic_theta|single_dim_trust|no_context_fusion)
    run_variant "$1"
    aggregate
    ;;
  all)
    for variant in "${VARIANTS[@]}"; do
      run_variant "${variant}"
      aggregate
    done
    ;;
  aggregate)
    aggregate
    ;;
  *)
    echo "usage: $0 [all|final|no_dynamic_theta|single_dim_trust|no_context_fusion|aggregate]" >&2
    exit 1
    ;;
esac
