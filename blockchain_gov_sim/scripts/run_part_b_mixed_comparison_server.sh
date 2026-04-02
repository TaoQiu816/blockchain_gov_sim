#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SYNC_SCRIPT="${SCRIPT_DIR}/git_sync_formal_artifacts.sh"
AUTO_SHUTDOWN="${AUTO_SHUTDOWN:-1}"

cd "${REPO_ROOT}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

ensure_file() {
  local path="$1"
  if [ ! -f "$path" ]; then
    echo "[run] missing required file: $path" >&2
    exit 1
  fi
}

sync_train_artifacts() {
  local tag="$1"
  local base="outputs/governance_dqn_formal/part_b_mixed/${tag}/seed42/train"
  ensure_file "${base}/best_checkpoint.pt"
  ensure_file "${base}/latest_checkpoint.pt"
  ensure_file "${base}/train_summary.json"
  ensure_file "${base}/train_log.csv"
  ensure_file "${base}/eval_log.csv"
  ensure_file "${base}/eval_log_by_scenario.csv"
  ensure_file "${base}/action_distribution.csv"
  ensure_file "${base}/episode_schedule.csv"
  "${SYNC_SCRIPT}" \
    "Add part B mixed ${tag} seed42 artifacts" \
    "${base}/train_summary.json" \
    "${base}/train_log.csv" \
    "${base}/train_log_steps.csv" \
    "${base}/eval_log.csv" \
    "${base}/eval_log_by_scenario.csv" \
    "${base}/best_eval_episode.csv" \
    "${base}/action_distribution.csv" \
    "${base}/episode_schedule.csv"
}

sync_comparison_artifacts() {
  local base="outputs/governance_dqn_formal/part_b_comparison_seed42"
  ensure_file "${base}/comparison_by_scenario.csv"
  ensure_file "${base}/comparison_by_scenario.md"
  ensure_file "${base}/comparison_metadata.json"
  ensure_file "${base}/baseline_compare/benchmark_by_scenario.csv"
  ensure_file "${base}/baseline_compare/benchmark_by_scenario_summary.json"
  "${SYNC_SCRIPT}" \
    "Add part B mixed comparison seed42 artifacts" \
    "${base}/comparison_by_scenario.csv" \
    "${base}/comparison_by_scenario.md" \
    "${base}/comparison_metadata.json" \
    "${base}/baseline_compare/benchmark_by_scenario.csv" \
    "${base}/baseline_compare/benchmark_by_scenario_summary.json"
}

run_train() {
  local variant="$1"
  local tag="$2"
  local checkpoint_mode="$3"
  echo "[run] training ${tag} ..."
  "${PYTHON_BIN}" scripts/train_governance_dqn.py \
    --config configs/default.yaml \
    --output-dir "outputs/governance_dqn_formal/part_b_mixed/${tag}/seed42/train" \
    --num-episodes 800 \
    --episode-length 120 \
    --total-env-steps 96000 \
    --eval-every 20 \
    --eval-episodes 20 \
    --seed 42 \
    --mixed-scenario \
    --checkpoint-mode "${checkpoint_mode}" \
    --agent-variant "${variant}" \
    --lr 5e-5 \
    --target-update-period 500 \
    --warmup-steps 10000 \
    --epsilon-decay-steps 120000 \
    --train-freq 2
  sync_train_artifacts "${tag}"
}

run_comparison() {
  echo "[run] evaluating mixed comparison ..."
  "${PYTHON_BIN}" scripts/eval_mixed_formal_comparison.py \
    --config configs/default.yaml \
    --proposed-checkpoint outputs/governance_dqn_formal/part_b_mixed/proposed/seed42/train/best_checkpoint.pt \
    --vanilla-checkpoint outputs/governance_dqn_formal/part_b_mixed/vanilla_dqn/seed42/train/best_checkpoint.pt \
    --no-constraint-checkpoint outputs/governance_dqn_formal/part_b_mixed/no_constraint_dueling_double_dqn/seed42/train/best_checkpoint.pt \
    --output-dir outputs/governance_dqn_formal/part_b_comparison_seed42 \
    --episodes 20 \
    --seed 42
  sync_comparison_artifacts
}

echo "[run] repo root: ${REPO_ROOT}"
echo "[run] branch: $(git branch --show-current)"
echo "[run] auto_shutdown: ${AUTO_SHUTDOWN}"

run_train proposed proposed periodic
run_train vanilla_dqn vanilla_dqn overwrite
run_train no_constraint_dueling_double_dqn no_constraint_dueling_double_dqn overwrite
run_comparison

if [ "${AUTO_SHUTDOWN}" = "1" ]; then
  echo "[run] all part B mixed comparison jobs completed; shutdown in 2 minutes."
  shutdown -h +2 "governance_dqn part B mixed comparison finished" || (sleep 120 && poweroff) &
else
  echo "[run] all part B mixed comparison jobs completed; auto shutdown disabled."
fi
