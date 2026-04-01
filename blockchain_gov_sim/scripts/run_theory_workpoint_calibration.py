"""最小工作点校准与 smoke 复验脚本。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gov_sim.utils.io import ensure_dir, load_config, write_json


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(payload, file, sort_keys=False, allow_unicode=True)


def _run_smoke(
    *,
    base_config: Path,
    train_config: Path,
    calibration_config: Path,
    override_config: Path,
    seed: int,
    run_name: str,
    output_root: Path,
) -> None:
    cmd = [
        sys.executable,
        "scripts/train_hierarchical_formal.py",
        "--config",
        str(base_config),
        "--train-config",
        str(train_config),
        "--override",
        str(calibration_config),
        "--override",
        str(override_config),
        "--variant",
        "final",
        "--seed",
        str(seed),
        "--run-name",
        run_name,
        "--output-root",
        str(output_root),
    ]
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def _load_candidate_summary(root_dir: Path, run_name: str) -> dict[str, Any]:
    run_dir = root_dir / "hierarchical" / run_name
    with (run_dir / "stage2_high_train" / "training_export_summary.json").open("r", encoding="utf-8") as file:
        stage2 = json.load(file)
    with (run_dir / "hard_eval" / "hard_compare.json").open("r", encoding="utf-8") as file:
        hard_eval = json.load(file)
    with (run_dir / "hierarchical_summary.json").open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    return {
        "run_dir": str(run_dir),
        "stage2": stage2,
        "hard_eval": hard_eval,
        "manifest": manifest,
    }


def _top_high_ratio(stage2_summary: dict[str, Any]) -> float:
    usages = stage2_summary.get("step_stats", {}).get("high_template_usage", [])
    if not usages:
        return 1.0
    return float(max(float(item.get("ratio", 0.0)) for item in usages))


def _top_high_distribution(stage2_summary: dict[str, Any], top_k: int = 3) -> str:
    usages = stage2_summary.get("step_stats", {}).get("high_template_usage", [])
    if not usages:
        return ""
    parts = []
    for item in usages[:top_k]:
        parts.append(f"{item['template']}:{float(item['ratio']):.2f}")
    return ", ".join(parts)


def _extract_row(candidate: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    ep = summary["stage2"]["episode_stats"]
    hard = summary["hard_eval"]["scenarios"]
    row = {
        "candidate_id": candidate["candidate_id"],
        "phase": candidate["phase"],
        "seed": candidate["seed"],
        "h_min": candidate["h_min"],
        "h_warn": candidate["h_warn"],
        "lambda_h": candidate["lambda_h"],
        "timeout_w": candidate["timeout_w"],
        "margin_w": candidate["margin_w"],
        "stage2_unsafe": float(ep["mean_unsafe_rate"]),
        "stage2_timeout": float(ep.get("mean_timeout_rate", 0.0)),
        "stage2_structural_infeasible": float(ep.get("mean_structural_infeasible_rate", 0.0)),
        "stage2_lambda_final": float(ep["final_lambda"]),
        "malicious_unsafe": float(hard["malicious_burst"]["hierarchical_learned"]["unsafe_rate"]),
        "high_rtt_unsafe": float(hard["high_rtt_burst"]["hierarchical_learned"]["unsafe_rate"]),
        "high_rtt_timeout": float(hard["high_rtt_burst"]["hierarchical_learned"]["timeout_failure_rate"]),
        "top_high_ratio": _top_high_ratio(summary["stage2"]),
        "high_template_distribution": _top_high_distribution(summary["stage2"]),
        "run_dir": summary["run_dir"],
    }
    row["primary_score"] = row["malicious_unsafe"] + row["high_rtt_unsafe"]
    row["secondary_score"] = row["stage2_lambda_final"]
    row["tertiary_score"] = row["top_high_ratio"]
    return row


def _candidate_override(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_name": candidate["run_name"],
        "env": {
            "h_min": float(candidate["h_min"]),
            "h_warn": float(candidate["h_warn"]),
            "cost_weights": {
                "timeout": float(candidate["timeout_w"]),
                "margin": float(candidate["margin_w"]),
            },
        },
        "chain": {
            "lambda_h": float(candidate["lambda_h"]),
            "h_warn": float(candidate["h_warn"]),
        },
    }


def _select_best(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return min(
        rows,
        key=lambda row: (
            round(float(row["primary_score"]), 6),
            round(float(row["secondary_score"]), 6),
            round(float(row["tertiary_score"]), 6),
            round(float(row["stage2_structural_infeasible"]), 6),
        ),
    )


def _build_phase_a_candidates(seed: int) -> list[dict[str, Any]]:
    h_min_values = (0.64, 0.66, 0.68)
    h_warn_gaps = (0.08, 0.10)
    lambda_h_values = (0.25, 0.35, 0.45)
    candidates: list[dict[str, Any]] = []
    for h_min in h_min_values:
        for gap in h_warn_gaps:
            for lambda_h in lambda_h_values:
                candidate_id = f"A_hm{h_min:.2f}_gap{gap:.2f}_lh{lambda_h:.2f}"
                candidates.append(
                    {
                        "candidate_id": candidate_id,
                        "phase": "A",
                        "seed": seed,
                        "h_min": h_min,
                        "h_warn": min(h_min + gap, 0.99),
                        "lambda_h": lambda_h,
                        "timeout_w": 0.12,
                        "margin_w": 0.15,
                        "run_name": candidate_id,
                    }
                )
    return candidates


def _build_phase_b_candidates(best_a: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    timeout_values = (0.08, 0.12, 0.16)
    margin_values = (0.10, 0.15, 0.20)
    candidates: list[dict[str, Any]] = []
    for timeout_w in timeout_values:
        for margin_w in margin_values:
            candidate_id = f"B_tw{timeout_w:.2f}_mw{margin_w:.2f}"
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "phase": "B",
                    "seed": seed,
                    "h_min": float(best_a["h_min"]),
                    "h_warn": float(best_a["h_warn"]),
                    "lambda_h": float(best_a["lambda_h"]),
                    "timeout_w": timeout_w,
                    "margin_w": margin_w,
                    "run_name": candidate_id,
                }
            )
    return candidates


def _run_phase(
    *,
    candidates: list[dict[str, Any]],
    base_config: Path,
    train_config: Path,
    calibration_config: Path,
    output_root: Path,
    override_dir: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        override_path = override_dir / f"{candidate['candidate_id']}.yaml"
        override_payload = _candidate_override(candidate)
        _write_yaml(override_path, override_payload)
        _run_smoke(
            base_config=base_config,
            train_config=train_config,
            calibration_config=calibration_config,
            override_config=override_path,
            seed=int(candidate["seed"]),
            run_name=str(candidate["run_name"]),
            output_root=output_root,
        )
        summary = _load_candidate_summary(output_root, str(candidate["run_name"]))
        rows.append(_extract_row(candidate, summary))
    return rows


def _run_two_seed_recheck(
    *,
    best_row: dict[str, Any],
    base_config: Path,
    train_config: Path,
    calibration_config: Path,
    output_root: Path,
    override_dir: Path,
    seeds: tuple[int, int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        run_name = f"recheck_seed{seed}"
        candidate = {
            "candidate_id": run_name,
            "phase": "recheck",
            "seed": seed,
            "h_min": float(best_row["h_min"]),
            "h_warn": float(best_row["h_warn"]),
            "lambda_h": float(best_row["lambda_h"]),
            "timeout_w": float(best_row["timeout_w"]),
            "margin_w": float(best_row["margin_w"]),
            "run_name": run_name,
        }
        override_path = override_dir / f"{run_name}.yaml"
        _write_yaml(override_path, _candidate_override(candidate))
        _run_smoke(
            base_config=base_config,
            train_config=train_config,
            calibration_config=calibration_config,
            override_config=override_path,
            seed=seed,
            run_name=run_name,
            output_root=output_root,
        )
        summary = _load_candidate_summary(output_root, run_name)
        rows.append(_extract_row(candidate, summary))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--train-config", default="configs/train_hierarchical_theory_smoke.yaml")
    parser.add_argument("--calibration-config", default="configs/default_theory_workpoint_calibration.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recheck-seeds", nargs=2, type=int, default=(42, 43))
    parser.add_argument("--output-root", default="outputs/theory_workpoint_calibration")
    args = parser.parse_args()

    base_config = Path(args.config)
    train_config = Path(args.train_config)
    calibration_config = Path(args.calibration_config)
    output_root = ensure_dir(PROJECT_ROOT / args.output_root)
    work_dir = ensure_dir(output_root / "calibration_work")
    override_dir = ensure_dir(work_dir / "overrides")

    # 仅用于记录最终基线快照，训练仍通过 --override 精确覆盖允许参数。
    baseline = load_config(base_config, train_config, calibration_config)
    write_json(work_dir / "baseline_snapshot.json", baseline)

    phase_a_rows = _run_phase(
        candidates=_build_phase_a_candidates(seed=int(args.seed)),
        base_config=base_config,
        train_config=train_config,
        calibration_config=calibration_config,
        output_root=output_root,
        override_dir=override_dir,
    )
    best_a = _select_best(phase_a_rows)

    phase_b_rows = _run_phase(
        candidates=_build_phase_b_candidates(best_a=best_a, seed=int(args.seed)),
        base_config=base_config,
        train_config=train_config,
        calibration_config=calibration_config,
        output_root=output_root,
        override_dir=override_dir,
    )
    best_b = _select_best(phase_b_rows)

    recheck_rows = _run_two_seed_recheck(
        best_row=best_b,
        base_config=base_config,
        train_config=train_config,
        calibration_config=calibration_config,
        output_root=output_root,
        override_dir=override_dir,
        seeds=tuple(int(seed) for seed in args.recheck_seeds),
    )

    summary = {
        "phase_a_rows": phase_a_rows,
        "phase_b_rows": phase_b_rows,
        "best_phase_a": best_a,
        "best_workpoint": best_b,
        "recheck_rows": recheck_rows,
    }
    write_json(work_dir / "calibration_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
