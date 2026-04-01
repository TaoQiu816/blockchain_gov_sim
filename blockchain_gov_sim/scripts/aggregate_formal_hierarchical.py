"""聚合正式多 seed 分层实验结果。"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


CORE_METRICS = [
    "TPS",
    "mean_latency",
    "unsafe_rate",
    "policy_invalid_rate",
    "structural_infeasible_rate",
    "used_backstop_template_rate",
    "template_dominant_ratio",
    "low_action_dominant_ratio",
]


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    if len(values) == 1:
        return float(mean), 0.0
    var = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return float(mean), float(math.sqrt(var))


def _distribution_rows(entries: list[dict[str, Any]], variant: str, scenario: str, controller: str, dist_type: str) -> list[dict[str, Any]]:
    bucket: dict[str, list[float]] = defaultdict(list)
    for entry in entries:
        for item in entry:
            bucket[str(item["action"])].append(float(item.get("ratio", 0.0)))
    rows: list[dict[str, Any]] = []
    for action, values in sorted(bucket.items()):
        mean, std = _mean_std(values)
        rows.append(
            {
                "variant": variant,
                "scenario": scenario,
                "controller": controller,
                "distribution_type": dist_type,
                "action": action,
                "mean": mean,
                "std": std,
                "seeds_covered": len(values),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", required=True, type=int)
    args = parser.parse_args()

    root = Path(args.output_root)
    run_root = root / "hierarchical"
    aggregate_root = root / "aggregate"
    aggregate_root.mkdir(parents=True, exist_ok=True)

    per_seed_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {"variants": {}, "seeds": [int(seed) for seed in args.seeds]}

    for variant in args.variants:
        variant_rows: list[dict[str, Any]] = []
        variant_distribution_inputs: dict[tuple[str, str, str], list[list[dict[str, Any]]]] = defaultdict(list)
        for seed in args.seeds:
            run_dir = run_root / f"formal_{variant}_seed{seed}"
            compare_csv = run_dir / "hard_eval" / "hard_compare.csv"
            compare_json = run_dir / "hard_eval" / "hard_compare.json"
            if not compare_csv.exists() or not compare_json.exists():
                continue
            compare_df = pd.read_csv(compare_csv)
            compare_df["variant"] = variant
            compare_df["seed"] = int(seed)
            variant_rows.extend(compare_df.to_dict(orient="records"))

            payload = json.loads(compare_json.read_text())
            for scenario, scenario_metrics in payload.get("scenarios", {}).items():
                for controller, metrics in scenario_metrics.items():
                    variant_distribution_inputs[(scenario, controller, "executed_high_template_distribution")].append(
                        list(metrics.get("executed_high_template_distribution", []))
                    )
                    variant_distribution_inputs[(scenario, controller, "executed_low_action_distribution")].append(
                        list(metrics.get("executed_low_action_distribution", []))
                    )

        if not variant_rows:
            manifest["variants"][variant] = {"status": "missing"}
            continue

        variant_df = pd.DataFrame(variant_rows)
        per_seed_rows.extend(variant_rows)
        variant_aggregate_dir = aggregate_root / variant
        variant_aggregate_dir.mkdir(parents=True, exist_ok=True)
        variant_df.to_csv(variant_aggregate_dir / "hard_compare_multiseed.csv", index=False)

        grouped = variant_df.groupby(["scenario", "controller"], as_index=False)
        for (scenario, controller), part in grouped:
            row: dict[str, Any] = {
                "variant": variant,
                "scenario": scenario,
                "controller": controller,
                "seeds_covered": int(part["seed"].nunique()),
            }
            for metric in CORE_METRICS:
                values = [float(value) for value in part[metric].tolist()]
                mean, std = _mean_std(values)
                row[f"{metric}_mean"] = mean
                row[f"{metric}_std"] = std
            aggregate_rows.append(row)

        for (scenario, controller, dist_type), entries in sorted(variant_distribution_inputs.items()):
            distribution_rows.extend(_distribution_rows(entries, variant, scenario, controller, dist_type))

        aggregate_df = pd.DataFrame([row for row in aggregate_rows if row["variant"] == variant])
        aggregate_df.to_csv(variant_aggregate_dir / "hard_compare_aggregate.csv", index=False)
        if distribution_rows:
            pd.DataFrame([row for row in distribution_rows if row["variant"] == variant]).to_csv(
                variant_aggregate_dir / "governance_distribution_aggregate.csv",
                index=False,
            )
        manifest["variants"][variant] = {
            "status": "ok",
            "aggregate_csv": str(variant_aggregate_dir / "hard_compare_aggregate.csv"),
            "multiseed_csv": str(variant_aggregate_dir / "hard_compare_multiseed.csv"),
        }

    if per_seed_rows:
        pd.DataFrame(per_seed_rows).to_csv(aggregate_root / "all_variants_hard_compare_multiseed.csv", index=False)
    if aggregate_rows:
        pd.DataFrame(aggregate_rows).to_csv(aggregate_root / "all_variants_hard_compare_aggregate.csv", index=False)
    if distribution_rows:
        pd.DataFrame(distribution_rows).to_csv(aggregate_root / "all_variants_governance_distribution_aggregate.csv", index=False)
    (aggregate_root / "aggregate_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
