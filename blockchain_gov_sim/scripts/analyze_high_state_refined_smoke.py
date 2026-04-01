"""汇总 high-state 改前/改后的 smoke 结果。"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OLD_ROOT = PROJECT_ROOT / "outputs/theory_workpoint_calibration/hierarchical"
NEW_ROOT = PROJECT_ROOT / "outputs/high_state_refined_smoke/hierarchical"
OUT_DIR = PROJECT_ROOT / "outputs/final_chapter4_closure"


def _load_training_row(run_dir: Path, version: str, seed: int) -> dict:
    summary = json.load((run_dir / "stage2_high_train" / "training_export_summary.json").open("r", encoding="utf-8"))
    episode_stats = summary["episode_stats"]
    step_stats = summary["step_stats"]
    hard = json.load((run_dir / "hard_eval" / "hard_compare.json").open("r", encoding="utf-8"))["scenarios"]
    return {
        "version": version,
        "seed": int(seed),
        "stage2_unsafe": float(episode_stats["mean_unsafe_rate"]),
        "stage2_timeout": float(episode_stats.get("mean_timeout_rate", 0.0)),
        "stage2_structural_infeasible": float(episode_stats.get("mean_structural_infeasible_rate", 0.0)),
        "lambda_final": float(episode_stats["final_lambda"]),
        "malicious_burst_unsafe": float(hard["malicious_burst"]["hierarchical_learned"]["unsafe_rate"]),
        "high_rtt_burst_unsafe": float(hard["high_rtt_burst"]["hierarchical_learned"]["unsafe_rate"]),
        "high_rtt_burst_timeout": float(hard["high_rtt_burst"]["hierarchical_learned"]["timeout_failure_rate"]),
        "stage2_high_template_usage": json.dumps(step_stats.get("high_template_usage", []), ensure_ascii=False),
        "run_dir": str(run_dir),
    }


def _scenario_template_summary(run_dir: Path, version: str, seed: int) -> list[dict]:
    rows: list[dict] = []
    hard_eval_dir = run_dir / "hard_eval"
    for csv_path in sorted(hard_eval_dir.glob("*_hierarchical_learned.csv")):
        scenario = csv_path.stem.replace("_hierarchical_learned", "")
        frame = pd.read_csv(csv_path)
        if "executed_high_template" not in frame.columns:
            continue
        counter = Counter(str(value) for value in frame["executed_high_template"] if str(value))
        total = max(sum(counter.values()), 1)
        top3 = [{"template": action, "ratio": float(count / total)} for action, count in counter.most_common(3)]
        rows.append(
            {
                "version": version,
                "seed": int(seed),
                "scenario": scenario,
                "top3_high_templates": json.dumps(top3, ensure_ascii=False),
                "dominant_template": top3[0]["template"] if top3 else "",
                "dominant_ratio": top3[0]["ratio"] if top3 else 0.0,
            }
        )
    return rows


def _aggregate_scenario_templates(frame: pd.DataFrame) -> pd.DataFrame:
    bucket: list[dict] = []
    grouped: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for row in frame.to_dict(orient="records"):
        items = json.loads(row["top3_high_templates"])
        for item in items:
            grouped[(row["version"], row["scenario"])][item["template"]] += float(item["ratio"])
    for (version, scenario), counter in sorted(grouped.items()):
        total = sum(counter.values()) or 1.0
        top3 = [{"template": action, "ratio": float(value / total)} for action, value in counter.most_common(3)]
        bucket.append(
            {
                "version": version,
                "scenario": scenario,
                "top3_high_templates_avg": json.dumps(top3, ensure_ascii=False),
                "dominant_template": top3[0]["template"] if top3 else "",
                "dominant_ratio": top3[0]["ratio"] if top3 else 0.0,
            }
        )
    return pd.DataFrame(bucket)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    scenario_rows: list[dict] = []
    old_runs = {42: OLD_ROOT / "recheck_seed42", 43: OLD_ROOT / "recheck_seed43"}
    new_runs = {42: NEW_ROOT / "high_state_refined_seed42", 43: NEW_ROOT / "high_state_refined_seed43"}

    for seed, run_dir in old_runs.items():
        rows.append(_load_training_row(run_dir, "old", seed))
        scenario_rows.extend(_scenario_template_summary(run_dir, "old", seed))
    for seed, run_dir in new_runs.items():
        rows.append(_load_training_row(run_dir, "new", seed))
        scenario_rows.extend(_scenario_template_summary(run_dir, "new", seed))

    summary_frame = pd.DataFrame(rows).sort_values(["version", "seed"])
    scenario_frame = pd.DataFrame(scenario_rows).sort_values(["version", "seed", "scenario"])
    agg = (
        summary_frame.groupby("version")
        .agg(
            stage2_unsafe=("stage2_unsafe", "mean"),
            stage2_timeout=("stage2_timeout", "mean"),
            stage2_structural_infeasible=("stage2_structural_infeasible", "mean"),
            lambda_final=("lambda_final", "mean"),
            malicious_burst_unsafe=("malicious_burst_unsafe", "mean"),
            high_rtt_burst_unsafe=("high_rtt_burst_unsafe", "mean"),
            high_rtt_burst_timeout=("high_rtt_burst_timeout", "mean"),
        )
        .reset_index()
        .sort_values("version")
    )
    scenario_agg = _aggregate_scenario_templates(scenario_frame)

    summary_frame.to_csv(OUT_DIR / "high_state_smoke_comparison.csv", index=False)
    scenario_frame.to_csv(OUT_DIR / "high_state_scene_template_usage_by_seed.csv", index=False)
    scenario_agg.to_csv(OUT_DIR / "high_state_scene_template_usage_comparison.csv", index=False)

    lines = [
        "# High-State Smoke Comparison",
        "",
        "## Aggregate",
        "",
        agg.to_markdown(index=False),
        "",
        "## Scene-Wise High Template Usage",
        "",
        scenario_agg.to_markdown(index=False),
    ]
    (OUT_DIR / "HIGH_STATE_SMOKE_JUDGMENT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
