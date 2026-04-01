"""汇总 high credit assignment 修正前后 smoke 结果。"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BEFORE_ROOT = PROJECT_ROOT / "outputs/high_state_refined_smoke/hierarchical"
AFTER_ROOT = PROJECT_ROOT / "outputs/high_credit_fixed_smoke/hierarchical"
OUT_DIR = PROJECT_ROOT / "outputs/final_chapter4_closure"


def _load_summary(run_dir: Path, version: str, seed: int) -> dict:
    summary = json.load((run_dir / "stage2_high_train" / "training_export_summary.json").open("r", encoding="utf-8"))
    hard = json.load((run_dir / "hard_eval" / "hard_compare.json").open("r", encoding="utf-8"))["scenarios"]
    episode_stats = summary["episode_stats"]
    step_stats = summary["step_stats"]
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


def _scene_usage(run_dir: Path, version: str, seed: int) -> list[dict]:
    rows: list[dict] = []
    for csv_path in sorted((run_dir / "hard_eval").glob("*_hierarchical_learned.csv")):
        scenario = csv_path.stem.replace("_hierarchical_learned", "")
        frame = pd.read_csv(csv_path)
        counter = Counter(str(item) for item in frame.get("executed_high_template", []) if str(item))
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


def _aggregate_scene_usage(frame: pd.DataFrame) -> pd.DataFrame:
    grouped: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for row in frame.to_dict(orient="records"):
        for item in json.loads(row["top3_high_templates"]):
            grouped[(row["version"], row["scenario"])][item["template"]] += float(item["ratio"])
    rows: list[dict] = []
    for (version, scenario), counter in sorted(grouped.items()):
        total = sum(counter.values()) or 1.0
        top3 = [{"template": action, "ratio": float(value / total)} for action, value in counter.most_common(3)]
        rows.append(
            {
                "version": version,
                "scenario": scenario,
                "top3_high_templates_avg": json.dumps(top3, ensure_ascii=False),
                "dominant_template": top3[0]["template"] if top3 else "",
                "dominant_ratio": top3[0]["ratio"] if top3 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict] = []
    scene_rows: list[dict] = []
    before_runs = {42: BEFORE_ROOT / "high_state_refined_seed42", 43: BEFORE_ROOT / "high_state_refined_seed43"}
    after_runs = {42: AFTER_ROOT / "high_credit_fixed_seed42", 43: AFTER_ROOT / "high_credit_fixed_seed43"}

    for seed, run_dir in before_runs.items():
        summary_rows.append(_load_summary(run_dir, "before_fix", seed))
        scene_rows.extend(_scene_usage(run_dir, "before_fix", seed))
    for seed, run_dir in after_runs.items():
        summary_rows.append(_load_summary(run_dir, "after_fix", seed))
        scene_rows.extend(_scene_usage(run_dir, "after_fix", seed))

    summary_df = pd.DataFrame(summary_rows).sort_values(["version", "seed"])
    scene_df = pd.DataFrame(scene_rows).sort_values(["version", "seed", "scenario"])
    agg_df = (
        summary_df.groupby("version")
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
    scene_agg_df = _aggregate_scene_usage(scene_df)

    summary_df.to_csv(OUT_DIR / "high_credit_fixed_smoke_comparison.csv", index=False)
    scene_df.to_csv(OUT_DIR / "high_credit_fixed_scene_template_usage_by_seed.csv", index=False)
    scene_agg_df.to_csv(OUT_DIR / "high_credit_fixed_scene_template_usage_comparison.csv", index=False)

    lines = [
        "# High Credit Fixed Smoke Comparison",
        "",
        "## Aggregate",
        "",
        agg_df.to_markdown(index=False),
        "",
        "## Scene-Wise High Template Usage",
        "",
        scene_agg_df.to_markdown(index=False),
    ]
    (OUT_DIR / "HIGH_CREDIT_FIXED_SMOKE_JUDGMENT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
