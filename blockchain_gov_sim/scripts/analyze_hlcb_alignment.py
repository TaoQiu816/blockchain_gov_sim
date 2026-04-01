"""h_LCB 与真实委员会诚实比例对齐分析。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


SCENES = ("load_shock", "malicious_burst", "high_rtt_burst", "churn_burst")


def _load_eval_rows(root: Path, seeds: list[int]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for seed in seeds:
        seed_dir = root / f"recheck_seed{seed}" / "hard_eval"
        for scene in SCENES:
            path = seed_dir / f"{scene}_hierarchical_learned.csv"
            frame = pd.read_csv(path)
            frame["seed"] = int(seed)
            frame["scene"] = str(scene)
            frames.append(frame)
    if not frames:
        raise RuntimeError(f"No hard-eval frames found under {root}")
    return pd.concat(frames, ignore_index=True)


def _template_summary(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        frame.groupby(["m", "theta"])
        .agg(
            sample_count=("h_LCB", "size"),
            mean_h_lcb=("h_LCB", "mean"),
            mean_committee_honest_ratio=("committee_honest_ratio", "mean"),
            unsafe_rate=("unsafe", "mean"),
        )
        .reset_index()
        .sort_values(["sample_count", "mean_h_lcb"], ascending=[False, True])
    )
    return grouped


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="outputs/theory_workpoint_calibration/hierarchical",
        help="包含 recheck_seed42/recheck_seed43 的目录",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    parser.add_argument(
        "--output-dir",
        default="outputs/final_chapter4_closure",
        help="分析产物输出目录",
    )
    args = parser.parse_args()

    root = Path(args.root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = _load_eval_rows(root=root, seeds=list(args.seeds))
    alignment = raw[
        [
            "scene",
            "seed",
            "m_e",
            "theta_e",
            "eligible_size",
            "committee_members",
            "committee_honest_ratio",
            "h_LCB",
            "unsafe",
            "timeout_failure",
            "structural_infeasible",
            "episode_seed",
            "episode",
            "step",
        ]
    ].rename(
        columns={
            "m_e": "m",
            "theta_e": "theta",
            "committee_members": "selected_committee",
            "timeout_failure": "timeout",
        }
    )
    alignment.to_csv(output_dir / "hlcb_honest_ratio_alignment.csv", index=False)

    mid = alignment[(alignment["h_LCB"] >= 0.62) & (alignment["h_LCB"] <= 0.66)]
    unsafe_rows = alignment[alignment["unsafe"] == 1]

    overall_mid_stats = {
        "count": int(len(mid)),
        "honest_mean": float(mid["committee_honest_ratio"].mean()),
        "honest_std": float(mid["committee_honest_ratio"].std()),
        "honest_p25": float(mid["committee_honest_ratio"].quantile(0.25)),
        "honest_p50": float(mid["committee_honest_ratio"].quantile(0.50)),
        "honest_p75": float(mid["committee_honest_ratio"].quantile(0.75)),
    }
    unsafe_stats = {
        "count": int(len(unsafe_rows)),
        "h_lcb_mean": float(unsafe_rows["h_LCB"].mean()),
        "h_lcb_std": float(unsafe_rows["h_LCB"].std()),
        "h_lcb_min": float(unsafe_rows["h_LCB"].min()),
        "h_lcb_max": float(unsafe_rows["h_LCB"].max()),
        "honest_mean": float(unsafe_rows["committee_honest_ratio"].mean()),
        "honest_std": float(unsafe_rows["committee_honest_ratio"].std()),
        "honest_min": float(unsafe_rows["committee_honest_ratio"].min()),
        "honest_max": float(unsafe_rows["committee_honest_ratio"].max()),
    }

    per_scene_lines: list[str] = []
    for scene in ("malicious_burst", "high_rtt_burst"):
        frame = alignment[alignment["scene"] == scene]
        scene_mid = frame[(frame["h_LCB"] >= 0.62) & (frame["h_LCB"] <= 0.66)]
        template_low = _template_summary(frame[frame["h_LCB"] < 0.64]).head(5)
        template_mid = _template_summary(scene_mid).head(5)
        per_scene_lines.append(f"### {scene}")
        per_scene_lines.append(
            f"- `h_LCB∈[0.62,0.66]` 样本数 `{len(scene_mid)}`，`committee_honest_ratio` 均值/标准差 `{scene_mid['committee_honest_ratio'].mean():.4f}/{scene_mid['committee_honest_ratio'].std():.4f}`，P25/P50/P75 为 `{scene_mid['committee_honest_ratio'].quantile(0.25):.4f}/{scene_mid['committee_honest_ratio'].quantile(0.5):.4f}/{scene_mid['committee_honest_ratio'].quantile(0.75):.4f}`。"
        )
        per_scene_lines.append("- `h_LCB<0.64` 的高频模板：")
        per_scene_lines.append("")
        per_scene_lines.append(template_low.to_markdown(index=False) if not template_low.empty else "_none_")
        per_scene_lines.append("")
        per_scene_lines.append("- `h_LCB∈[0.62,0.66]` 的模板对齐：")
        per_scene_lines.append("")
        per_scene_lines.append(template_mid.to_markdown(index=False) if not template_mid.empty else "_none_")
        per_scene_lines.append("")

    acceptability = overall_mid_stats["honest_p25"] >= 0.8 and unsafe_stats["honest_mean"] >= 0.95
    answers = [
        f"A. `h_min=0.64` {'具有' if acceptability else '不具有'}理论可接受性。",
        (
            "B. 该阈值可解释为 `h_LCB` 的保守下界校准阈值，而不是把真实安全门槛直接放松到 `0.64`。"
            if acceptability
            else "B. 当前不能把 `0.64` 解释成纯 `LCB` 校准阈值。"
        ),
        (
            "C. 不需要调整 `h_min/h_warn/lambda_h`。"
            if acceptability
            else "C. 唯一最小必要修正应回到 `lambda_h`，重新校准下界保守度。"
        ),
    ]

    lines = [
        "# HLCB Alignment Judgment",
        "",
        "## Overall",
        "",
        f"- `h_LCB∈[0.62,0.66]` 样本数 `{overall_mid_stats['count']}`。",
        f"- `committee_honest_ratio` 均值/标准差 `{overall_mid_stats['honest_mean']:.4f}/{overall_mid_stats['honest_std']:.4f}`。",
        f"- P25/P50/P75 为 `{overall_mid_stats['honest_p25']:.4f}/{overall_mid_stats['honest_p50']:.4f}/{overall_mid_stats['honest_p75']:.4f}`。",
        f"- `unsafe=1` 样本数 `{unsafe_stats['count']}`，对应 `h_LCB` 均值/标准差 `{unsafe_stats['h_lcb_mean']:.4f}/{unsafe_stats['h_lcb_std']:.4f}`，真实 `committee_honest_ratio` 均值/标准差 `{unsafe_stats['honest_mean']:.4f}/{unsafe_stats['honest_std']:.4f}`。",
        "",
        "## Scene Split",
        "",
        *per_scene_lines,
        "## Judgment",
        "",
        *[f"- {item}" for item in answers],
    ]

    (output_dir / "HLCB_ALIGNMENT_JUDGMENT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
