"""Benchmark runner。

该文件实现第四章要求的四类实验：
- 恶意节点渗透
- on-off / zigzag / collusion 攻击
- 阶跃负载
- 高 RTT / 高 churn

除汇总 csv/json 外，也尽量保留代表性场景的 step 级日志与图表。
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from gov_sim.experiments import (
    apply_override,
    attack_override,
    evaluate_controller,
    instantiate_baseline,
    malicious_ratio_override,
    prepare_output_dir,
)
from gov_sim.experiments.eval_runner import load_controller
from gov_sim.utils.io import write_json
from gov_sim.utils.plotting import save_bar_plot, save_line_plot


def _run_suite(controller: Any, is_baseline: bool, config: dict[str, Any], override: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """在覆盖配置下运行一次实验子场景。"""
    experiment_cfg = apply_override(config, override)
    dataframe, summary = evaluate_controller(
        controller=controller,
        config=experiment_cfg,
        episodes=int(experiment_cfg["eval"]["episodes"]),
        deterministic=bool(experiment_cfg["eval"]["deterministic"]),
        is_baseline=is_baseline,
    )
    return dataframe, summary


def run_benchmarks(config: dict[str, Any], model_path: str | None = None) -> dict[str, Any]:
    """执行全部 benchmark，并导出汇总表与代表性曲线。"""
    output_dir = prepare_output_dir(config, "benchmark")
    controller, is_baseline = load_controller(model_path=model_path, config=config, baseline_name=None) if model_path else (
        instantiate_baseline("Static-Param", config),
        True,
    )
    results: list[dict[str, Any]] = []
    representative_logs: dict[str, pd.DataFrame] = {}

    for ratio in [0.0, 0.1, 0.2, 0.3, 0.4]:
        dataframe, summary = _run_suite(controller, is_baseline, config, malicious_ratio_override(ratio))
        summary["experiment"] = "malicious_penetration"
        summary["malicious_ratio"] = ratio
        results.append(summary)
        if abs(ratio - 0.2) < 1.0e-8:
            representative_logs["malicious_penetration"] = dataframe

    for period in [4, 8, 12]:
        dataframe, summary = _run_suite(controller, is_baseline, config, attack_override(on_off_period=period))
        summary["experiment"] = "onoff"
        summary["on_off_period"] = period
        results.append(summary)
        if period == 8:
            representative_logs["onoff"] = dataframe
    for freq in [0.08, 0.18, 0.32]:
        dataframe, summary = _run_suite(controller, is_baseline, config, attack_override(zigzag_freq=freq))
        summary["experiment"] = "zigzag"
        summary["zigzag_freq"] = freq
        results.append(summary)
        if abs(freq - 0.18) < 1.0e-8:
            representative_logs["zigzag"] = dataframe
    for size in [2, 3, 5]:
        dataframe, summary = _run_suite(controller, is_baseline, config, attack_override(collusion_group_size=size))
        summary["experiment"] = "collusion"
        summary["collusion_group_size"] = size
        results.append(summary)
        if size == 3:
            representative_logs["collusion"] = dataframe

    for e0, k in [(30, 1.5), (40, 2.0), (50, 2.5)]:
        override = {"scenario": {"default_name": "step", "step": {"type": "step", "lambda": 150, "e0": e0, "k": k}}}
        dataframe, summary = _run_suite(controller, is_baseline, config, override)
        summary["experiment"] = "step_load"
        summary["e0"] = e0
        summary["k"] = k
        results.append(summary)
        if e0 == 40 and abs(k - 2.0) < 1.0e-8:
            representative_logs["step_load"] = dataframe

    for rtt_max, p_off, p_on in [(55.0, 0.06, 0.30), (70.0, 0.10, 0.25), (90.0, 0.14, 0.18)]:
        override = {"scenario": {"network": {"rtt_max": rtt_max, "p_off": p_off, "p_on": p_on}}}
        dataframe, summary = _run_suite(controller, is_baseline, config, override)
        summary["experiment"] = "high_rtt_churn"
        summary["RTT_max"] = rtt_max
        summary["p_off"] = p_off
        summary["p_on"] = p_on
        results.append(summary)
        if abs(rtt_max - 70.0) < 1.0e-8:
            representative_logs["high_rtt_churn"] = dataframe

    df = pd.DataFrame(results)
    df.to_csv(output_dir / "benchmark_log.csv", index=False)
    write_json(output_dir / "benchmark_summary.json", {"rows": results})
    for name, dataframe in representative_logs.items():
        dataframe.to_csv(output_dir / f"{name}_steps.csv", index=False)

    mal_df = df[df["experiment"] == "malicious_penetration"]
    if not mal_df.empty:
        save_line_plot(mal_df["malicious_ratio"], {"TPS": mal_df["tps"]}, output_dir / "tps_vs_malicious_ratio.png", "TPS vs Malicious Ratio", "Malicious ratio", "TPS")
        save_line_plot(
            mal_df["malicious_ratio"],
            {"Latency": mal_df["mean_latency"]},
            output_dir / "latency_vs_malicious_ratio.png",
            "Latency vs Malicious Ratio",
            "Malicious ratio",
            "Latency (ms)",
        )
    step_df = df[df["experiment"] == "step_load"]
    if not step_df.empty:
        save_line_plot(step_df["k"], {"queue_peak": step_df["queue_peak"], "recovery_time": step_df["recovery_time"]}, output_dir / "queue_recovery.png", "Queue and Recovery", "Step multiplier", "Value")
    if "step_load" in representative_logs and not representative_logs["step_load"].empty:
        step_log = representative_logs["step_load"]
        save_line_plot(
            step_log.index,
            {
                "m": step_log["m_e"],
                "b/32": step_log["b_e"] / 32.0,
                "tau": step_log["tau_e"],
                "theta*100": step_log["theta_e"] * 100.0,
            },
            output_dir / "action_trajectory.png",
            "Action Trajectory Under Step Load",
            "Step",
            "Scaled value",
        )
    high_df = df[df["experiment"] == "high_rtt_churn"]
    if not high_df.empty:
        save_bar_plot(
            [f"{r:.0f}" for r in high_df["RTT_max"]],
            high_df["unsafe_rate"].tolist(),
            output_dir / "unsafe_vs_high_rtt.png",
            "Unsafe Rate Under High RTT/Churn",
            "Unsafe rate",
        )
    if "high_rtt_churn" in representative_logs and not representative_logs["high_rtt_churn"].empty:
        eligible_log = representative_logs["high_rtt_churn"]["eligible_size"].value_counts().sort_index()
        save_bar_plot(
            [str(idx) for idx in eligible_log.index.tolist()],
            eligible_log.values.astype(float).tolist(),
            output_dir / "eligible_size_distribution.png",
            "Eligible Size Distribution",
            "Count",
        )
    return {"output_dir": str(output_dir), "rows": results}
