"""第四章正式实验编排 runner。

该文件不改动核心建模与算法，只负责把已经审计通过的训练/评估组件
组织成论文第四章需要的正式实验矩阵，包括：

1. 默认工作点总体对比；
2. 恶意节点比例扫描；
3. 动态攻击鲁棒性；
4. 阶跃负载抗冲击；
5. 高 RTT / 高 churn；
6. 正式消融。

设计原则：
- 单控制器评估仍复用 `evaluate_controller()`；
- 输出目录按实验组独立，避免覆盖；
- 同时导出 csv/json/png/pdf，便于论文制表和附录审计；
- Ours 与 baseline 使用统一环境与统一评估协议，保证公平。
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re
from typing import Any

import pandas as pd

from gov_sim.agent.callbacks import TrainLoggingCallback
from gov_sim.experiments import (
    apply_override,
    build_model,
    build_plain_ppo,
    evaluate_controller,
    make_vec_env,
)
from gov_sim.experiments.eval_runner import load_controller
from gov_sim.utils.io import ensure_dir, write_json
from gov_sim.utils.plotting import save_bar_plot, save_line_plot


def _slug(text: str) -> str:
    """把方法名或实验名转成稳定目录名。"""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _stage_dir(config: dict[str, Any], stage: str, substage: str | None = None) -> Path:
    """生成正式实验的固定目录。

    与通用 runner 使用 `outputs/<stage>/<run_name>` 不同，
    正式实验强调“实验组”而不是单次 run，因此目录直接组织为：
    `outputs/formal/<stage>/<substage>`
    """

    path = Path(config["output_root"]) / stage
    if substage is not None:
        path = path / substage
    return ensure_dir(path)


def _load_method(method: str, config: dict[str, Any], ours_model_path: str) -> tuple[Any, bool]:
    """按方法名加载 Ours 或 baseline。"""
    if method == "Ours":
        return load_controller(model_path=ours_model_path, config=config, baseline_name=None)
    return load_controller(model_path=None, config=config, baseline_name=method)


def _evaluate_method(
    method: str,
    base_config: dict[str, Any],
    ours_model_path: str,
    override: dict[str, Any],
    output_dir: Path,
    tag: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """执行单个方法在单个场景下的评估，并落盘 step 日志与 summary。"""
    config = apply_override(base_config, override)
    controller, is_baseline = _load_method(method, config, ours_model_path)
    dataframe, summary = evaluate_controller(
        controller=controller,
        config=config,
        episodes=int(config["eval"]["episodes"]),
        deterministic=bool(config["eval"]["deterministic"]),
        is_baseline=is_baseline,
    )
    method_dir = ensure_dir(output_dir / _slug(method))
    dataframe.to_csv(method_dir / f"{tag}_steps.csv", index=False)
    write_json(method_dir / f"{tag}_summary.json", summary)
    return dataframe, summary


def _plot_metric_bars(df: pd.DataFrame, metric_labels: dict[str, str], output_dir: Path, prefix: str) -> None:
    """对总体对比或消融结果按指标出柱状图。"""
    for metric, ylabel in metric_labels.items():
        save_bar_plot(
            df["method"].tolist(),
            df[metric].astype(float).tolist(),
            output_dir / f"{prefix}_{metric}.png",
            f"{prefix.replace('_', ' ').title()} - {metric}",
            ylabel,
        )


def _plot_scan_lines(df: pd.DataFrame, x_col: str, metrics: dict[str, str], output_dir: Path, prefix: str) -> None:
    """把扫描实验画成多方法折线图。"""
    methods = list(dict.fromkeys(df["method"].tolist()))
    x_values = sorted(df[x_col].unique().tolist())
    for metric, ylabel in metrics.items():
        series: dict[str, list[float]] = {}
        for method in methods:
            sub_df = df[df["method"] == method].sort_values(x_col)
            series[method] = sub_df[metric].astype(float).tolist()
        save_line_plot(
            x_values,
            series,
            output_dir / f"{prefix}_{metric}.png",
            f"{prefix.replace('_', ' ').title()} - {metric}",
            x_col,
            ylabel,
        )


def run_main_compare(config: dict[str, Any], ours_model_path: str) -> dict[str, Any]:
    """实验 1：默认 hard 工作点总体对比。"""
    stage_dir = _stage_dir(config, "main_compare")
    methods = config["formal"]["main_compare"]["methods"]
    rows: list[dict[str, Any]] = []
    for method in methods:
        _, summary = _evaluate_method(method, config, ours_model_path, {}, stage_dir, "hard_default")
        summary["method"] = method
        rows.append(summary)
    df = pd.DataFrame(rows)
    df.to_csv(stage_dir / "main_compare.csv", index=False)
    write_json(stage_dir / "main_compare.json", {"rows": rows})
    _plot_metric_bars(
        df,
        {
            "R_unsafe": "Unsafe rate",
            "R_pollute": "Pollute rate",
            "TPS": "TPS",
            "mean_latency": "Latency (ms)",
            "queue_peak": "Queue peak",
            "eligible_size_mean": "Eligible size mean",
        },
        stage_dir,
        "main_compare",
    )
    return {"output_dir": str(stage_dir), "rows": rows}


def run_malicious_scan(config: dict[str, Any], ours_model_path: str) -> dict[str, Any]:
    """实验 2：恶意节点比例扫描。"""
    stage_dir = _stage_dir(config, "malicious_scan")
    plan = config["formal"]["malicious_scan"]
    rows: list[dict[str, Any]] = []
    for ratio in plan["ratios"]:
        override = {"env": {"malicious_ratio": float(ratio)}}
        tag = f"pm_{str(ratio).replace('.', 'p')}"
        for method in plan["methods"]:
            _, summary = _evaluate_method(method, config, ours_model_path, override, stage_dir, tag)
            summary["method"] = method
            summary["p_m"] = float(ratio)
            rows.append(summary)
    df = pd.DataFrame(rows)
    df.to_csv(stage_dir / "malicious_scan.csv", index=False)
    write_json(stage_dir / "malicious_scan.json", {"rows": rows})
    _plot_scan_lines(
        df,
        "p_m",
        {
            "unsafe_rate": "Unsafe rate",
            "pollute_rate": "Pollute rate",
            "TPS": "TPS",
            "mean_latency": "Latency (ms)",
        },
        stage_dir,
        "pm_scan",
    )
    return {"output_dir": str(stage_dir), "rows": rows}


def run_dynamic_attacks(config: dict[str, Any], ours_model_path: str) -> dict[str, Any]:
    """实验 3：on-off / zigzag / collusion 动态攻击鲁棒性。"""
    root_dir = _stage_dir(config, "dynamic_attacks")
    plan = config["formal"]["dynamic_attacks"]
    outputs: dict[str, Any] = {}
    subplans = {
        "onoff": ("level", plan["on_off_periods"], lambda value: {"scenario": {"attack": {"on_off_period": int(value)}}}),
        "zigzag": ("level", plan["zigzag_freqs"], lambda value: {"scenario": {"attack": {"zigzag_freq": float(value)}}}),
        "collusion": ("level", plan["collusion_group_sizes"], lambda value: {"scenario": {"attack": {"collusion_group_size": int(value)}}}),
    }
    for subname, (x_col, levels, build_override) in subplans.items():
        sub_dir = ensure_dir(root_dir / subname)
        rows: list[dict[str, Any]] = []
        for level in levels:
            tag = f"{subname}_{str(level).replace('.', 'p')}"
            for method in plan["methods"]:
                _, summary = _evaluate_method(method, config, ours_model_path, build_override(level), sub_dir, tag)
                summary["method"] = method
                summary[x_col] = float(level)
                rows.append(summary)
        df = pd.DataFrame(rows)
        df.to_csv(sub_dir / f"{subname}.csv", index=False)
        write_json(sub_dir / f"{subname}.json", {"rows": rows})
        _plot_scan_lines(
            df,
            x_col,
            {
                "malicious_detection_f1": "F1",
                "false_positive_rate": "FPR",
                "pollute_rate": "Pollute rate",
                "unsafe_rate": "Unsafe rate",
            },
            sub_dir,
            subname,
        )
        outputs[subname] = {"output_dir": str(sub_dir), "rows": rows}
    return outputs


def run_load_shock(config: dict[str, Any], ours_model_path: str) -> dict[str, Any]:
    """实验 4：阶跃负载抗冲击。"""
    stage_dir = _stage_dir(config, "load_shock")
    plan = config["formal"]["load_shock"]
    lambda1 = int(plan["lambda1"])
    e0 = int(plan["e0"])
    rows: list[dict[str, Any]] = []
    representative: dict[str, pd.DataFrame] = {}
    for lambda2 in plan["lambda2_values"]:
        override = {
            "scenario": {
                "default_name": "step",
                "step": {"type": "step", "lambda": lambda1, "e0": e0, "k": float(lambda2) / float(lambda1)},
            }
        }
        tag = f"lambda2_{lambda2}"
        for method in plan["methods"]:
            dataframe, summary = _evaluate_method(method, config, ours_model_path, override, stage_dir, tag)
            summary["method"] = method
            summary["lambda1"] = lambda1
            summary["lambda2"] = int(lambda2)
            rows.append(summary)
            if int(lambda2) == max(plan["lambda2_values"]):
                representative[method] = dataframe
    df = pd.DataFrame(rows)
    df.to_csv(stage_dir / "load_shock.csv", index=False)
    write_json(stage_dir / "load_shock.json", {"rows": rows})
    _plot_scan_lines(
        df,
        "lambda2",
        {
            "queue_peak": "Queue peak",
            "recovery_time": "Recovery time",
            "TPS": "TPS",
            "mean_latency": "Latency (ms)",
            "oscillation_index": "Oscillation index",
        },
        stage_dir,
        "load_shock",
    )
    if representative:
        queue_series: dict[str, list[float]] = {}
        tps_series: dict[str, list[float]] = {}
        action_series: dict[str, list[float]] = {}
        x_axis: list[int] = []
        for method, dataframe in representative.items():
            grouped = dataframe.groupby("step").mean(numeric_only=True).reset_index()
            if not x_axis:
                x_axis = grouped["step"].astype(int).tolist()
            queue_series[method] = grouped["Q_e"].astype(float).tolist()
            tps_series[method] = grouped["tps"].astype(float).tolist()
            action_series[method] = grouped["m_e"].astype(float).tolist()
        save_line_plot(x_axis, queue_series, stage_dir / "queue_trajectory.png", "Queue Trajectory", "Step", "Queue")
        save_line_plot(x_axis, tps_series, stage_dir / "tps_trajectory.png", "TPS Trajectory", "Step", "TPS")
        save_line_plot(x_axis, action_series, stage_dir / "action_trajectory.png", "Action Trajectory (m)", "Step", "Committee size")
    return {"output_dir": str(stage_dir), "rows": rows}


def run_high_rtt(config: dict[str, Any], ours_model_path: str) -> dict[str, Any]:
    """实验 5.1：高 RTT 鲁棒性。"""
    stage_dir = _stage_dir(config, "high_rtt")
    plan = config["formal"]["high_rtt"]
    rows: list[dict[str, Any]] = []
    for rtt_max in plan["rtt_max_values"]:
        override = {"scenario": {"network": {"rtt_max": float(rtt_max)}}}
        tag = f"rtt_{int(rtt_max)}"
        for method in plan["methods"]:
            _, summary = _evaluate_method(method, config, ours_model_path, override, stage_dir, tag)
            summary["method"] = method
            summary["RTT_max"] = float(rtt_max)
            rows.append(summary)
    df = pd.DataFrame(rows)
    df.to_csv(stage_dir / "high_rtt.csv", index=False)
    write_json(stage_dir / "high_rtt.json", {"rows": rows})
    _plot_scan_lines(
        df,
        "RTT_max",
        {
            "TPS": "TPS",
            "mean_latency": "Latency (ms)",
            "unsafe_rate": "Unsafe rate",
            "timeout_failure_rate": "Timeout failure rate",
            "eligible_size_mean": "Eligible size mean",
        },
        stage_dir,
        "high_rtt",
    )
    return {"output_dir": str(stage_dir), "rows": rows}


def run_high_churn(config: dict[str, Any], ours_model_path: str) -> dict[str, Any]:
    """实验 5.2：高 churn 鲁棒性。"""
    stage_dir = _stage_dir(config, "high_churn")
    plan = config["formal"]["high_churn"]
    rows: list[dict[str, Any]] = []
    for level in plan["levels"]:
        override = {"scenario": {"network": {"p_off": float(level["p_off"]), "p_on": float(level["p_on"])}}}
        tag = str(level["name"])
        for method in plan["methods"]:
            _, summary = _evaluate_method(method, config, ours_model_path, override, stage_dir, tag)
            summary["method"] = method
            summary["level"] = str(level["name"])
            rows.append(summary)
    df = pd.DataFrame(rows)
    df.to_csv(stage_dir / "high_churn.csv", index=False)
    write_json(stage_dir / "high_churn.json", {"rows": rows})
    methods = list(dict.fromkeys(df["method"].tolist()))
    x_values = [str(level["name"]) for level in plan["levels"]]
    for metric, ylabel in {
        "TPS": "TPS",
        "mean_latency": "Latency (ms)",
        "unsafe_rate": "Unsafe rate",
        "timeout_failure_rate": "Timeout failure rate",
        "eligible_size_mean": "Eligible size mean",
    }.items():
        series: dict[str, list[float]] = {}
        for method in methods:
            sub_df = df[df["method"] == method].set_index("level").reindex(x_values)
            series[method] = sub_df[metric].astype(float).tolist()
        save_line_plot(x_values, series, stage_dir / f"high_churn_{metric}.png", f"High Churn - {metric}", "Level", ylabel)
    return {"output_dir": str(stage_dir), "rows": rows}


def run_ablation_formal(config: dict[str, Any]) -> dict[str, Any]:
    """实验 6：正式消融。

    正式消融与通用 `ablation_runner` 的区别在于：
    - 使用更接近正式训练的 timesteps；
    - 按 seed 划分独立输出目录；
    - 对每个变体聚合均值，便于直接生成论文图表。
    """

    stage_dir = _stage_dir(config, "ablation")
    plan = config["formal"]["ablation"]
    rows: list[dict[str, Any]] = []
    for variant in plan["variants"]:
        variant_name = str(variant["name"])
        variant_dir = ensure_dir(stage_dir / _slug(variant_name))
        seed_rows: list[dict[str, Any]] = []
        for seed in plan["seeds"]:
            experiment_cfg = apply_override(config, variant.get("override", {}))
            experiment_cfg["seed"] = int(seed)
            env = make_vec_env(experiment_cfg)
            callback = TrainLoggingCallback(log_path=variant_dir / f"seed_{seed}_train_log.csv")
            if bool(variant.get("plain_ppo", False)):
                controller = build_plain_ppo(experiment_cfg, env)
            else:
                controller = build_model(experiment_cfg, env, use_lagrangian=True)
            controller.learn(total_timesteps=int(plan["timesteps"]), callback=callback, progress_bar=False)
            controller.save(str(variant_dir / f"seed_{seed}_model"))
            eval_df, summary = evaluate_controller(
                controller=controller,
                config=experiment_cfg,
                episodes=int(experiment_cfg["eval"]["episodes"]),
                deterministic=True,
                is_baseline=False,
            )
            eval_df.to_csv(variant_dir / f"seed_{seed}_eval.csv", index=False)
            summary["method"] = variant_name
            summary["seed"] = int(seed)
            seed_rows.append(summary)
            rows.append(summary)
        seed_df = pd.DataFrame(seed_rows)
        seed_df.to_csv(variant_dir / "seed_summary.csv", index=False)
        write_json(variant_dir / "seed_summary.json", {"rows": seed_rows})
    df = pd.DataFrame(rows)
    agg = df.groupby("method", as_index=False).mean(numeric_only=True)
    agg.to_csv(stage_dir / "ablation_summary.csv", index=False)
    write_json(stage_dir / "ablation_summary.json", {"rows": agg.to_dict(orient="records")})
    _plot_metric_bars(
        agg,
        {
            "unsafe_rate": "Unsafe rate",
            "pollute_rate": "Pollute rate",
            "TPS": "TPS",
            "mean_latency": "Latency (ms)",
        },
        stage_dir,
        "ablation",
    )
    return {"output_dir": str(stage_dir), "rows": agg.to_dict(orient="records")}


def run_formal_suite(config: dict[str, Any], ours_model_path: str, sections: list[str] | None = None) -> dict[str, Any]:
    """执行正式 benchmark / ablation 全套实验。"""
    selected = set(sections or ["main_compare", "malicious_scan", "dynamic_attacks", "load_shock", "high_rtt", "high_churn", "ablation"])
    results: dict[str, Any] = {}
    if "main_compare" in selected:
        results["main_compare"] = run_main_compare(config, ours_model_path)
    if "malicious_scan" in selected:
        results["malicious_scan"] = run_malicious_scan(config, ours_model_path)
    if "dynamic_attacks" in selected:
        results["dynamic_attacks"] = run_dynamic_attacks(config, ours_model_path)
    if "load_shock" in selected:
        results["load_shock"] = run_load_shock(config, ours_model_path)
    if "high_rtt" in selected:
        results["high_rtt"] = run_high_rtt(config, ours_model_path)
    if "high_churn" in selected:
        results["high_churn"] = run_high_churn(config, ours_model_path)
    if "ablation" in selected:
        results["ablation"] = run_ablation_formal(deepcopy(config))
    write_json(Path(config["output_root"]) / "manifest.json", results)
    return results
