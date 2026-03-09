"""训练结果后处理工具。

该模块的职责是把训练日志 `train_log.csv` 与训练后评估日志 `post_train_eval.csv`
补充成论文中更直观可展示的一组结果图，包括：

1. 训练收敛总览图：reward / cost / unsafe / lambda 四联图；
2. 训练性能图：TPS / latency / mask ratio / constraint violation；
3. 训练后评估轨迹图：latency / TPS / queue / 动作轨迹；
4. 为兼容论文命名习惯，额外导出 `train_metrics.csv` 作为 `train_log.csv` 的别名。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save_panel(fig: plt.Figure, path: Path) -> None:
    """保存面板图，并在 png 保存时同步导出 pdf。"""
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    if str(path).lower().endswith(".png"):
        fig.savefig(str(path)[:-4] + ".pdf")
    plt.close(fig)


def _cleanup_noise(run_dir: Path) -> None:
    """删除不应提交到正式结果目录的噪声文件。"""
    for candidate in [run_dir / ".DS_Store", run_dir / ".ipynb_checkpoints"]:
        if candidate.is_file():
            candidate.unlink()
        elif candidate.is_dir():
            for sub in candidate.rglob("*"):
                if sub.is_file():
                    sub.unlink()
            candidate.rmdir()


def _smooth_series(values: np.ndarray, rolling_window: int) -> tuple[np.ndarray, np.ndarray]:
    """同时计算滚动均值与 EMA。

    设计原因：
    - 滚动均值更适合展示“局部平均性能”；
    - EMA 更适合展示“整体趋势”；
    - 两者一起保留，可以避免只看一种平滑方式而误判收敛。
    """

    series = pd.Series(values.astype(float))
    rolling = series.rolling(window=rolling_window, min_periods=1).mean().to_numpy()
    ema = series.ewm(span=max(rolling_window // 2, 5), adjust=False).mean().to_numpy()
    return rolling, ema


def _plot_with_smoothing(
    axis: plt.Axes,
    x: np.ndarray,
    values: np.ndarray,
    title: str,
    xlabel: str,
    raw_color: str,
    smooth_color: str,
    rolling_window: int,
) -> None:
    """在单个子图上同时绘制原始曲线、滚动均值和 EMA。"""

    rolling, ema = _smooth_series(values, rolling_window)
    axis.plot(x, values, color=raw_color, alpha=0.18, linewidth=1.0, label="raw")
    axis.plot(x, rolling, color=smooth_color, linewidth=2.0, label=f"rolling-{rolling_window}")
    axis.plot(x, ema, color="#111111", alpha=0.75, linewidth=1.5, linestyle="--", label="ema")
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)


def generate_train_artifacts(run_dir: str | Path) -> None:
    """根据已有训练日志补充结果图。

    参数：
    - `run_dir`：一个具体训练种子目录，例如 `outputs/formal/train/moderate_seed42`
    """

    directory = Path(run_dir)
    if not directory.exists():
        raise FileNotFoundError(f"run_dir not found: {directory}")
    _cleanup_noise(directory)

    train_log_path = directory / "train_log.csv"
    eval_log_path = directory / "post_train_eval.csv"
    if not train_log_path.exists():
        raise FileNotFoundError(f"train_log.csv not found under {directory}")

    train_df = pd.read_csv(train_log_path)
    # 兼容论文中的 `train_metrics.csv` 命名。
    train_df.to_csv(directory / "train_metrics.csv", index=False)

    if not train_df.empty:
        x = np.arange(len(train_df))
        rolling_window = int(min(max(len(train_df) // 25, 50), 200))
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        series = [
            ("episode_reward", "Reward"),
            ("episode_cost", "Cost"),
            ("unsafe_rate", "Unsafe Rate"),
            ("lagrangian_lambda", "Lambda"),
        ]
        for axis, (column, title) in zip(axes, series):
            if column in train_df.columns:
                _plot_with_smoothing(
                    axis,
                    x,
                    train_df[column].astype(float).to_numpy(),
                    title,
                    "Episode",
                    raw_color="#6f93c5",
                    smooth_color="#2f6db2",
                    rolling_window=rolling_window,
                )
        _save_panel(fig, directory / "convergence_panel.png")

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        extra_series = [
            ("tps", "Training TPS"),
            ("latency", "Training Latency"),
            ("mask_ratio", "Mask Ratio"),
            ("constraint_violation", "Constraint Violation"),
        ]
        for axis, (column, title) in zip(axes, extra_series):
            if column in train_df.columns:
                _plot_with_smoothing(
                    axis,
                    x,
                    train_df[column].astype(float).to_numpy(),
                    title,
                    "Episode",
                    raw_color="#e7aa8c",
                    smooth_color="#c95f2d",
                    rolling_window=rolling_window,
                )
        _save_panel(fig, directory / "training_diagnostics_panel.png")

        # 单张平滑曲线：方便论文正文单图展示。
        single_series = [
            ("episode_reward", "Reward", "reward_curve_smoothed.png", "Reward"),
            ("episode_cost", "Cost", "cost_curve_smoothed.png", "Cost"),
            ("unsafe_rate", "Unsafe Rate", "unsafe_curve_smoothed.png", "Unsafe rate"),
            ("lagrangian_lambda", "Lambda", "lambda_curve_smoothed.png", "Lambda"),
        ]
        for column, title, filename, ylabel in single_series:
            if column not in train_df.columns:
                continue
            fig, ax = plt.subplots(figsize=(9, 4.8))
            _plot_with_smoothing(
                ax,
                x,
                train_df[column].astype(float).to_numpy(),
                title,
                "Episode",
                raw_color="#8aa1bf",
                smooth_color="#244a87",
                rolling_window=rolling_window,
            )
            ax.set_ylabel(ylabel)
            _save_panel(fig, directory / filename)

    if eval_log_path.exists():
        eval_df = pd.read_csv(eval_log_path)
        if not eval_df.empty:
            x = np.arange(len(eval_df))
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            axes = axes.flatten()
            eval_series = [
                ("L_bar_e", "Post-train Latency"),
                ("tps", "Post-train TPS"),
                ("Q_e", "Queue Length"),
                ("eligible_size", "Eligible Size"),
            ]
            for axis, (column, title) in zip(axes, eval_series):
                if column in eval_df.columns:
                    axis.plot(x, eval_df[column].astype(float).to_numpy(), color="#3b8f44")
                axis.set_title(title)
                axis.set_xlabel("Step")
                axis.grid(alpha=0.25)
            _save_panel(fig, directory / "post_train_eval_panel.png")

            fig, ax = plt.subplots(figsize=(10, 4.8))
            if "m_e" in eval_df.columns:
                ax.plot(x, eval_df["m_e"].astype(float).to_numpy(), label="m")
            if "b_e" in eval_df.columns:
                ax.plot(x, eval_df["b_e"].astype(float).to_numpy() / 32.0, label="b/32")
            if "tau_e" in eval_df.columns:
                ax.plot(x, eval_df["tau_e"].astype(float).to_numpy(), label="tau")
            if "theta_e" in eval_df.columns:
                ax.plot(x, eval_df["theta_e"].astype(float).to_numpy() * 100.0, label="theta*100")
            ax.set_title("Post-train Action Trajectory")
            ax.set_xlabel("Step")
            ax.grid(alpha=0.25)
            ax.legend()
            _save_panel(fig, directory / "post_train_action_trajectory.png")
