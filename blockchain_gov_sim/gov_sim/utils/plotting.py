"""绘图辅助函数。"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


def save_line_plot(
    x: Iterable[float],
    ys: dict[str, Iterable[float]],
    path: str | Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    """保存折线图，并在保存 png 时额外导出 pdf。"""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x_arr = np.asarray(list(x))
    for label, values in ys.items():
        ax.plot(x_arr, list(values), label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    if len(ys) > 1:
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    if str(path).lower().endswith(".png"):
        fig.savefig(str(path)[:-4] + ".pdf")
    plt.close(fig)


def save_bar_plot(labels: list[str], values: list[float], path: str | Path, title: str, ylabel: str) -> None:
    """保存柱状图，并在保存 png 时额外导出 pdf。"""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(labels, values, color="#2f6db2")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    if str(path).lower().endswith(".png"):
        fig.savefig(str(path)[:-4] + ".pdf")
    plt.close(fig)
