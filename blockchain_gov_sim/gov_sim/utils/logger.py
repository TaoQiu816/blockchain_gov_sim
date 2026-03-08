"""轻量 CSV 日志工具。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from gov_sim.utils.io import ensure_dir


class CsvLogger:
    """先缓存再统一落盘的 csv logger。"""

    def __init__(self, output_dir: str | Path, filename: str) -> None:
        self.output_dir = ensure_dir(output_dir)
        self.path = self.output_dir / filename
        self.rows: list[dict[str, Any]] = []

    def log(self, row: dict[str, Any]) -> None:
        """缓存一行结构化记录。"""
        self.rows.append(row)

    def flush(self) -> None:
        """把缓存写出到 csv。"""
        if not self.rows:
            return
        pd.DataFrame(self.rows).to_csv(self.path, index=False)
