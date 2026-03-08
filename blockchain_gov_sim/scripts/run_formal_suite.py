"""第四章正式实验脚本入口。

该脚本专门用于正式 benchmark / 消融矩阵，不替代通用的
`run_benchmarks.py` 和 `run_ablation.py`。原因是正式实验需要：

- 多方法统一汇总；
- 多横轴扫描；
- 固定输出目录；
- 最终 manifest。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gov_sim.experiments.formal_runner import run_formal_suite
from gov_sim.utils.io import load_config
from gov_sim.utils.seed import seed_everything


def main() -> None:
    """解析正式实验命令并执行指定 section。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--benchmark-config", default=None)
    parser.add_argument("--override", action="append", default=[], help="额外覆盖配置，可重复传入多个 YAML。")
    parser.add_argument("--model-path", required=True, help="Ours 最终模型路径。")
    parser.add_argument(
        "--section",
        action="append",
        default=[],
        help="可重复传入；为空时执行 main_compare/malicious_scan/dynamic_attacks/load_shock/high_rtt/high_churn/ablation 全部。",
    )
    args = parser.parse_args()
    config = load_config(args.config, args.benchmark_config, *args.override)
    seed_everything(int(config["seed"]))
    run_formal_suite(config=config, ours_model_path=args.model_path, sections=args.section or None)


if __name__ == "__main__":
    main()
