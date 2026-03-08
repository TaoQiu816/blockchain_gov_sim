"""消融实验脚本入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gov_sim.experiments.ablation_runner import run_ablation
from gov_sim.utils.io import load_config
from gov_sim.utils.seed import seed_everything


def main() -> None:
    """解析命令行并启动 ablation。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-config", default=None)
    parser.add_argument("--override", action="append", default=[], help="额外覆盖配置，可重复传入多个 YAML。")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    config = load_config(args.config, args.train_config, *args.override)
    if args.seed is not None:
        config["seed"] = int(args.seed)
    if args.run_name is not None:
        config["run_name"] = str(args.run_name)
    if args.output_root is not None:
        config["output_root"] = str(args.output_root)
    seed_everything(int(config["seed"]))
    run_ablation(config)


if __name__ == "__main__":
    main()
