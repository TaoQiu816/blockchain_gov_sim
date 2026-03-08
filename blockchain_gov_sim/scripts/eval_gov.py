"""评估脚本入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gov_sim.experiments.eval_runner import run_evaluation
from gov_sim.utils.io import load_config
from gov_sim.utils.seed import seed_everything


def main() -> None:
    """解析命令行并启动评估。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--eval-config", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--baseline", default=None)
    args = parser.parse_args()
    config = load_config(args.config, args.eval_config)
    seed_everything(int(config["seed"]))
    run_evaluation(config=config, model_path=args.model_path, baseline_name=args.baseline)


if __name__ == "__main__":
    main()
