"""轻量 smoke test 脚本。"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gov_sim.experiments.train_runner import run_training
from gov_sim.utils.io import load_config
from gov_sim.utils.seed import seed_everything


def main() -> None:
    """用极小配置跑通训练闭环。"""
    config = load_config("configs/default.yaml")
    config["run_name"] = "smoke_test"
    config["agent"]["total_timesteps"] = 512
    config["eval"]["episodes"] = 2
    seed_everything(int(config["seed"]))
    result = run_training(config)
    print(result["summary"])


if __name__ == "__main__":
    main()
