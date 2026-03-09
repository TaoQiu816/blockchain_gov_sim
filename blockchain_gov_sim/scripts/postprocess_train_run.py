"""对已完成的训练目录补生成论文结果图。

典型用途：
- 远程服务器已经跑完训练，但训练时尚未生成综合收敛图；
- 需要把旧训练结果补齐成统一的论文输出格式。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gov_sim.utils.train_artifacts import generate_train_artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="具体训练结果目录，如 outputs/formal/train/moderate_seed42")
    args = parser.parse_args()
    generate_train_artifacts(args.run_dir)


if __name__ == "__main__":
    main()
