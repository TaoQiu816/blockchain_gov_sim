"""生成或执行治理 DQN 的小范围超参数搜索。"""

from __future__ import annotations

import argparse
import itertools
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gov_sim.utils.io import ensure_dir, write_json


SEARCH_SPACE = {
    "lr": [5.0e-5, 1.0e-4, 2.0e-4],
    "target_update_period": [500, 1000, 2000],
    "warmup_steps": [5000, 10000],
    "epsilon_decay_steps": [50000, 80000, 120000],
    "train_freq": [2, 4],
}


def _build_manifest(args: argparse.Namespace) -> list[dict[str, Any]]:
    output_root = ensure_dir(ROOT / args.output_root if not Path(args.output_root).is_absolute() else args.output_root)
    manifest: list[dict[str, Any]] = []
    keys = list(SEARCH_SPACE.keys())
    values = [SEARCH_SPACE[key] for key in keys]
    for idx, combo in enumerate(itertools.product(*values), start=1):
        params = {key: combo[pos] for pos, key in enumerate(keys)}
        run_id = f"cfg{idx:03d}"
        run_root = output_root / run_id
        manifest.append(
            {
                "run_id": run_id,
                "seed": int(args.seed),
                "num_episodes": int(args.num_episodes),
                "episode_length": int(args.episode_length),
                "total_env_steps": int(args.total_env_steps),
                "eval_every": int(args.eval_every),
                "eval_episodes": int(args.eval_episodes),
                "config": str(args.config),
                "train_output_dir": str(run_root / "train"),
                "eval_output_dir": str(run_root / "eval"),
                **params,
            }
        )
    return manifest


def _train_command(entry: dict[str, Any]) -> list[str]:
    return [
        sys.executable,
        "scripts/train_governance_dqn.py",
        "--config",
        str(entry["config"]),
        "--output-dir",
        str(entry["train_output_dir"]),
        "--num-episodes",
        str(entry["num_episodes"]),
        "--episode-length",
        str(entry["episode_length"]),
        "--total-env-steps",
        str(entry["total_env_steps"]),
        "--eval-every",
        str(entry["eval_every"]),
        "--eval-episodes",
        str(entry["eval_episodes"]),
        "--seed",
        str(entry["seed"]),
        "--mixed-scenario",
        "--lr",
        str(entry["lr"]),
        "--target-update-period",
        str(entry["target_update_period"]),
        "--warmup-steps",
        str(entry["warmup_steps"]),
        "--epsilon-decay-steps",
        str(entry["epsilon_decay_steps"]),
        "--train-freq",
        str(entry["train_freq"]),
    ]


def _eval_command(entry: dict[str, Any]) -> list[str]:
    train_dir = Path(str(entry["train_output_dir"]))
    return [
        sys.executable,
        "scripts/eval_governance_dqn.py",
        "--config",
        str(entry["config"]),
        "--checkpoint",
        str(train_dir / "best_checkpoint.pt"),
        "--output-dir",
        str(entry["eval_output_dir"]),
        "--episodes",
        str(entry["eval_episodes"]),
        "--seed",
        str(entry["seed"]),
    ]


def _shared_baseline_command(args: argparse.Namespace, output_root: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "gov_sim.experiments.benchmark_runner",
        "--config",
        str(args.config),
        "--episodes",
        str(args.eval_episodes),
        "--output-dir",
        str(output_root / "shared_baseline"),
        "--methods",
        "Global-Static-Fixed",
        "--by-scenario",
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-root", default="outputs/governance_dqn_search")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-episodes", type=int, default=240)
    parser.add_argument("--episode-length", type=int, default=120)
    parser.add_argument("--total-env-steps", type=int, default=28800)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    output_root = ensure_dir(ROOT / args.output_root if not Path(args.output_root).is_absolute() else args.output_root)
    manifest = _build_manifest(args)
    pd.DataFrame(manifest).to_csv(output_root / "search_space.csv", index=False)
    write_json(
        output_root / "search_manifest.json",
        {
            "runs": manifest,
            "search_space": SEARCH_SPACE,
            "shared_baseline_output_dir": str(output_root / "shared_baseline"),
        },
    )

    command_lines: list[str] = ["set -euo pipefail", f"cd {shlex.quote(str(ROOT))}"]
    baseline_cmd = _shared_baseline_command(args, output_root)
    command_lines.append(shlex.join(baseline_cmd))
    if args.run:
        subprocess.run(baseline_cmd, cwd=ROOT, check=True)
    for entry in manifest:
        train_cmd = _train_command(entry)
        eval_cmd = _eval_command(entry)
        command_lines.append(shlex.join(train_cmd))
        command_lines.append(shlex.join(eval_cmd))
        if args.run:
            subprocess.run(train_cmd, cwd=ROOT, check=True)
            subprocess.run(eval_cmd, cwd=ROOT, check=True)
    (output_root / "run_search_commands.sh").write_text("\n".join(command_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
