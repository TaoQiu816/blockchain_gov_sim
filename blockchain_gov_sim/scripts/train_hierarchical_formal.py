"""正式多 seed 分层实验启动脚本。"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gov_sim.utils.io import deep_update, load_config
from gov_sim.utils.seed import seed_everything

FORMAL_VARIANTS = ("final", "no_dynamic_theta", "single_dim_trust", "no_context_fusion")


def _patch_no_dynamic_theta() -> None:
    spec = importlib.import_module("gov_sim.hierarchical.spec")
    spec.HIGH_LEVEL_THETA_VALUES = (0.50,)
    spec.HIGH_LEVEL_TEMPLATES = ((7, 0.50), (9, 0.50))
    spec.HIGH_LEVEL_ENCODING_TEMPLATES = ((5, 0.50), (7, 0.50), (9, 0.50))
    spec.EXECUTABLE_HIGH_LEVEL_TEMPLATES = spec.HIGH_LEVEL_TEMPLATES + (spec.BACKSTOP_HIGH_TEMPLATE,)
    spec.HIGH_LEVEL_DIM = len(spec.HIGH_LEVEL_TEMPLATES)


def _apply_variant_overrides(config: dict, variant: str) -> dict:
    if variant == "final":
        return config
    if variant == "single_dim_trust":
        return deep_update(config, {"reputation": {"fusion_dims": ["svc"], "use_penalties": False}})
    if variant == "no_context_fusion":
        return deep_update(config, {"reputation": {"use_context_gate": False}})
    if variant == "no_dynamic_theta":
        _patch_no_dynamic_theta()
        return config
    raise ValueError(f"Unsupported variant: {variant}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-config", default=None)
    parser.add_argument("--override", action="append", default=[], help="额外覆盖配置，可重复传入多个 YAML。")
    parser.add_argument("--variant", choices=FORMAL_VARIANTS, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()

    config = load_config(args.config, args.train_config, *args.override)
    config = _apply_variant_overrides(config, args.variant)
    config.setdefault("formal_experiment", {})
    config["formal_experiment"]["variant"] = str(args.variant)
    if args.seed is not None:
        config["seed"] = int(args.seed)
    if args.run_name is not None:
        config["run_name"] = str(args.run_name)
    if args.output_root is not None:
        config["output_root"] = str(args.output_root)

    seed_everything(int(config["seed"]))
    from gov_sim.hierarchical.trainer import run_hierarchical_training

    run_hierarchical_training(config)


if __name__ == "__main__":
    main()
