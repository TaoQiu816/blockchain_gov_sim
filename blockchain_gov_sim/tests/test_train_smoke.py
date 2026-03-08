"""训练与 baseline 轻量 smoke 测试。"""

from __future__ import annotations

from pathlib import Path

from gov_sim.baselines.multirep_topk_static import MultiRepTopKStaticBaseline
from gov_sim.env.gov_env import BlockchainGovEnv
from gov_sim.experiments.train_runner import run_training
from gov_sim.utils.io import load_config


def test_train_smoke(tmp_path: Path) -> None:
    """几百步训练应能完成并保存模型。"""
    config = load_config("configs/default.yaml")
    config["output_root"] = str(tmp_path)
    config["run_name"] = "pytest_smoke"
    config["agent"]["total_timesteps"] = 256
    config["agent"]["n_steps"] = 64
    config["agent"]["batch_size"] = 32
    config["eval"]["episodes"] = 1
    result = run_training(config)
    assert Path(result["output_dir"]).exists()
    assert (Path(result["output_dir"]) / "model.zip").exists()


def test_topk_baseline_switches_committee_method() -> None:
    """Top-K baseline 不能继续沿用主方案的 soft sortition。"""

    config = load_config("configs/default.yaml")
    env = BlockchainGovEnv(config)
    obs, _ = env.reset(seed=42)
    baseline = MultiRepTopKStaticBaseline(config)
    _ = baseline.select_action(env, obs)
    assert env.committee_override_method == "topk"
