"""环境 step/reset 测试。"""

from __future__ import annotations

import numpy as np

from gov_sim.env.gov_env import BlockchainGovEnv
from gov_sim.utils.io import load_config


def test_env_step_random_action_runs() -> None:
    """随机动作下环境不应崩溃，且必须返回 cost 与 action_mask。"""
    config = load_config("configs/default.yaml")
    env = BlockchainGovEnv(config)
    obs, _ = env.reset(seed=42)
    for _ in range(5):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert "action_mask" in obs
        assert "cost" in info
        if terminated or truncated:
            break


def test_env_reset_seed_is_reproducible() -> None:
    """相同 seed 下首个观测应可复现。"""

    config = load_config("configs/default.yaml")
    env = BlockchainGovEnv(config)
    obs_a, _ = env.reset(seed=123)
    obs_b, _ = env.reset(seed=123)
    assert np.allclose(obs_a["state"], obs_b["state"])
    assert (obs_a["action_mask"] == obs_b["action_mask"]).all()
