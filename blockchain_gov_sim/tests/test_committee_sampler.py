"""委员会采样测试。"""

from __future__ import annotations

import numpy as np

from gov_sim.modules.committee_sampler import CommitteeSampler


def test_committee_sampler_reproducible_and_unique() -> None:
    """相同 seed 下采样应可复现，且必须无放回。"""
    candidates = np.arange(10)
    weights = np.linspace(0.1, 1.0, 10)
    sampler_a = CommitteeSampler(seed=7)
    sampler_b = CommitteeSampler(seed=7)
    sample_a = sampler_a.sample(candidates, weights, 4)
    sample_b = sampler_b.sample(candidates, weights, 4)
    assert np.array_equal(sample_a, sample_b)
    assert len(np.unique(sample_a)) == 4
