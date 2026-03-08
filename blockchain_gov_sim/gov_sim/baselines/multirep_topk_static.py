"""多维信誉 Top-K 静态 baseline。"""

from __future__ import annotations

from typing import Any

from gov_sim.baselines import BaselineBase, quantile_threshold


class MultiRepTopKStaticBaseline(BaselineBase):
    """使用多维融合信誉，但委员会采用静态 Top-K 而非软抽签。"""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config=config, name="MultiRep-TopK-Static")

    def select_action(self, env: Any, obs: dict[str, Any]) -> int:
        self._set_committee_method(env, "topk")
        state = env.unwrapped.get_governance_state()
        theta = quantile_threshold(state["snapshot"].final_scores, float(self.config["baselines"]["topk_quantile"]))
        return self._nearest_action(11, 256, 40, theta)
