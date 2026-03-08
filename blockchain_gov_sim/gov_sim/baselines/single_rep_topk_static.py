"""单维信誉 Top-K 静态 baseline。"""

from __future__ import annotations

from typing import Any

from gov_sim.baselines import BaselineBase, quantile_threshold


class SingleRepTopKStaticBaseline(BaselineBase):
    """仅使用 `svc` 单维信誉，并用 Top-K 委员会替代软抽签。"""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config=config, name="SingleRep-TopK-Static")

    def select_action(self, env: Any, obs: dict[str, Any]) -> int:
        self._set_committee_method(env, "topk")
        state = env.unwrapped.get_governance_state()
        svc_scores = state["snapshot"].mu["svc"]
        theta = quantile_threshold(svc_scores, float(self.config["baselines"]["topk_quantile"]))
        return self._nearest_action(11, 256, 40, theta)
