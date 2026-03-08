"""AIMD 启发式 baseline。"""

from __future__ import annotations

from typing import Any

from gov_sim.baselines import BaselineBase


class HeuristicAIMDBaseline(BaselineBase):
    """根据队列积压和平均信誉做加性增/乘性减调参。"""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config=config, name="Heuristic-AIMD")
        self.action = tuple(config["baselines"]["static_action"])

    def reset(self) -> None:
        self.action = tuple(self.config["baselines"]["static_action"])

    def select_action(self, env: Any, obs: dict[str, Any]) -> int:
        self._set_committee_method(env, "soft_sortition")
        state = env.unwrapped.get_governance_state()
        queue = float(state["queue_size"])
        snapshot = state["snapshot"]
        aimd = self.config["baselines"]["aimd"]
        m, b, tau, theta = self.action
        # 队列过高时扩大委员会、批量和等待窗口，优先稳住吞吐。
        if queue > float(aimd["queue_high"]):
            m += int(aimd["ai_m"])
            b += int(aimd["ai_b"])
            tau += 20
        # 队列低时保守缩减，降低共识成本与不必要时延。
        elif queue < float(aimd["queue_low"]):
            m = int(round(m * float(aimd["md_m"])))
            b = int(round(b * float(aimd["md_b"])))
            tau -= 20
        if float(snapshot.final_scores.mean()) < 0.6:
            theta += 0.1
        self.action = (m, b, tau, theta)
        return self._nearest_action(m, b, tau, theta)
