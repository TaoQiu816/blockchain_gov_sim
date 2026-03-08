"""固定参数 baseline。"""

from __future__ import annotations

from typing import Any

import numpy as np

from gov_sim.baselines import BaselineBase


class StaticParamBaseline(BaselineBase):
    """始终输出同一组治理参数，用作最基础参考线。"""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config=config, name="Static-Param")

    def select_action(self, env: Any, obs: dict[str, np.ndarray]) -> int:
        self._set_committee_method(env, "soft_sortition")
        m, b, tau, theta = self.config["baselines"]["static_action"]
        return self._nearest_action(int(m), int(b), int(tau), float(theta))
