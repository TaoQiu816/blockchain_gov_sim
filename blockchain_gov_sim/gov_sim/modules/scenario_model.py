"""场景生成模块。

该文件负责生成第四章仿真中的“外部环境扰动”：

1. 负载到达过程 `A_e`；
2. RSU 在线/离线马尔可夫链；
3. 网络 RTT 与 churn；
4. 在给定上下文下诚实节点四维基础成功概率。

这里刻意不依赖第三章任何 DAG / 卸载逻辑，保证第四章链侧治理实验独立可复现。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from gov_sim.constants import REPUTATION_DIMS
from gov_sim.utils.math_utils import clip01


@dataclass
class ScenarioStep:
    """单个治理周期的场景快照。

    该对象是环境在每一步开始时获得的“外部世界状态”，后续证据生成、
    信誉更新与链侧性能计算都以此为输入。
    """

    epoch: int
    arrivals: int
    load: float
    rtt: float
    churn: float
    online: np.ndarray
    malicious: np.ndarray
    uptime: np.ndarray
    base_probs: dict[str, np.ndarray]


class ScenarioModel:
    """联盟链治理场景生成器。

    职责：
    - 生成平稳 / 阶跃 / MMPP 到达量；
    - 为每个 RSU 维护在线/离线马尔可夫链；
    - 维护节点稳定性 `uptime`；
    - 在当前负载、RTT、churn 下生成诚实节点四维基础成功概率。

    注意：
    - `malicious` 的抽样在 reset 时固定，以便同一 episode 内攻击者集合稳定；
    - reset 支持 seed 覆盖，保证评估多 episode 时既可复现，又不是同一条轨迹反复拷贝。
    """

    def __init__(self, config: dict[str, Any], num_rsus: int, malicious_ratio: float, seed: int) -> None:
        self.config = config
        self.num_rsus = num_rsus
        self.malicious_ratio = malicious_ratio
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.default_name = str(config["default_name"])
        self.network_cfg = config["network"]
        self.attack_cfg = config["attack"]
        self.honest_base = config["honest_base"]
        self.degradation = config["degradation"]
        self.scenario_state = 0
        self.epoch = 0
        self.online = np.ones(self.num_rsus, dtype=np.int8)
        self.prev_online = self.online.copy()
        self.uptime = np.ones(self.num_rsus, dtype=np.float32)
        self.malicious = np.zeros(self.num_rsus, dtype=np.int8)

    def reset(self, seed: int | None = None) -> None:
        """重置场景内部随机源与节点状态。

        如果外部传入 seed，则本轮 episode 使用新 seed；否则沿用构造时 seed。
        这能保证 benchmark/eval 中不同 episode 真正是“可复现的不同样本”。
        """

        if seed is not None:
            self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.epoch = 0
        self.online = np.ones(self.num_rsus, dtype=np.int8)
        self.prev_online = self.online.copy()
        self.uptime = np.ones(self.num_rsus, dtype=np.float32)
        self.scenario_state = int(self.config.get("mmpp", {}).get("initial_state", 0))
        malicious_count = int(round(self.malicious_ratio * self.num_rsus))
        malicious_idx = self.rng.choice(self.num_rsus, size=malicious_count, replace=False) if malicious_count > 0 else np.array([], dtype=int)
        self.malicious = np.zeros(self.num_rsus, dtype=np.int8)
        self.malicious[malicious_idx] = 1

    def _sample_arrivals(self, epoch: int) -> int:
        """按照配置的到达模型生成本周期到达量 `A_e`。"""
        scenario_cfg = self.config[self.default_name]
        mode = scenario_cfg["type"]
        if mode == "stable":
            lam = float(scenario_cfg["lambda"])
        elif mode == "step":
            lam = float(scenario_cfg["lambda"])
            if epoch >= int(scenario_cfg["e0"]):
                lam *= float(scenario_cfg["k"])
        elif mode == "mmpp":
            lambdas = np.asarray(scenario_cfg["lambdas"], dtype=np.float32)
            transition = np.asarray(scenario_cfg["transition"], dtype=np.float32)
            self.scenario_state = int(self.rng.choice(len(lambdas), p=transition[self.scenario_state]))
            lam = float(lambdas[self.scenario_state])
        else:
            raise ValueError(f"Unsupported scenario type: {mode}")
        return int(self.rng.poisson(lam=max(lam, 1.0)))

    def _update_online_state(self) -> tuple[np.ndarray, float]:
        """更新 RSU 在线/离线状态并返回 churn。

        `churn` 定义为本周期状态发生切换的节点比例，用于后续共识时延与信誉稳定性建模。
        """

        self.prev_online = self.online.copy()
        p_off = float(self.network_cfg["p_off"])
        p_on = float(self.network_cfg["p_on"])
        going_off = (self.prev_online == 1) & (self.rng.random(self.num_rsus) < p_off)
        coming_on = (self.prev_online == 0) & (self.rng.random(self.num_rsus) < p_on)
        self.online = self.prev_online.copy()
        self.online[going_off] = 0
        self.online[coming_on] = 1
        churn = float(np.mean(self.online != self.prev_online))
        self.uptime = 0.9 * self.uptime + 0.1 * self.online.astype(np.float32)
        return self.online.copy(), churn

    def _base_probs(self, load: float, rtt_norm: float, churn: float) -> dict[str, np.ndarray]:
        """根据负载/RTT/churn 生成诚实节点四维基础成功概率。

        对应论文中的：
        `p^H_d(e) = clip(p0_d - a_d * load_e - b_d * RTT_norm_e - c_d * chi_e, 0, 1)`
        """

        probs: dict[str, np.ndarray] = {}
        for dim in REPUTATION_DIMS:
            p0 = float(self.honest_base[dim])
            a_d, b_d, c_d = self.degradation[dim]
            honest = clip01(p0 - a_d * load - b_d * rtt_norm - c_d * churn)
            probs[dim] = np.full(self.num_rsus, honest, dtype=np.float32)
        return probs

    def step(self, epoch: int, queue_size: float, last_latency: float, eligible_size: int) -> ScenarioStep:
        """推进一个治理周期并产出场景快照。

        参数中的 `queue_size` / `last_latency` / `eligible_size` 会在主环境中进入上下文感知信誉融合；
        这里保留接口是为了后续可以继续扩展成“场景对治理结果有二次反馈”的更复杂模型。
        """

        self.epoch = epoch
        arrivals = self._sample_arrivals(epoch)
        online, churn = self._update_online_state()
        rtt = float(self.rng.uniform(self.network_cfg["rtt_min"], self.network_cfg["rtt_max"]))
        rtt_norm = rtt / max(float(self.network_cfg["rtt_max"]), 1.0)
        load = float(np.clip(arrivals / max(self.num_rsus * 15.0, 1.0) + queue_size / max(self.num_rsus * 50.0, 1.0), 0.0, 1.5))
        base_probs = self._base_probs(load=load, rtt_norm=rtt_norm, churn=churn)
        return ScenarioStep(
            epoch=epoch,
            arrivals=arrivals,
            load=load,
            rtt=rtt,
            churn=churn,
            online=online,
            malicious=self.malicious.copy(),
            uptime=self.uptime.copy(),
            base_probs=base_probs,
        )
