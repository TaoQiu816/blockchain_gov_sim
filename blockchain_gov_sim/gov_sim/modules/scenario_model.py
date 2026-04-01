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
    scenario_type: str
    scenario_phase: str


class ScenarioModel:
    """联盟链治理场景生成器。

    职责：
    - 生成平稳 / 阶跃 / MMPP 到达量；
    - 为每个 RSU 维护在线/离线马尔可夫链；
    - 维护节点稳定性 `uptime`；
    - 在当前负载、RTT、churn 下生成诚实节点四维基础成功概率；
    - 训练时按 episode 抽样 mixed scenario，并在局部窗口内触发 regime shift。
    """

    def __init__(
        self,
        config: dict[str, Any],
        num_rsus: int,
        malicious_ratio: float,
        seed: int,
        episode_length: int,
    ) -> None:
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
        self.training_mix_cfg = config.get("training_mix", {})
        self.training_profiles = self.training_mix_cfg.get("profiles", {})
        self.episode_length = max(int(episode_length), 1)
        self.scenario_state = 0
        self.epoch = 0
        self.online = np.ones(self.num_rsus, dtype=np.int8)
        self.prev_online = self.online.copy()
        self.uptime = np.ones(self.num_rsus, dtype=np.float32)
        self.malicious = np.zeros(self.num_rsus, dtype=np.int8)
        self.active_scenario_name = self.default_name
        self.active_profile_cfg: dict[str, Any] = {}
        self.burst_start_epoch = self.episode_length
        self.burst_end_epoch = self.episode_length
        self.extra_malicious_idx = np.array([], dtype=np.int64)

    def reset(self, seed: int | None = None) -> None:
        """重置场景内部随机源与节点状态。"""

        if seed is not None:
            self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.epoch = 0
        self.online = np.ones(self.num_rsus, dtype=np.int8)
        self.prev_online = self.online.copy()
        self.uptime = np.ones(self.num_rsus, dtype=np.float32)
        self.scenario_state = int(self.config.get("mmpp", {}).get("initial_state", 0))
        malicious_count = int(round(self.malicious_ratio * self.num_rsus))
        malicious_idx = (
            self.rng.choice(self.num_rsus, size=malicious_count, replace=False) if malicious_count > 0 else np.array([], dtype=np.int64)
        )
        self.malicious = np.zeros(self.num_rsus, dtype=np.int8)
        self.malicious[malicious_idx] = 1
        self._select_episode_profile()

    def _select_episode_profile(self) -> None:
        """在训练混合场景启用时，为当前 episode 抽样一个 profile。"""

        self.active_scenario_name = self.default_name
        self.active_profile_cfg = {}
        self.burst_start_epoch = self.episode_length
        self.burst_end_epoch = self.episode_length
        self.extra_malicious_idx = np.array([], dtype=np.int64)
        if not bool(self.training_mix_cfg.get("enabled", False)) or not self.training_profiles:
            return

        names = list(self.training_profiles.keys())
        weights = np.asarray([float(self.training_profiles[name].get("weight", 0.0)) for name in names], dtype=np.float64)
        if np.sum(weights) <= 0.0:
            weights = np.full(len(names), 1.0 / max(len(names), 1), dtype=np.float64)
        else:
            weights = weights / np.sum(weights)
        chosen = str(self.rng.choice(names, p=weights))
        self.active_scenario_name = chosen
        self.active_profile_cfg = dict(self.training_profiles.get(chosen, {}))
        start_frac = float(self.active_profile_cfg.get("burst_start_frac", 1.0))
        end_frac = float(self.active_profile_cfg.get("burst_end_frac", 1.0))
        self.burst_start_epoch = int(np.clip(round(start_frac * self.episode_length), 0, self.episode_length - 1))
        self.burst_end_epoch = int(np.clip(round(end_frac * self.episode_length), self.burst_start_epoch + 1, self.episode_length))
        if chosen == "load_shock":
            self.burst_end_epoch = self.episode_length
        if chosen == "malicious_burst":
            honest_idx = np.flatnonzero(self.malicious == 0)
            extra_ratio = float(self.active_profile_cfg.get("extra_malicious_ratio", 0.0))
            extra_count = int(round(extra_ratio * self.num_rsus))
            extra_count = min(extra_count, honest_idx.size)
            if extra_count > 0:
                self.extra_malicious_idx = self.rng.choice(honest_idx, size=extra_count, replace=False).astype(np.int64)

    def _burst_active(self, epoch: int) -> bool:
        return self.burst_start_epoch <= int(epoch) < self.burst_end_epoch

    def _scenario_phase(self, epoch: int) -> str:
        if not self.active_profile_cfg:
            return "default"
        if self.active_scenario_name == "load_shock":
            return "shock" if self._burst_active(epoch) else "pre_shock"
        return "burst" if self._burst_active(epoch) else "base"

    def _sample_default_arrivals(self, epoch: int) -> int:
        """按照原始默认场景配置生成本周期到达量 `A_e`。"""

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

    def _base_lambda(self) -> float:
        scenario_cfg = self.config[self.default_name]
        mode = scenario_cfg["type"]
        if mode == "stable":
            return float(scenario_cfg["lambda"])
        if mode == "step":
            return float(scenario_cfg["lambda"])
        if mode == "mmpp":
            return float(np.mean(np.asarray(scenario_cfg["lambdas"], dtype=np.float32)))
        raise ValueError(f"Unsupported scenario type: {mode}")

    def _sample_arrivals(self, epoch: int) -> int:
        """按照当前 episode profile 生成本周期到达量 `A_e`。"""

        if not self.active_profile_cfg:
            return self._sample_default_arrivals(epoch)
        if self.active_scenario_name == "load_shock":
            base_lambda = self._base_lambda()
            lam_scale = (
                float(self.active_profile_cfg.get("high_lambda_scale", 1.0))
                if self._burst_active(epoch)
                else float(self.active_profile_cfg.get("low_lambda_scale", 1.0))
            )
            lam = max(base_lambda * lam_scale, 1.0)
            return int(self.rng.poisson(lam=lam))
        return int(self.rng.poisson(lam=max(self._base_lambda(), 1.0)))

    def _network_params(self, epoch: int) -> tuple[float, float]:
        """返回当前周期 online/offline 马尔可夫链参数。"""

        p_off = float(self.network_cfg["p_off"])
        p_on = float(self.network_cfg["p_on"])
        if self.active_scenario_name == "churn_burst" and self._burst_active(epoch):
            p_off = float(self.active_profile_cfg.get("p_off", p_off))
            p_on = float(self.active_profile_cfg.get("p_on", p_on))
        return p_off, p_on

    def _update_online_state(self, epoch: int) -> tuple[np.ndarray, float]:
        """更新 RSU 在线/离线状态并返回 churn。"""

        self.prev_online = self.online.copy()
        p_off, p_on = self._network_params(epoch)
        going_off = (self.prev_online == 1) & (self.rng.random(self.num_rsus) < p_off)
        coming_on = (self.prev_online == 0) & (self.rng.random(self.num_rsus) < p_on)
        self.online = self.prev_online.copy()
        self.online[going_off] = 0
        self.online[coming_on] = 1
        churn = float(np.mean(self.online != self.prev_online))
        self.uptime = 0.9 * self.uptime + 0.1 * self.online.astype(np.float32)
        return self.online.copy(), churn

    def _sample_rtt(self, epoch: int) -> float:
        """返回当前周期 RTT。"""

        rtt_min = float(self.network_cfg["rtt_min"])
        rtt_max = float(self.network_cfg["rtt_max"])
        if self.active_scenario_name == "high_rtt_burst" and self._burst_active(epoch):
            rtt_min = float(self.active_profile_cfg.get("rtt_min", rtt_min))
            rtt_max = float(self.active_profile_cfg.get("rtt_max", rtt_max))
        return float(self.rng.uniform(rtt_min, rtt_max))

    def _effective_malicious(self, epoch: int) -> np.ndarray:
        """返回当前周期有效恶意节点集合。"""

        effective = self.malicious.copy()
        if self.active_scenario_name == "malicious_burst" and self._burst_active(epoch) and self.extra_malicious_idx.size > 0:
            effective[self.extra_malicious_idx] = 1
        return effective

    def _base_probs(self, load: float, rtt_norm: float, churn: float) -> dict[str, np.ndarray]:
        """根据负载/RTT/churn 生成诚实节点四维基础成功概率。"""

        probs: dict[str, np.ndarray] = {}
        for dim in REPUTATION_DIMS:
            p0 = float(self.honest_base[dim])
            a_d, b_d, c_d = self.degradation[dim]
            honest = clip01(p0 - a_d * load - b_d * rtt_norm - c_d * churn)
            probs[dim] = np.full(self.num_rsus, honest, dtype=np.float32)
        return probs

    def step(self, epoch: int, queue_size: float, last_latency: float, eligible_size: int) -> ScenarioStep:
        """推进一个治理周期并产出场景快照。"""

        self.epoch = epoch
        arrivals = self._sample_arrivals(epoch)
        online, churn = self._update_online_state(epoch)
        rtt = self._sample_rtt(epoch)
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
            malicious=self._effective_malicious(epoch),
            uptime=self.uptime.copy(),
            base_probs=base_probs,
            scenario_type=self.active_scenario_name,
            scenario_phase=self._scenario_phase(epoch),
        )
