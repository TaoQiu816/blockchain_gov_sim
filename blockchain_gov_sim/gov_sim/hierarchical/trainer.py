"""分层治理器训练、评估与 hard 对比。"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gov_sim.agent.callbacks import TrainLoggingCallback
from gov_sim.baselines import BaselineBase, StaticParamBaseline
from gov_sim.experiments import build_model, make_env, with_training_mix
from gov_sim.experiments.eval_runner import load_controller
from gov_sim.hierarchical.controller import (
    EpisodeSampledTemplateSelector,
    HierarchicalActionConstraintError,
    HierarchicalPolicyController,
    HighLevelInferenceAdapter,
    LowLevelInferenceAdapter,
    PolicyTemplateSelector,
)
from gov_sim.hierarchical.envs import HighLevelGovEnv, LowLevelGovEnv
from gov_sim.hierarchical.high_imitation import train_stage15_supervised_high_policy
from gov_sim.hierarchical.oracle_pretrain import OracleGuidedLowPolicy, train_stage1_supervised_low_policy
from gov_sim.hierarchical.spec import (
    DEFAULT_HIGH_UPDATE_INTERVAL,
    HIGH_LEVEL_TEMPLATES,
    LOW_LEVEL_ACTIONS,
)
from gov_sim.modules.metrics_tracker import MetricsTracker
from gov_sim.utils.device import device_runtime_info, resolve_device
from gov_sim.utils.io import deep_update, ensure_dir, write_json
from gov_sim.utils.train_artifacts import generate_train_artifacts

try:
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3 import PPO
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "导入分层训练 runner 失败：缺少 stable-baselines3 / sb3-contrib。\n"
        "请执行：pip install -r base_env_delta_requirements.txt"
    ) from exc

TOP_STATIC_ACTIONS: dict[str, tuple[int, int, int, float]] = {
    "load_shock": (7, 384, 80, 0.5),
    "high_rtt_burst": (7, 384, 40, 0.5),
    "churn_burst": (7, 256, 40, 0.5),
    "malicious_burst": (7, 384, 40, 0.5),
}
DEFAULT_FLAT_MODEL_CANDIDATES: tuple[str, ...] = (
    "outputs/obs_fix_hard/train/hard_obsfix_seed42_stage2_hard/model.zip",
    "outputs/hard_biasfix/train/hard_seed42_biasfix/model.zip",
    "outputs/hard_block_slack/train/hard_seed42_blockslack/model.zip",
    "outputs/hard_margin_cost/train/hard_seed42_margincost/model.zip",
    "outputs/formal/train/hard_seed42/model.zip",
)


class FixedActionController(BaselineBase):
    """hard 四场景 top static。"""

    def __init__(self, config: dict[str, Any], action_tuple: tuple[int, int, int, float]) -> None:
        super().__init__(config=config, name="Top-Static")
        self.action_tuple = action_tuple

    def select_action(self, env: Any, obs: dict[str, Any]) -> int:
        del obs
        self._set_committee_method(env, "soft_sortition")
        m, b, tau, theta = self.action_tuple
        return self._nearest_action(m=m, b=b, tau=tau, theta=theta)


def _hier_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return config.setdefault("hierarchical", {})


def _agent_override_config(base_config: dict[str, Any], overrides: dict[str, Any], run_name: str, output_root: Path) -> dict[str, Any]:
    updated = deepcopy(base_config)
    updated["agent"] = deep_update(updated["agent"], overrides)
    updated["run_name"] = run_name
    updated["output_root"] = str(output_root)
    return updated


def _training_stage_config(config: dict[str, Any]) -> dict[str, Any]:
    return with_training_mix(
        config,
        enabled=bool(config.get("scenario", {}).get("training_mix", {}).get("enabled_in_train", False)),
    )


def _make_low_vec_env(config: dict[str, Any], selector: EpisodeSampledTemplateSelector | Any) -> DummyVecEnv:
    return DummyVecEnv([lambda: Monitor(LowLevelGovEnv(config=config, template_selector=selector))])


def _make_high_vec_env(config: dict[str, Any], low_policy: LowLevelInferenceAdapter, update_interval: int) -> DummyVecEnv:
    return DummyVecEnv([lambda: Monitor(HighLevelGovEnv(config=config, low_policy=low_policy, update_interval=update_interval))])


def _save_stage_artifacts(stage_dir: Path) -> None:
    if (stage_dir / "train_log.csv").exists():
        generate_train_artifacts(stage_dir)


def _save_stage_summary(stage_dir: Path, summary: dict[str, Any]) -> None:
    write_json(stage_dir / "train_summary.json", summary)
    _save_stage_artifacts(stage_dir)


def _stage3_low_actor_train_mode(joint_cfg: dict[str, Any]) -> str:
    mode = str(joint_cfg.get("low_actor_train_mode", "frozen")).strip().lower()
    if mode != "frozen":
        raise ValueError("Unsupported stage3.low_actor_train_mode: only 'frozen' is allowed in the finalized protocol.")
    return mode


def _make_train_callback(log_path: Path, audit_path: Path) -> TrainLoggingCallback:
    return TrainLoggingCallback(log_path=log_path, audit_path=audit_path)


def _stage1_train_mode(stage1_cfg: dict[str, Any]) -> str:
    mode = str(stage1_cfg.get("train_mode", "oracle_supervised")).strip().lower()
    if mode != "oracle_supervised":
        raise ValueError("Unsupported stage1.train_mode: only 'oracle_supervised' is allowed in the finalized protocol.")
    return mode


def _audit_frozen_low_actor(
    config: dict[str, Any],
    high_model: Any,
    low_model: Any,
    stage_dir: Path,
    round_name: str,
    update_interval: int,
    total_timesteps: int,
) -> dict[str, Any]:
    selector = PolicyTemplateSelector(
        high_policy=HighLevelInferenceAdapter(model=high_model, deterministic=True),
        update_interval=update_interval,
    )
    stage_dir.mkdir(parents=True, exist_ok=True)
    env = LowLevelGovEnv(config=config, template_selector=selector)
    action_counts: Counter[str] = Counter()
    infeasible = 0.0
    invalid = 0.0
    rows: list[dict[str, Any]] = []
    steps = 0
    episodes = 0

    obs, info = env.reset(seed=int(config["seed"]))
    while steps < total_timesteps:
        mask = obs["action_mask"].astype(bool)
        if not bool(mask.sum()):
            infeasible += float(info.get("infeasible", 0.0))
            invalid += float(info.get("invalid_action", 0.0))
            episodes += 1
            obs, info = env.reset(seed=int(config["seed"]) + episodes)
            continue
        action, _ = low_model.predict(obs, deterministic=True, action_masks=mask)
        action_idx = int(np.asarray(action).reshape(-1)[0])
        obs, _, terminated, truncated, info = env.step(action_idx)
        executed_low_action = str(info.get("executed_low_action", ""))
        if executed_low_action:
            action_counts[executed_low_action] += 1
        infeasible += float(info.get("infeasible", 0.0))
        invalid += float(info.get("invalid_action", 0.0))
        rows.append(
            {
                "timesteps": steps + 1,
                "stage3_low_actor_frozen": 1.0,
                "infeasible_rate": float(infeasible / max(steps + 1, 1)),
                "invalid_action_rate": float(invalid / max(steps + 1, 1)),
            }
        )
        steps += 1
        if terminated or truncated:
            episodes += 1
            obs, info = env.reset(seed=int(config["seed"]) + episodes)

    log_path = stage_dir / f"{round_name}_low_train_log.csv"
    pd.DataFrame(rows).to_csv(log_path, index=False)
    total_actions = max(sum(action_counts.values()), 1)
    summary = {
        "episodes": int(max(episodes, 1)),
        "executed_low_action_distribution": [
            {"action": action, "count": int(count), "ratio": float(count / total_actions)}
            for action, count in action_counts.most_common()
        ],
        "infeasible_rate": float(infeasible / max(steps, 1)),
        "invalid_action_rate": float(invalid / max(steps, 1)),
        "stage3_low_actor_frozen": 1,
    }
    write_json(stage_dir / f"{round_name}_low_train_audit.json", summary)
    return summary


def _train_low_stage(base_config: dict[str, Any], stage_dir: Path) -> tuple[Any, dict[str, Any]]:
    hier_cfg = _hier_cfg(base_config)
    low_agent_cfg = hier_cfg.get("low_agent", {})
    stage1_cfg = hier_cfg.get("stage1", {})
    train_mode = _stage1_train_mode(stage1_cfg)
    pretrained_model_value = stage1_cfg.get("pretrained_model_path", "")
    pretrained_model_path = "" if pretrained_model_value in (None, "") else str(pretrained_model_value).strip()
    if pretrained_model_path:
        resolved_path = Path(pretrained_model_path)
        if not resolved_path.is_absolute():
            resolved_path = Path.cwd() / resolved_path
        if not resolved_path.exists():
            raise FileNotFoundError(f"Configured stage1 pretrained low model does not exist: {resolved_path}")
        model = OracleGuidedLowPolicy.load(
            resolved_path,
            device=resolve_device(low_agent_cfg.get("device")),
        )
        model.save(stage_dir / "model")
        summary = {
            "stage": "stage1_low_pretrain",
            "train_mode": train_mode,
            "reused_pretrained_model": 1,
            "pretrained_model_path": str(resolved_path),
            "device": str(model.device),
            "device_runtime": device_runtime_info(),
        }
        _save_stage_summary(stage_dir, summary)
        return model, summary
    stage_cfg = _training_stage_config(
        _agent_override_config(base_config, overrides=low_agent_cfg, run_name="stage1_low_pretrain", output_root=stage_dir.parent)
    )
    del train_mode
    model, summary = train_stage1_supervised_low_policy(base_config=base_config, stage_cfg=stage_cfg, stage_dir=stage_dir)
    summary["device_runtime"] = device_runtime_info()
    _save_stage_summary(stage_dir, summary)
    return model, summary


def _train_high_imitation_stage(base_config: dict[str, Any], low_model: Any, stage_dir: Path) -> tuple[Any, dict[str, Any]]:
    model, summary = train_stage15_supervised_high_policy(base_config=base_config, stage_cfg=base_config, low_model=low_model, stage_dir=stage_dir)
    summary["device_runtime"] = device_runtime_info()
    _save_stage_summary(stage_dir, summary)
    return model, summary


def _configure_oracle_anchor(
    base_config: dict[str, Any],
    model: Any,
    stage_dir: Path,
    *,
    stage_key: str,
    default_enabled: bool,
) -> None:
    stage_cfg = _hier_cfg(base_config).get(stage_key, {})
    anchor_cfg = dict(stage_cfg.get("oracle_anchor", {}))
    if not bool(anchor_cfg.get("enabled", default_enabled)):
        if hasattr(model, "clear_oracle_anchor"):
            model.clear_oracle_anchor()
        return
    dataset_path = stage_dir.parent / "stage15_high_imitation" / "high_imitation_dataset.npz"
    if not dataset_path.exists():
        raise FileNotFoundError(f"{stage_key} oracle anchor dataset does not exist: {dataset_path}")
    if not hasattr(model, "load_oracle_anchor_dataset") or not hasattr(model, "set_oracle_anchor_schedule"):
        raise TypeError("Current high model does not support oracle-anchor configuration.")
    model.load_oracle_anchor_dataset(dataset_path)
    model.set_oracle_anchor_schedule(
        beta_init=float(anchor_cfg.get("beta_init", 1.0)),
        beta_final=float(anchor_cfg.get("beta_final", 0.0)),
        decay_fraction=float(anchor_cfg.get("decay_fraction", 0.5)),
        batch_size=anchor_cfg.get("batch_size"),
    )


def _train_high_stage(base_config: dict[str, Any], low_model: Any, stage_dir: Path, initial_model: Any | None = None) -> tuple[Any, dict[str, Any]]:
    hier_cfg = _hier_cfg(base_config)
    update_interval = int(hier_cfg.get("update_interval", DEFAULT_HIGH_UPDATE_INTERVAL))
    high_agent_cfg = hier_cfg.get("high_agent", {})
    stage_cfg = _training_stage_config(
        _agent_override_config(base_config, overrides=high_agent_cfg, run_name="stage2_high_train", output_root=stage_dir.parent)
    )
    low_policy = LowLevelInferenceAdapter(model=low_model, deterministic=True)
    env = _make_high_vec_env(config=stage_cfg, low_policy=low_policy, update_interval=update_interval)
    model = initial_model if initial_model is not None else build_model(config=stage_cfg, env=env, use_lagrangian=True)
    model.set_env(env)
    _configure_oracle_anchor(base_config=base_config, model=model, stage_dir=stage_dir, stage_key="stage2", default_enabled=True)
    callback = _make_train_callback(log_path=stage_dir / "train_log.csv", audit_path=stage_dir / "train_audit.json")
    base_timesteps = int(hier_cfg.get("stage2", {}).get("total_base_timesteps", 12000))
    high_timesteps = int(math.ceil(base_timesteps / max(update_interval, 1)))
    model.learn(total_timesteps=high_timesteps, callback=callback, progress_bar=False)
    model.save(str(stage_dir / "model"))
    summary = {
        "stage": "stage2_high_train",
        "base_timesteps": base_timesteps,
        "high_level_timesteps": high_timesteps,
        "device": str(getattr(model, "device", "unknown")),
        "device_runtime": device_runtime_info(),
        "training_audit": callback.audit_summary(),
    }
    _save_stage_summary(stage_dir, summary)
    return model, summary


def _stage3_high_refine(base_config: dict[str, Any], high_model: Any, low_model: Any, stage_dir: Path) -> tuple[Any, Any, dict[str, Any]]:
    hier_cfg = _hier_cfg(base_config)
    update_interval = int(hier_cfg.get("update_interval", DEFAULT_HIGH_UPDATE_INTERVAL))
    low_agent_cfg = hier_cfg.get("low_agent", {})
    high_agent_cfg = hier_cfg.get("high_agent", {})
    stage3_cfg = hier_cfg.get("stage3", {})
    low_actor_train_mode = _stage3_low_actor_train_mode(stage3_cfg)
    rounds = int(stage3_cfg.get("rounds", 3))
    low_base_per_round = int(stage3_cfg.get("low_base_timesteps_per_round", 2000))
    high_base_per_round = int(stage3_cfg.get("high_base_timesteps_per_round", 2000))
    round_summaries: list[dict[str, Any]] = []

    for round_idx in range(rounds):
        round_name = f"round_{round_idx + 1:02d}"

        low_stage_cfg = _training_stage_config(
            _agent_override_config(base_config, overrides=low_agent_cfg, run_name=f"{round_name}_low", output_root=stage_dir)
        )
        policy_selector = PolicyTemplateSelector(
            high_policy=HighLevelInferenceAdapter(model=high_model, deterministic=True),
            update_interval=update_interval,
        )
        low_audit = _audit_frozen_low_actor(
            config=low_stage_cfg,
            high_model=high_model,
            low_model=low_model,
            stage_dir=stage_dir,
            round_name=round_name,
            update_interval=update_interval,
            total_timesteps=low_base_per_round,
        )

        high_stage_cfg = _training_stage_config(
            _agent_override_config(base_config, overrides=high_agent_cfg, run_name=f"{round_name}_high", output_root=stage_dir)
        )
        low_policy = LowLevelInferenceAdapter(model=low_model, deterministic=True)
        high_env = _make_high_vec_env(config=high_stage_cfg, low_policy=low_policy, update_interval=update_interval)
        high_model.set_env(high_env)
        _configure_oracle_anchor(base_config=base_config, model=high_model, stage_dir=stage_dir, stage_key="stage3", default_enabled=False)
        high_callback = _make_train_callback(log_path=stage_dir / f"{round_name}_high_train_log.csv", audit_path=stage_dir / f"{round_name}_high_train_audit.json")
        high_timesteps = int(math.ceil(high_base_per_round / max(update_interval, 1)))
        high_model.learn(total_timesteps=high_timesteps, callback=high_callback, progress_bar=False, reset_num_timesteps=False)

        round_summaries.append(
            {
                "round": round_idx + 1,
                "low_base_timesteps": low_base_per_round,
                "high_base_timesteps": high_base_per_round,
                "high_level_timesteps": high_timesteps,
                "low_actor_train_mode": low_actor_train_mode,
                "low_audit": low_audit,
                "high_audit": high_callback.audit_summary(),
            }
        )

    low_model.save(str(stage_dir / "low_model"))
    high_model.save(str(stage_dir / "high_model"))
    summary = {
        "stage": "stage3_high_refine",
        "rounds": rounds,
        "update_interval": update_interval,
        "low_base_timesteps_per_round": low_base_per_round,
        "high_base_timesteps_per_round": high_base_per_round,
        "device_runtime": device_runtime_info(),
        "low_actor_train_mode": low_actor_train_mode,
        "round_summaries": round_summaries,
    }
    _save_stage_summary(stage_dir, summary)
    return high_model, low_model, summary


def _build_dynamic_config(base_config: dict[str, Any], scenario_name: str, run_name: str) -> dict[str, Any]:
    config = deepcopy(base_config)
    profile = deepcopy(config["scenario"]["training_mix"]["profiles"][scenario_name])
    profile["weight"] = 1.0
    config["scenario"]["training_mix"] = {
        "enabled": True,
        "enabled_in_train": True,
        "profiles": {scenario_name: profile},
    }
    config["run_name"] = run_name
    return config


def _action_summary(frame: pd.DataFrame, cols: list[str]) -> dict[str, Any]:
    if frame.empty:
        return {"dominant_ratio": 0.0, "top": "", "distribution": []}
    signature = frame[cols[0]].astype(str)
    for col in cols[1:]:
        signature = signature + "|" + frame[col].astype(str)
    counts = signature.value_counts(normalize=True)
    top_key = str(counts.index[0])
    top_ratio = float(counts.iloc[0])
    return {
        "dominant_ratio": top_ratio,
        "top": top_key,
        "distribution": [{"action": str(idx), "ratio": float(value)} for idx, value in counts.head(5).items()],
    }


def _string_action_summary(frame: pd.DataFrame, col: str) -> dict[str, Any]:
    if frame.empty or col not in frame:
        return {"dominant_ratio": 0.0, "top": "", "distribution": []}
    values = frame[col].fillna("").astype(str)
    values = values[values != ""]
    if values.empty:
        return {"dominant_ratio": 0.0, "top": "", "distribution": []}
    counts = values.value_counts(normalize=True)
    top_key = str(counts.index[0])
    top_ratio = float(counts.iloc[0])
    return {
        "dominant_ratio": top_ratio,
        "top": top_key,
        "distribution": [{"action": str(idx), "ratio": float(value)} for idx, value in counts.head(5).items()],
    }



def _evaluate_controller(
    controller: Any,
    config: dict[str, Any],
    controller_name: str,
    episodes: int,
    controller_type: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    env = make_env(config)
    if controller_type == "baseline" and hasattr(env, "set_invalid_action_mode"):
        env.set_invalid_action_mode("terminate")
    elif controller_type == "hierarchical" and hasattr(env, "set_invalid_action_mode"):
        env.set_invalid_action_mode("raise")
    tracker = MetricsTracker()
    rows: list[dict[str, Any]] = []

    for episode in range(episodes):
        obs, _ = env.reset(seed=int(config["seed"]) + episode)
        if hasattr(controller, "reset"):
            controller.reset()
        done = False
        step_idx = 0
        while not done:
            try:
                if controller_type == "flat":
                    if isinstance(controller, PPO):
                        action, _ = controller.predict(obs, deterministic=True)
                    else:
                        action, _ = controller.predict(obs, deterministic=True, action_masks=obs["action_mask"])
                    if hasattr(action, "item"):
                        action = action.item()
                else:
                    action = controller.select_action(env, obs)
                obs, reward, terminated, truncated, info = env.step(int(action))
                done = terminated or truncated
            except HierarchicalActionConstraintError:
                scenario = getattr(env, "current_scenario", None)
                info = {
                    "epoch": int(getattr(env, "epoch", 0)),
                    "episode_seed": int(getattr(env, "current_episode_seed", int(config["seed"]) + episode)),
                    "scenario_type": str(getattr(scenario, "scenario_type", "")),
                    "scenario_phase": str(getattr(scenario, "scenario_phase", "")),
                    "reward": 0.0,
                    "cost": 0.0,
                    "tps": 0.0,
                    "L_bar_e": 0.0,
                    "unsafe": 0,
                    "pollute_rate": 0.0,
                    "timeout_failure": 0,
                    "mask_ratio": 0.0,
                    "eligible_size": int(getattr(env, "_eligible_nodes")(getattr(env, "prev_action").theta).size),
                    "Q_e": float(getattr(env, "queue_size", 0.0)),
                    "A_e": int(getattr(scenario, "arrivals", 0)) if scenario is not None else 0,
                    "Z_e": 0,
                    "infeasible": 1,
                    "infeasible_reason": "no_legal_high_template",
                    "invalid_action": 1,
                    "policy_invalid": 0,
                    "structural_infeasible": 1,
                    "action_source": "invalid_abort",
                    "attempted_action": "",
                    "executed_action": "",
                    "attempted_action_idx": -1,
                    "executed_action_idx": -1,
                    "policy_high_template": "",
                    "selected_high_template": "",
                    "executed_high_template": "",
                    "selected_low_action": "",
                    "executed_low_action": "",
                    "num_legal_nominal_high_templates": 0,
                    "m_e": 0,
                    "b_e": 0,
                    "tau_e": 0,
                    "theta_e": -1.0,
                    "committee_members": [],
                }
                reward = 0.0
                terminated = True
                truncated = False
                done = True
            tracker.update(info)
            row = info.copy()
            row["episode"] = episode
            row["step"] = step_idx
            row["controller"] = controller_name
            row["reward"] = float(reward)
            row["template"] = f"{int(info['m_e'])}|{float(info['theta_e']):.2f}"
            row["low_action"] = f"{int(info['b_e'])}|{int(info['tau_e'])}"
            rows.append(row)
            step_idx += 1

    frame = pd.DataFrame(rows)
    summary = tracker.summary()
    template_summary = _string_action_summary(frame=frame, col="executed_high_template")
    low_summary = _string_action_summary(frame=frame, col="executed_low_action")
    policy_summary = _string_action_summary(frame=frame, col="policy_high_template")
    summary["template_dominant_ratio"] = float(template_summary["dominant_ratio"])
    summary["top_template"] = template_summary["top"]
    summary["template_distribution"] = template_summary["distribution"]
    summary["low_action_dominant_ratio"] = float(low_summary["dominant_ratio"])
    summary["top_low_action"] = low_summary["top"]
    summary["low_action_distribution"] = low_summary["distribution"]
    summary["top_policy_high_template"] = policy_summary["top"]
    summary["policy_high_template_distribution"] = policy_summary["distribution"]
    return frame, summary


def _judge_results(compare_df: pd.DataFrame) -> dict[str, Any]:
    wins = 0
    residual_perf_ok = True
    unsafe_ok = True
    policy_invalid_zero = True
    structural_ok = True
    hierarchical = compare_df[compare_df["controller"] == "hierarchical_learned"].set_index("scenario")
    top_static = compare_df[compare_df["controller"] == "top_static"].set_index("scenario")
    per_scenario: dict[str, dict[str, Any]] = {}

    for scenario in TOP_STATIC_ACTIONS:
        hier_row = hierarchical.loc[scenario]
        top_row = top_static.loc[scenario]
        hier_tps = float(hier_row["TPS"])
        top_tps = float(top_row["TPS"])
        hier_latency = float(hier_row["mean_latency"])
        top_latency = float(top_row["mean_latency"])
        hier_unsafe = float(hier_row["unsafe_rate"])
        top_unsafe = float(top_row["unsafe_rate"])
        tps_ratio = hier_tps / max(top_tps, 1e-12)
        latency_ratio = hier_latency / max(top_latency, 1e-12)
        is_win = hier_tps > top_tps and hier_unsafe <= top_unsafe + 0.02
        if is_win:
            wins += 1
        else:
            if hier_tps < 0.95 * top_tps or hier_latency > 1.05 * top_latency:
                residual_perf_ok = False
        if hier_unsafe > top_unsafe + 0.02:
            unsafe_ok = False
        if float(hier_row.get("policy_invalid_rate", 0.0)) > 0.0:
            policy_invalid_zero = False
        structural_reference = float(top_row.get("structural_infeasible_rate", 0.0))
        if float(hier_row.get("structural_infeasible_rate", 0.0)) > structural_reference + 0.005:
            structural_ok = False
        per_scenario[scenario] = {
            "win_vs_top_static": bool(is_win),
            "tps_ratio_vs_top_static": float(tps_ratio),
            "latency_ratio_vs_top_static": float(latency_ratio),
            "unsafe_delta_vs_top_static": float(hier_unsafe - top_unsafe),
            "policy_invalid_rate": float(hier_row.get("policy_invalid_rate", 0.0)),
            "structural_infeasible_rate": float(hier_row.get("structural_infeasible_rate", 0.0)),
        }

    unique_templates = set(hierarchical["top_template"].tolist())
    unique_low_actions = set(hierarchical["top_low_action"].tolist())
    low_level_not_collapsed = not bool((hierarchical["low_action_dominant_ratio"].astype(float) >= 0.999).all())
    success = wins >= 2 and residual_perf_ok and unsafe_ok and policy_invalid_zero and structural_ok and low_level_not_collapsed

    next_step = "进入正式 5 seed。" if success else "单 seed 未达标，仅按拆分后的 strict eval 口径继续复核。"
    return {
        "wins_vs_top_static": int(wins),
        "residual_perf_ok": bool(residual_perf_ok),
        "unsafe_ok": bool(unsafe_ok),
        "policy_invalid_zero": bool(policy_invalid_zero),
        "structural_infeasible_ok": bool(structural_ok),
        "high_template_focus": bool(len(unique_templates) <= 4),
        "low_action_switching": bool(len(unique_low_actions) >= 2),
        "low_level_not_single_point_collapsed": bool(low_level_not_collapsed),
        "solves_structural_issue": bool(success),
        "can_run_5_seeds": bool(success),
        "per_scenario": per_scenario,
        "next_step": next_step,
    }


def _load_flat_comparator(base_config: dict[str, Any], preferred_path: str | None) -> tuple[Any, str]:
    candidates: list[str] = []
    if preferred_path:
        candidates.append(preferred_path)
    candidates.extend(path for path in DEFAULT_FLAT_MODEL_CANDIDATES if path not in candidates)
    last_error: Exception | None = None
    for path in candidates:
        try:
            model, is_baseline = load_controller(model_path=path, config=base_config, baseline_name=None)
            if is_baseline:
                continue
            return model, path
        except Exception as exc:  # pragma: no cover - 只在候选模型不兼容时触发
            last_error = exc
    raise RuntimeError(f"Unable to load a compatible flat learned comparator. Last error: {last_error}")


def evaluate_hierarchical_suite(
    base_config: dict[str, Any],
    high_model: Any,
    low_model: Any,
    output_dir: Path,
) -> dict[str, Any]:
    hier_cfg = _hier_cfg(base_config)
    compare_cfg = hier_cfg.get("compare", {})
    episodes = int(compare_cfg.get("episodes", 4))
    update_interval = int(hier_cfg.get("update_interval", DEFAULT_HIGH_UPDATE_INTERVAL))
    skip_flat_comparator = bool(compare_cfg.get("skip_flat_comparator", False))
    selected_flat_model_path = ""
    flat_model = None
    if not skip_flat_comparator:
        flat_model_path = str(compare_cfg.get("flat_model_path", DEFAULT_FLAT_MODEL_CANDIDATES[0]))
        flat_model, selected_flat_model_path = _load_flat_comparator(base_config=base_config, preferred_path=flat_model_path)

    hierarchical_controller = HierarchicalPolicyController(
        high_model=high_model,
        low_model=low_model,
        update_interval=update_interval,
        deterministic=True,
    )
    static_param = StaticParamBaseline(base_config)

    compare_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "episodes": episodes,
        "flat_model_path": selected_flat_model_path,
        "action_space": {
            "policy_high_templates": [f"{m}|{theta:.2f}" for m, theta in HIGH_LEVEL_TEMPLATES],
            "low_actions": [f"{b}|{tau}" for b, tau in LOW_LEVEL_ACTIONS],
        },
        "scenarios": {},
    }
    for scenario_name, static_action in TOP_STATIC_ACTIONS.items():
        scenario_config = _build_dynamic_config(base_config, scenario_name=scenario_name, run_name=f"hier_{scenario_name}")
        top_static = FixedActionController(config=base_config, action_tuple=static_action)
        summary["scenarios"][scenario_name] = {}
        controller_specs = [
            ("hierarchical_learned", hierarchical_controller, "hierarchical"),
            ("top_static", top_static, "baseline"),
            ("Static-Param", static_param, "baseline"),
        ]
        if flat_model is not None:
            controller_specs.insert(2, ("flat_learned", flat_model, "flat"))
        for controller_name, controller, controller_type in controller_specs:
            frame, metrics = _evaluate_controller(
                controller=controller,
                config=scenario_config,
                controller_name=controller_name,
                episodes=episodes,
                controller_type=controller_type,
            )
            frame.to_csv(output_dir / f"{scenario_name}_{controller_name}.csv", index=False)
            summary["scenarios"][scenario_name][controller_name] = metrics
            compare_rows.append(
                {
                    "scenario": scenario_name,
                    "controller": controller_name,
                    "TPS": float(metrics.get("TPS", metrics.get("tps", 0.0))),
                    "mean_latency": float(metrics.get("mean_latency", 0.0)),
                    "unsafe_rate": float(metrics.get("unsafe_rate", 0.0)),
                    "invalid_action_rate": float(metrics.get("invalid_action_rate", 0.0)),
                    "policy_invalid_rate": float(metrics.get("policy_invalid_rate", 0.0)),
                    "structural_infeasible_rate": float(metrics.get("structural_infeasible_rate", 0.0)),
                    "timeout_rate": float(metrics.get("timeout_failure_rate", 0.0)),
                    "infeasible_rate": float(metrics.get("infeasible_rate", 0.0)),
                    "queue_peak": float(metrics.get("queue_peak", 0.0)),
                    "top_template": str(metrics.get("top_template", "")),
                    "template_dominant_ratio": float(metrics.get("template_dominant_ratio", 0.0)),
                    "top_low_action": str(metrics.get("top_low_action", "")),
                    "low_action_dominant_ratio": float(metrics.get("low_action_dominant_ratio", 0.0)),
                }
            )

    compare_df = pd.DataFrame(compare_rows)
    compare_df.to_csv(output_dir / "hard_compare.csv", index=False)
    judgement = _judge_results(compare_df)
    summary["judgement"] = judgement
    write_json(output_dir / "hard_compare.json", summary)
    return {"compare_df": compare_df, "summary": summary}


def run_hierarchical_training(config: dict[str, Any]) -> dict[str, Any]:
    """按 24k / 12k / 12k 协议训练并在 hard 四场景评估。"""

    root_dir = ensure_dir(Path(config["output_root"]) / "hierarchical" / config["run_name"])
    config_snapshot_path = root_dir / "config_snapshot.json"
    stage1_dir = ensure_dir(root_dir / "stage1_low_pretrain")
    stage15_dir = ensure_dir(root_dir / "stage15_high_imitation")
    stage2_dir = ensure_dir(root_dir / "stage2_high_train")
    stage3_dir = ensure_dir(root_dir / "stage3_high_refine")
    eval_dir = ensure_dir(root_dir / "hard_eval")
    write_json(config_snapshot_path, config)

    low_model, stage1_summary = _train_low_stage(base_config=config, stage_dir=stage1_dir)
    high_model, stage15_summary = _train_high_imitation_stage(base_config=config, low_model=low_model, stage_dir=stage15_dir)
    high_model, stage2_summary = _train_high_stage(base_config=config, low_model=low_model, stage_dir=stage2_dir, initial_model=high_model)
    high_model, low_model, stage3_summary = _stage3_high_refine(
        base_config=config,
        high_model=high_model,
        low_model=low_model,
        stage_dir=stage3_dir,
    )
    eval_result = evaluate_hierarchical_suite(base_config=config, high_model=high_model, low_model=low_model, output_dir=eval_dir)

    manifest = {
        "output_dir": str(root_dir),
        "action_space": {
            "policy_high_templates": [f"{m}|{theta:.2f}" for m, theta in HIGH_LEVEL_TEMPLATES],
            "low_actions": [f"{b}|{tau}" for b, tau in LOW_LEVEL_ACTIONS],
        },
        "config_snapshot_path": str(config_snapshot_path),
        "stage1_low_pretrain": stage1_summary,
        "stage15_high_imitation": stage15_summary,
        "stage2_high_train": stage2_summary,
        "stage3_high_refine": stage3_summary,
        "hard_eval": eval_result["summary"],
        "artifacts": {
            "low_model_path": str(stage3_dir / "low_model.zip"),
            "high_model_path": str(stage3_dir / "high_model.zip"),
            "hard_compare_csv": str(eval_dir / "hard_compare.csv"),
            "hard_compare_json": str(eval_dir / "hard_compare.json"),
        },
    }
    write_json(root_dir / "hierarchical_summary.json", manifest)
    return manifest
