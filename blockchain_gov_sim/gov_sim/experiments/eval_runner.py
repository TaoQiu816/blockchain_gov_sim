"""评估入口 runner。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO

from gov_sim.agent.masked_ppo_lagrangian import MaskablePPOLagrangian
from gov_sim.experiments import evaluate_controller, instantiate_baseline, make_vec_env, prepare_output_dir
from gov_sim.utils.device import device_runtime_info
from gov_sim.utils.io import write_json
from gov_sim.utils.plotting import save_line_plot


def load_controller(model_path: str | None, config: dict[str, Any], baseline_name: str | None = None) -> tuple[Any, bool]:
    """加载 RL 模型或 baseline。"""
    if baseline_name is not None:
        return instantiate_baseline(baseline_name, config), True
    if model_path is None:
        raise ValueError("model_path is required when baseline_name is not provided")
    env = make_vec_env(config)
    try:
        model = MaskablePPOLagrangian.load(model_path, env=env)
    except Exception:
        try:
            model = MaskablePPO.load(model_path, env=env)
        except Exception:
            model = PPO.load(model_path, env=env)
    return model, False


def run_evaluation(config: dict[str, Any], model_path: str | None = None, baseline_name: str | None = None) -> dict[str, Any]:
    """执行评估并导出 step 级日志与汇总。"""
    output_dir = prepare_output_dir(config, "eval")
    controller, is_baseline = load_controller(model_path=model_path, config=config, baseline_name=baseline_name)
    eval_df, summary = evaluate_controller(
        controller=controller,
        config=config,
        episodes=int(config["eval"]["episodes"]),
        deterministic=bool(config["eval"]["deterministic"]),
        is_baseline=is_baseline,
    )
    eval_df.to_csv(output_dir / "eval_log.csv", index=False)
    summary["device_runtime"] = device_runtime_info()
    write_json(output_dir / "eval_summary.json", summary)
    if not eval_df.empty:
        save_line_plot(eval_df.index, {"latency": eval_df["L_bar_e"]}, output_dir / "latency_curve.png", "Latency", "Step", "Latency (ms)")
        save_line_plot(
            eval_df.index,
            {
                "m": eval_df["m_e"],
                "b/32": eval_df["b_e"] / 32.0,
                "tau": eval_df["tau_e"],
                "theta*100": eval_df["theta_e"] * 100.0,
            },
            output_dir / "action_trajectory.png",
            "Action Trajectory",
            "Step",
            "Scaled value",
        )
    return {"output_dir": str(output_dir), "summary": summary}
