"""训练入口 runner。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from gov_sim.agent.callbacks import TrainLoggingCallback
from gov_sim.experiments import build_model, evaluate_controller, make_vec_env, prepare_output_dir
from gov_sim.utils.device import device_runtime_info
from gov_sim.utils.io import write_json
from gov_sim.utils.plotting import save_line_plot


def run_training(config: dict[str, Any]) -> dict[str, Any]:
    """执行训练并导出训练日志、后评估结果与曲线。"""
    output_dir = prepare_output_dir(config, "train")
    env = make_vec_env(config)
    model = build_model(config=config, env=env, use_lagrangian=True)
    callback = TrainLoggingCallback(log_path=output_dir / "train_log.csv")
    model.learn(total_timesteps=int(config["agent"]["total_timesteps"]), callback=callback, progress_bar=False)
    model.save(str(output_dir / "model"))

    log_df = pd.read_csv(output_dir / "train_log.csv") if (output_dir / "train_log.csv").exists() else pd.DataFrame()
    eval_df, eval_summary = evaluate_controller(
        controller=model,
        config=config,
        episodes=int(config["eval"]["episodes"]),
        deterministic=bool(config["eval"]["deterministic"]),
        is_baseline=False,
    )
    eval_df.to_csv(output_dir / "post_train_eval.csv", index=False)
    summary = {
        "train_episodes": int(len(log_df)),
        "ep_reward_mean": float(log_df["episode_reward"].mean()) if not log_df.empty else 0.0,
        "ep_cost_mean": float(log_df["episode_cost"].mean()) if not log_df.empty else 0.0,
        "lambda_final": float(getattr(model, "lambda_value", 0.0)),
        "device": str(getattr(model, "device", "unknown")),
        "device_runtime": device_runtime_info(),
        "post_train_eval": eval_summary,
    }
    write_json(output_dir / "train_summary.json", summary)
    if not log_df.empty:
        x = range(len(log_df))
        save_line_plot(x, {"reward": log_df["episode_reward"]}, output_dir / "reward_curve.png", "Reward Curve", "Episode", "Reward")
        save_line_plot(x, {"cost": log_df["episode_cost"]}, output_dir / "cost_curve.png", "Cost Curve", "Episode", "Cost")
        save_line_plot(x, {"unsafe": log_df["unsafe_rate"]}, output_dir / "unsafe_curve.png", "Unsafe Curve", "Episode", "Unsafe")
    return {"output_dir": str(output_dir), "summary": summary}
