"""消融实验 runner。"""

from __future__ import annotations

import pandas as pd

from gov_sim.experiments import apply_override, build_model, build_plain_ppo, evaluate_controller, make_vec_env, prepare_output_dir
from gov_sim.utils.io import write_json
from gov_sim.utils.plotting import save_bar_plot


def run_ablation(config: dict[str, Any]) -> dict[str, Any]:
    """执行消融实验并导出比较表。"""
    output_dir = prepare_output_dir(config, "ablation")
    ablations = {
        "no_context_gate": {"reputation": {"use_context_gate": False}},
        "no_penalties": {"reputation": {"use_penalties": False}},
        "topk_committee": {"env": {"committee_method": "topk"}},
        "plain_ppo": {"env": {"mask_illegal_actions": False}},
        "no_action_mask": {"env": {"mask_illegal_actions": False}},
    }
    rows: list[dict[str, Any]] = []
    for name, override in ablations.items():
        experiment_cfg = apply_override(config, override)
        env = make_vec_env(experiment_cfg)
        if name == "plain_ppo":
            controller = build_plain_ppo(experiment_cfg, env)
            controller.learn(total_timesteps=int(experiment_cfg["agent"]["n_steps"]) * 2, progress_bar=False)
            is_baseline = False
        else:
            controller = build_model(experiment_cfg, env, use_lagrangian=True)
            controller.learn(total_timesteps=int(experiment_cfg["agent"]["n_steps"]) * 2, progress_bar=False)
            is_baseline = False
        _, summary = evaluate_controller(
            controller=controller,
            config=experiment_cfg,
            episodes=int(experiment_cfg["eval"]["episodes"]),
            deterministic=True,
            is_baseline=is_baseline,
        )
        summary["ablation"] = name
        rows.append(summary)
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "ablation_log.csv", index=False)
    write_json(output_dir / "ablation_summary.json", {"rows": rows})
    if not df.empty:
        save_bar_plot(df["ablation"].tolist(), df["unsafe_rate"].tolist(), output_dir / "ablation_unsafe_bar.png", "Ablation Unsafe Rate", "Unsafe rate")
        save_bar_plot(df["ablation"].tolist(), df["tps"].tolist(), output_dir / "ablation_tps_bar.png", "Ablation TPS", "TPS")
    return {"output_dir": str(output_dir), "rows": rows}
