"""正式实验 runner。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from gov_sim.experiments.benchmark_runner import DEFAULT_METHODS, run_benchmark
from gov_sim.utils.io import deep_update, ensure_dir, load_config, write_json


SCENARIO_NAMES = ["normal", "load_shock", "high_rtt_burst", "churn_burst", "malicious_burst"]


def _scale_arrivals(config: dict[str, Any], num_rsus: int) -> dict[str, Any]:
    ratio = float(num_rsus) / 27.0
    override = {"env": {"num_rsus": int(num_rsus)}}
    for section in ("stable", "step"):
        if section in config.get("scenario", {}) and "lambda" in config["scenario"][section]:
            override.setdefault("scenario", {}).setdefault(section, {})["lambda"] = int(round(float(config["scenario"][section]["lambda"]) * ratio))
    if "mmpp" in config.get("scenario", {}):
        override.setdefault("scenario", {}).setdefault("mmpp", {})["lambdas"] = [
            int(round(float(value) * ratio)) for value in config["scenario"]["mmpp"].get("lambdas", [])
        ]
    return override


def _scenario_override(name: str) -> dict[str, Any]:
    if name == "normal":
        return {"scenario": {"training_mix": {"enabled": False}}}
    if name == "load_shock":
        return {"scenario": {"training_mix": {"enabled": True, "profiles": {"load_shock": {"weight": 1.0, "low_lambda_scale": 0.72, "high_lambda_scale": 2.5, "burst_start_frac": 0.45}}}}}
    if name == "high_rtt_burst":
        return {"scenario": {"training_mix": {"enabled": True, "profiles": {"high_rtt_burst": {"weight": 1.0, "burst_start_frac": 0.40, "burst_end_frac": 0.75, "rtt_min": 50.0, "rtt_max": 100.0}}}}}
    if name == "churn_burst":
        return {"scenario": {"training_mix": {"enabled": True, "profiles": {"churn_burst": {"weight": 1.0, "burst_start_frac": 0.40, "burst_end_frac": 0.75, "p_off": 0.22, "p_on": 0.08}}}}}
    if name == "malicious_burst":
        return {"scenario": {"training_mix": {"enabled": True, "profiles": {"malicious_burst": {"weight": 1.0, "burst_start_frac": 0.45, "burst_end_frac": 0.85, "extra_malicious_ratio": 0.15}}}}}
    raise ValueError(f"Unsupported scenario name: {name}")


def _intensity_override(rtt_intensity: str, churn_intensity: str) -> dict[str, Any]:
    rtt_table = {
        "moderate": {"rtt_min": 20.0, "rtt_max": 55.0},
        "hard": {"rtt_min": 30.0, "rtt_max": 75.0},
    }
    churn_table = {
        "moderate": {"p_off": 0.10, "p_on": 0.20},
        "hard": {"p_off": 0.16, "p_on": 0.12},
    }
    return {"scenario": {"network": {**rtt_table[rtt_intensity], **churn_table[churn_intensity]}}}


def run_formal(
    config: dict[str, Any],
    checkpoint_path: str | Path | None,
    methods: list[str] | None = None,
    scenarios: list[str] | None = None,
    num_rsus_list: list[int] | None = None,
    malicious_ratios: list[float] | None = None,
    rtt_intensities: list[str] | None = None,
    churn_intensities: list[str] | None = None,
    episodes: int | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    methods = list(DEFAULT_METHODS if methods is None else methods)
    scenarios = list(["normal"] if scenarios is None else scenarios)
    num_rsus_list = list([int(config["env"]["num_rsus"])] if num_rsus_list is None else num_rsus_list)
    malicious_ratios = list([float(config["env"]["malicious_ratio"])] if malicious_ratios is None else malicious_ratios)
    rtt_intensities = list(["moderate"] if rtt_intensities is None else rtt_intensities)
    churn_intensities = list(["moderate"] if churn_intensities is None else churn_intensities)
    output = ensure_dir(output_dir or Path(config.get("output_root", "outputs")) / "formal" / str(config.get("run_name", "default_run")))

    rows: list[dict[str, Any]] = []
    for scenario_name in scenarios:
        for num_rsus in num_rsus_list:
            for malicious_ratio in malicious_ratios:
                for rtt_intensity in rtt_intensities:
                    for churn_intensity in churn_intensities:
                        scenario_cfg = deep_update(config, _scale_arrivals(config, int(num_rsus)))
                        scenario_cfg = deep_update(scenario_cfg, _scenario_override(scenario_name))
                        scenario_cfg = deep_update(scenario_cfg, _intensity_override(rtt_intensity, churn_intensity))
                        scenario_cfg = deep_update(scenario_cfg, {"env": {"malicious_ratio": float(malicious_ratio)}})
                        tag = f"{scenario_name}_n{num_rsus}_pm{str(malicious_ratio).replace('.', 'p')}_{rtt_intensity}_{churn_intensity}"
                        result = run_benchmark(
                            config=scenario_cfg,
                            checkpoint_path=checkpoint_path,
                            methods=methods,
                            episodes=episodes,
                            output_dir=output / tag,
                        )
                        for row in result["rows"]:
                            row["scenario_name"] = scenario_name
                            row["num_rsus"] = int(num_rsus)
                            row["malicious_ratio"] = float(malicious_ratio)
                            row["rtt_intensity"] = rtt_intensity
                            row["churn_intensity"] = churn_intensity
                            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(output / "formal_table.csv", index=False)
    write_json(output / "formal_summary.json", {"rows": rows})
    return {"output_dir": str(output), "rows": rows}


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--scenarios", nargs="*", default=["normal"])
    args = parser.parse_args()

    config = load_config(args.config)
    run_formal(
        config=config,
        checkpoint_path=args.checkpoint,
        scenarios=args.scenarios,
        episodes=int(args.episodes),
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    _main()
