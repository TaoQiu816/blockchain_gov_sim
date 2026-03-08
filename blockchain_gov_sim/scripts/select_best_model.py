"""从正式训练输出中筛选最佳模型。

筛选规则：
1. 先按 unsafe_rate 升序；
2. 再按 TPS 降序；
3. 再按 mean_latency 升序。

这比只看 reward 更符合第四章最终评价目标。
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def load_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", required=True, help="例如 outputs/formal/train")
    parser.add_argument("--prefix", required=True, help="例如 hard 或 moderate")
    args = parser.parse_args()

    train_root = Path(args.train_root)
    candidates = sorted(train_root.glob(f"{args.prefix}_seed*/train_summary.json"))
    if not candidates:
        raise FileNotFoundError(f"No train_summary.json found under {train_root} for prefix={args.prefix}")

    scored: list[tuple[float, float, float, Path, dict]] = []
    for summary_path in candidates:
        payload = load_summary(summary_path)
        eval_summary = payload.get("post_train_eval", {})
        scored.append(
            (
                float(eval_summary.get("unsafe_rate", 1e9)),
                -float(eval_summary.get("tps", 0.0)),
                float(eval_summary.get("mean_latency", 1e9)),
                summary_path.parent,
                payload,
            )
        )
    scored.sort(key=lambda item: (item[0], item[1], item[2]))
    _, _, _, best_dir, payload = scored[0]
    output_dir = train_root / f"{args.prefix}_best"
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_dir / "model.zip", output_dir / "model.zip")
    shutil.copy2(best_dir / "train_summary.json", output_dir / "train_summary.json")
    with (output_dir / "selection.json").open("w", encoding="utf-8") as file:
        json.dump({"selected_from": str(best_dir), "summary": payload}, file, indent=2, ensure_ascii=False)
    print(f"selected_best={best_dir}", flush=True)


if __name__ == "__main__":
    main()
