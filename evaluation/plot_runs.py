"""Training curves + eval summary table for all runs.

python evaluation/plot_runs.py  -> results/training_curves.png, results/eval_table.md
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "training" / "runs"
OUT = ROOT / "results"


def main():
    OUT.mkdir(exist_ok=True)
    runs = sorted(p for p in RUNS.iterdir() if (p / "progress.csv").exists())
    tasks = sorted({json.loads((p / "config.json").read_text())["task"] for p in runs})
    fig, axs = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 4), squeeze=False)
    for ax, task in zip(axs[0], tasks):
        for p in runs:
            cfg = json.loads((p / "config.json").read_text())
            if cfg["task"] != task:
                continue
            df = pd.read_csv(p / "progress.csv")
            if "eval/episode_reward" in df:
                ax.plot(df["step"] / 1e6, df["eval/episode_reward"], label=p.name)
        ax.set_title(task)
        ax.set_xlabel("env steps [M]")
        ax.set_ylabel("eval episode reward")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "training_curves.png", dpi=110)
    rows = []
    for p in sorted(RUNS.iterdir()):
        for f in p.glob("eval*.json"):
            e = json.loads(f.read_text())
            rows.append({"run": p.name, "eval": f.stem, "task": e["task"], "controller": e["controller"],
                         "success": e.get("success_rate"), "no_fall": e.get("no_fall_rate"),
                         **{k: round(v, 3) for k, v in e.items() if isinstance(v, float) and k not in
                            ("success_rate", "no_fall_rate", "dt")}})
    if rows:
        df = pd.DataFrame(rows).sort_values(["task", "run"])
        (OUT / "eval_table.md").write_text(df.to_markdown(index=False))
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
