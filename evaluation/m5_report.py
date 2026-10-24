"""Aggregate M5 robustness sweeps -> results/m5_robustness.md + results/m5_robustness.png"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "training" / "runs"
OUT = ROOT / "results"

GROUPS = [
    ("hover_v5", "m5_hover_v5_", ["alt_err_rms_m", "tilt_deg_p95", "pos_drift_m_mean", "hover_power_kw"]),
    ("transition_v3", "m5_transition_v3_", ["liftoff_rate", "landed_rate", "impact_mps_p95", "unassisted_stand_rate"]),
    ("stand_v2", "m5_stand_v2_", ["sway_cm_mean", "max_tilt_deg_p95"]),
    ("walk_v4", "m5_walk_v4_", ["speed_mps_mean", "pod_clearance_mm_mean", "slip_fraction_of_travel"]),
    ("stand_v2 pushes (no fans)", "stand_v2_push", ["sway_cm_mean"]),
    ("stand_v3 pushes (thrust-assist)", "stand_v3_thrustassist_push", ["sway_cm_mean"]),
]


def main():
    lines = ["# M5 robustness sweeps", "",
             "One perturbation at a time, layered on top of the standard domain randomization. "
             "100 trials each on held-out seed 12345.", ""]
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
    for gi, (name, prefix, keys) in enumerate(GROUPS):
        rows = []
        for d in sorted(RUNS.glob(prefix + "*")):
            f = d / "eval.json"
            if not f.exists():
                continue
            e = json.loads(f.read_text())
            rows.append((d.name[len(prefix):], e))
        if not rows:
            continue
        lines += [f"## {name}", "", "| condition | success | no-fall | " + " | ".join(keys) + " |",
                  "|---|---|---|" + "---|" * len(keys)]
        for cond, e in rows:
            vals = " | ".join(f"{e.get(k, float('nan')):.3g}" for k in keys)
            lines.append(f"| {cond} | {e['success_rate']:.2f} | {e['no_fall_rate']:.2f} | {vals} |")
        lines.append("")
        ax = axs[min(gi, 2)]
        ax.bar([f"{name.split()[0][:6]}:{c}" for c, _ in rows], [e["success_rate"] for _, e in rows], alpha=0.8)
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=75, labelsize=7)
        ax.set_ylabel("success rate")
    for ax, t in zip(axs, ("hover_v5", "transition_v3", "stand / walk / pushes")):
        ax.set_title(t)
    fig.tight_layout()
    fig.savefig(OUT / "m5_robustness.png", dpi=120)
    (OUT / "m5_robustness.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
