"""Curated figures for the README / portfolio -> docs/assets/fig_*.png

python evaluation/portfolio_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "training" / "runs"
OUT = ROOT / "docs" / "assets"
C = {"good": "#2e7d32", "bad": "#c62828", "neutral": "#5c6bc0", "grey": "#9e9e9e", "amber": "#f9a825"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def ev(run, key="success_rate", f="eval.json"):
    return json.loads((RUNS / run / f).read_text())[key]


def curve(run):
    d = pd.read_csv(RUNS / run / "progress.csv")
    return d["step"] / 1e6, d["eval/episode_reward"]


def learning_curves():
    panels = [("Stand (M1)", ["stand_v2"]), ("Walk (M2)", ["walk_v4"]),
              ("Hover (M3), 3 seeds + wide-DR", ["hover_v5", "hover_v5_s1", "hover_v5_s2", "hover_v7_wideDR_pos"]),
              ("Transition (M4), warm-started", ["transition_v2", "transition_v3", "transition_v5_wideDR"]),
              ("Maneuver (stretch)", ["maneuver_v1"]), ("Fan-failure landing", ["failsafe_v1", "failsafe_v2_wideDR"])]
    fig, axs = plt.subplots(2, 3, figsize=(14, 7))
    for ax, (title, runs) in zip(axs.flat, panels):
        for r in runs:
            x, y = curve(r)
            ax.plot(x, y, marker="o", ms=3, label=r)
        ax.set_title(title)
        ax.set_xlabel("environment steps [M]")
        ax.set_ylabel("eval episode reward")
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle("Learning curves of the reference policies (PPO, 8192 parallel MJX-Warp envs, deterministic eval; "
                 "reward scales differ between versions)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_learning_curves.png", dpi=110)


def reset_bug():
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for r, col, lab in (("hover_v2", C["bad"], "hover_v2: leaky auto-reset"),
                        ("hover_v3", C["amber"], "hover_v3: leaky reset + PPO tuning"),
                        ("hover_v5", C["good"], "hover_v5: full reset (fixed)")):
        x, y = curve(r)
        ax.plot(x, y, marker="o", ms=4, color=col, label=lab)
    ax.set_xlabel("environment steps [M]")
    ax.set_ylabel("eval episode reward")
    ax.set_title("The 'mysterious collapse': battery SoC & timers leaked across auto-resets")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_reset_bug.png", dpi=110)


def walk_iterations():
    # walk_v1 gait numbers come from the kinematic gait analysis (docs/experiments.md); later versions from eval.json
    names = ["v1\nbaseline", "v2\n+gait clock\n+slip", "v3\nfine-tune\n+swing", "v4\nscratch\n-pod_flat"]
    clear = [7.5] + [ev(r, "pod_clearance_mm_mean") for r in ("walk_v2", "walk_v3", "walk_v4")]
    slip = [45.0] + [100 * ev(r, "slip_fraction_of_travel") for r in ("walk_v2", "walk_v3", "walk_v4")]
    succ = [97.0] + [100 * ev(r) for r in ("walk_v2", "walk_v3", "walk_v4")]
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, vals, t, good_hi in ((axs[0], succ, "reach 2 m [%]", True), (axs[1], clear, "pod step clearance [mm]", True),
                                 (axs[2], slip, "slip while in contact [% of travel]", False)):
        cols = [C["good"] if (i == 3) else C["grey"] for i in range(4)]
        ax.bar(names, vals, color=cols)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.0f}", ha="center", va="bottom")
        ax.set_title(t)
    fig.suptitle("Walking: every version 'passed' the distance metric. Gait quality told the real story")
    fig.tight_layout()
    fig.savefig(OUT / "fig_walk_iterations.png", dpi=110)


def transition_iterations():
    names = ["classical\nFSM", "v1", "v2", "v3\n+ground-thrust\npenalty", "v4\n+impact x4", "v5\n+wide DR"]
    contact_only = [0, 74, 96, None, None, None]
    strict = [0, None, 0, 100 * ev("transition_v3"), 100 * ev("transition_v4"), 100 * ev("transition_v5_wideDR")]
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(names))
    ax.bar(x - 0.18, [v or 0 for v in contact_only], 0.36, color=C["grey"], label="contact-only scoring")
    ax.bar(x + 0.18, [v or 0 for v in strict], 0.36, color=C["good"], label="strict: fans idle in final stand")
    for i, (a, b) in enumerate(zip(contact_only, strict)):
        if a is not None:
            ax.text(i - 0.18, a + 1, f"{a:.0f}", ha="center", fontsize=8)
        if b is not None:
            ax.text(i + 0.18, b + 1, f"{b:.0f}", ha="center", fontsize=8)
    ax.axhline(80, ls="--", color="k", lw=0.8)
    ax.text(5.45, 81, "spec 80%", fontsize=8)
    ax.set_xticks(x, names)
    ax.set_ylabel("full-cycle success [%]")
    ax.set_title("Ground-to-flight-to-ground: v2 'stood' with 30% of its weight on the fans")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_transition_iterations.png", dpi=110)


def rl_vs_classical():
    tasks = ["stand", "hover", "transition", "fan failure"]
    cls = [ev("stand_classical", f="eval_classical.json"), ev("hover_classical", f="eval_classical.json"),
           ev("transition_classical", f="eval_classical.json"), ev("transition_v4_failsafe_baseline")]
    rl = [ev("stand_v2"), ev("hover_v7_wideDR_pos"), ev("transition_v5_wideDR"), ev("failsafe_v2_wideDR")]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = np.arange(4)
    ax.bar(x - 0.2, np.array(cls) * 100, 0.4, color=C["grey"], label="classical baseline / untrained")
    ax.bar(x + 0.2, np.array(rl) * 100, 0.4, color=C["neutral"], label="PPO policy")
    for i, (a, b) in enumerate(zip(cls, rl)):
        ax.text(i - 0.2, a * 100 + 1, f"{a * 100:.0f}", ha="center", fontsize=8)
        ax.text(i + 0.2, b * 100 + 1, f"{b * 100:.0f}", ha="center", fontsize=8)
    ax.set_xticks(x, tasks)
    ax.set_ylabel("success [%] (100 trials)")
    ax.set_ylim(0, 125)
    ax.set_title("Same eval suite: classical controllers vs learned policies")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center")
    fig.tight_layout()
    fig.savefig(OUT / "fig_rl_vs_classical.png", dpi=110)


def robustness():
    conds = [("nominal", None), ("thrust 80%", "thrust80"), ("mass +20%", "mass120"), ("one fan 50%", "fan0_50")]
    pols = [("hover_v5 (DR x1)", "hover_v5", "m5_hover_v5_"), ("hover_v6 (DR x2)", "hover_v6_wideDR", "m5_hover_v6_"),
            ("hover_v7 (DR x2 + pos)", "hover_v7_wideDR_pos", "m5_hover_v7_")]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = np.arange(len(conds))
    for j, (lab, base, pre) in enumerate(pols):
        vals = [ev(base) if c is None else ev(pre + c) for _, c in conds]
        ax.bar(x + (j - 1) * 0.27, np.array(vals) * 100, 0.27, label=lab,
               color=[C["grey"], C["amber"], C["good"]][j])
    ax.set_xticks(x, [c for c, _ in conds])
    ax.set_ylabel("hover success [%]")
    ax.set_title("Domain-randomization width: robustness bought with policy, not hardware")
    ax.set_ylim(0, 125)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center")
    fig.tight_layout()
    fig.savefig(OUT / "fig_robustness.png", dpi=110)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (learning_curves, reset_bug, walk_iterations, transition_iterations, rl_vs_classical, robustness):
        f()
        print(f.__name__)


def parts_tradespace():
    import yaml
    cat = yaml.safe_load((ROOT / "configs" / "hardware.yaml").read_text())
    v = lambda x: x["value"] if isinstance(x, dict) else x
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axs[0]
    for key, e in cat["edf"].items():
        vt, a, t, p = e["test_points"][0]
        inst = t * cat["rules"]["install_eff"]
        verified = e["status"] == "verified"
        chosen = key == "jp90_12s"
        ax.scatter(v(e["od_lip_mm"]), inst, s=v(e["mass_g"]) * 0.8, alpha=0.75,
                   c=C["good"] if chosen else (C["neutral"] if verified else C["grey"]), edgecolor="k")
        ax.annotate(f"{key}\n{e['cells']}S, {v(e['mass_g'])} g, {t * 1000 / p:.2f} g/W", (v(e["od_lip_mm"]), inst),
                    textcoords="offset points", xytext=(8, -6), fontsize=7)
    ax.set_xlabel("housing OD incl. lip [mm]  (sets the pod size, hence the robot scale)")
    ax.set_ylabel("installed thrust per fan [kg]  (bench x 0.85)")
    ax.set_title("EDF candidates (size = unit mass; grey = no published test data)")
    ax.set_xlim(70, 135)
    ax = axs[1]
    for key, s in cat["servo"].items():
        st = s["stall_nm"]
        cont = st * cat["rules"]["servo_cont_frac"]
        ax.plot([s["mass_g"], s["mass_g"]], [cont, st], color=C["grey"], lw=2)
        ax.scatter([s["mass_g"]], [st], c=C["neutral"], s=50, zorder=3)
        ax.scatter([s["mass_g"]], [cont], c=C["amber"], s=50, zorder=3)
        ax.annotate(f"{key}\n{s['noload_rpm']} rpm", (s["mass_g"], st), textcoords="offset points", xytext=(6, 2), fontsize=7)
    ax.scatter([], [], c=C["neutral"], label="stall (momentary)")
    ax.scatter([], [], c=C["amber"], label="design continuous (35% stall)")
    ax.set_xlabel("mass [g]")
    ax.set_ylabel("torque [N m]")
    ax.set_title("Servo candidates (auto-selected per joint, 1x or 2x ganged)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_parts_tradespace.png", dpi=110)


def gate_matrix():
    res = json.loads((ROOT / "results" / "m0" / "summary.json").read_text())
    checks = [k for k in res[0]["checks"] if "info" not in k]
    M = np.array([[r["checks"][k] for k in checks] for r in res], float)
    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.imshow(M, cmap=matplotlib.colors.ListedColormap([C["bad"], C["good"]]), aspect="auto", vmin=0, vmax=1)
    ax.set_yticks(range(len(res)), [f"{r['name']}  ({r['mass_kg']:.1f} kg, T/W {r['thrust_margin']:.2f})" for r in res], fontsize=8)
    ax.set_xticks(range(len(checks)), [c.replace("_", " ") for c in checks], rotation=40, ha="right", fontsize=8)
    for i, r in enumerate(res):
        ax.text(len(checks) - 0.5 + 0.2, i, "PASS" if r["passed"] else "fail", va="center", fontsize=8,
                color=C["good"] if r["passed"] else C["bad"], fontweight="bold")
    ax.set_title(f"M0 feasibility gate: {len(res)} candidate designs x {len(checks)} smoke checks (green = pass; the 1.8 target is aspirational)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_gate_matrix.png", dpi=110)


def main_all():
    main()
    parts_tradespace()
    gate_matrix()


if __name__ == "__main__":
    main_all()
