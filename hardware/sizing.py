"""M0 feasibility gate: budgets, margins, allocation and Pareto across candidate scales.

python hardware/sizing.py            -> results/m0/{summary.json, candidates.csv, bom_*.csv, *.png}
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.design import G, load_candidates, load_catalog, nozzle_exit_height, val  # noqa: E402
from models.build_mjcf import generate, hover_pose, pose_qpos, standing_pose  # noqa: E402
from controllers.allocation import Allocator  # noqa: E402

OUT = ROOT / "results" / "m0"
REACTION_RESIDUAL = 0.25     # ASSUMPTION: fraction of motor torque not cancelled by EDF stator vanes
PACK_R_PER_CELL = 0.0025     # ohm per series cell (assumption, 6 Ah high-C pack)
SPEC_HEAT_PACK = 1000.0      # J/(kg K), LiPo pack


def thrust_voltage_exponent(tp):
    if len(tp) < 2:
        return 1.7
    v = np.log([p[0] for p in tp])
    t = np.log([p[2] for p in tp])
    return float(np.polyfit(v, t, 1)[0])


def static_stand_test(model, q0, seconds=3.0):
    d = mujoco.MjData(model)
    d.qpos[:] = q0
    act_j = [model.actuator_trnid[a][0] for a in range(model.nu)]
    for a in range(model.nu):
        if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT:
            d.ctrl[a] = q0[model.jnt_qposadr[act_j[a]]]
    mujoco.mj_forward(model, d)
    z0 = d.qpos[2]
    peak_ratio = np.zeros(model.nu)
    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, d)
        for a in range(model.nu):
            fr = model.actuator_forcerange[a][1]
            if fr > 0 and model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT:
                peak_ratio[a] = max(peak_ratio[a], abs(d.actuator_force[a]) / fr)
    up = d.xmat[1].reshape(3, 3)[2, 2]
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in range(model.nu)]
    worst = int(np.argmax(peak_ratio))
    return dict(stood=bool(d.qpos[2] > z0 - 0.05 and up > 0.95), dz=float(d.qpos[2] - z0), torso_up=float(up),
                worst_actuator=names[worst], worst_force_ratio=float(peak_ratio[worst]))


def support_margin(model, q):
    d = mujoco.MjData(model)
    d.qpos[:] = q
    mujoco.mj_forward(model, d)
    com = d.subtree_com[1]
    # support polygon = convex hull of the shin-pod rim pads (outer pad edges)
    pts = []
    for g in range(model.ngeom):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if "_pad_" in nm:
            R = d.geom_xmat[g].reshape(3, 3)
            hx, hy, _ = model.geom_size[g]
            for sx in (-1, 1):
                for sy in (-1, 1):
                    pts.append((d.geom_xpos[g] + R @ np.array([sx * hx, sy * hy, 0]))[:2])
    pts = np.array(pts)
    from scipy.spatial import ConvexHull
    hull = ConvexHull(pts)
    # signed distance of COM to each hull edge (positive inside)
    margins = [-(eq[:2] @ com[:2] + eq[2]) for eq in hull.equations]
    xs, ys = pts[:, 0], pts[:, 1]
    return dict(com=com.tolist(), margin_min=float(min(margins)),
                margin_fore=float(xs.max() - com[0]), margin_aft=float(com[0] - xs.min()),
                margin_lat=float(min(ys.max() - com[1], com[1] - ys.min())),
                pod_base_diameter=float(2 * model.site_size[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "l_sole")][0] / 1.05))


def inertia_about_com(model, q):
    d = mujoco.MjData(model)
    d.qpos[:] = q
    mujoco.mj_forward(model, d)
    c = d.subtree_com[1]
    I = np.zeros((3, 3))
    for b in range(1, model.nbody):
        m = model.body_mass[b]
        R = d.ximat[b].reshape(3, 3)
        Ib = R @ np.diag(model.body_inertia[b]) @ R.T
        r = d.xipos[b] - c
        I += Ib + m * (r @ r * np.eye(3) - np.outer(r, r))
    return I


def tune_hover_pose(model, d, T_use):
    """Grid-search shoulder/hip pitch so trim thrusts are balanced (max throttle headroom)
    and gimbals sit near centre. Returns pose overrides and the chosen angles in degrees."""
    best = None
    for sp in np.radians(np.arange(-20, 1, 4)):
        for hp in np.radians(np.arange(-24, 13, 3)):
            q = pose_qpos(model, hover_pose(d, sp, hp), ground=False, height=0.5)
            A = Allocator(model, q, T_use)
            u, res = A.trim()
            if np.abs(res).max() > 1e-3 * A.mass * G:
                continue
            thr = u[:4] / T_use
            gim = np.abs(u[4:]) / np.radians(d["vec_range"])
            cost = thr.max() + 0.3 * gim.max()
            if best is None or cost < best[0]:
                best = (cost, sp, hp, thr.max(), gim.max())
    if best is None:
        return {}, None
    _, sp, hp, tmax, gmax = best
    return ({"hover_shoulder_pitch": sp, "hover_hip_pitch": hp},
            dict(shoulder_pitch=float(np.degrees(sp)), hip_pitch=float(np.degrees(hp)), max_throttle=float(tmax), max_gimbal_frac=float(gmax)))


def analyze(name, cand, cat):
    rules = cat["rules"]
    path, d = generate(name)
    model = mujoco.MjModel.from_xml_path(str(path))
    m = float(model.body_subtreemass[1])
    mg = m * G
    edf = d["edf"]
    v_ref, a_ref, t_ref, p_ref = d["test_ref"]
    cells = edf["cells"]
    n_ser, n_par = cand["pack_series"], cand["pack_parallel"]
    bat = d["bat"]
    pack_cells = bat["cells"] * n_ser
    assert pack_cells == cells, f"{name}: pack {pack_cells}S vs EDF {cells}S"
    eta = rules["install_eff"]
    vexp = thrust_voltage_exponent(edf["test_points"])
    T_inst = d["T_inst_max"]
    T_end = T_inst * (rules["cell_v_end"] / rules["cell_v_loaded"]) ** vexp
    T_worst = T_end * rules["install_eff_band"][0] / eta

    # ---- battery current limit caps the usable per-fan thrust
    V_pack = pack_cells * rules["cell_v_loaded"]
    I_pack_cont = bat["cap_ah"] * bat["c_cont"] * cand["pack_parallel"]
    P_fan_avail = min(p_ref, (V_pack * I_pack_cont - rules["aux_power_w"]) / 4)
    T_fan_avail = t_ref * G * (P_fan_avail / p_ref) ** (1 / rules["power_exponent"]) * eta
    T_use = min(T_inst, T_fan_avail)

    q_stand = pose_qpos(model, standing_pose(d))
    pose, tune = tune_hover_pose(model, d, T_use)
    d.update(pose)
    q_hover = pose_qpos(model, hover_pose(d), ground=False, height=0.5)
    alloc = Allocator(model, q_hover, T_use).analyze(d["H"])
    T_trim = np.array(alloc["u_trim"][:4])

    # ---- power / endurance from measured fan curve P ~ T^1.5 (bench-equivalent thrust)
    T_bench_equiv = T_trim / eta
    P_fans = p_ref * (T_bench_equiv / (t_ref * G)) ** rules["power_exponent"]
    P_hover = float(P_fans.sum() + rules["aux_power_w"])
    I_hover = P_hover / V_pack
    I_fan_max = a_ref
    I_max = 4 * I_fan_max + rules["aux_power_w"] / V_pack
    E_use = pack_cells * 3.7 * bat["cap_ah"] * n_par * rules["usable_frac"]
    t_hover_s = E_use / P_hover * 3600
    R_pack = PACK_R_PER_CELL * pack_cells / n_par
    heat_w = I_hover ** 2 * R_pack
    dT_pack = heat_w * t_hover_s / (bat["mass_g"] / 1000 * n_ser * n_par * SPEC_HEAT_PACK)

    # ---- yaw authority vs reaction-torque uncertainty
    omega = d["gimbal_req"]["omega_hover"]
    Q_motor = (P_fans * 0.85) / omega
    yaw_disturb = float(REACTION_RESIDUAL * Q_motor.sum())   # all units same rotation (no reverse JP units)
    yaw_auth = min(abs(alloc["attainable"]["tau_z"][0]), abs(alloc["attainable"]["tau_z"][1]))

    # ---- servo margins
    servo_rows = []
    S = cat["servo"]
    for j, ch in d["servo_choice"].items():
        s = S[ch["key"]]
        cont, peak, spd = d["torque_req"][j]
        avail_c = ch["n"] * s["stall_nm"] * rules["servo_cont_frac"]
        avail_p = ch["n"] * s["stall_nm"] * rules["servo_peak_frac"]
        servo_rows.append(dict(joint=j, servo=ch["key"], n=ch["n"], req_cont=cont, avail_cont=avail_c,
                               req_peak=peak, avail_peak=avail_p, cont_margin=avail_c / cont, peak_margin=avail_p / peak,
                               ok=ch["ok"]))
    gch, greq = d["gimbal_choice"], d["gimbal_req"]
    gs = S[gch["key"]]
    servo_rows.append(dict(joint="gimbal", servo=gch["key"], n=gch["n"], req_cont=greq["cont_nm"],
                           avail_cont=gch["n"] * gs["stall_nm"] * rules["servo_cont_frac"], req_peak=greq["peak_nm"],
                           avail_peak=gch["n"] * gs["stall_nm"] * rules["servo_peak_frac"],
                           cont_margin=gch["n"] * gs["stall_nm"] * rules["servo_cont_frac"] / greq["cont_nm"],
                           peak_margin=gch["n"] * gs["stall_nm"] * rules["servo_peak_frac"] / greq["peak_nm"], ok=gch["ok"]))

    stand = static_stand_test(model, q_stand)
    sup = support_margin(model, q_stand)
    I_stand = inertia_about_com(model, q_stand)
    I_hov = inertia_about_com(model, q_hover)

    P = d["P"]
    H = d["H"]
    aesthetics = dict(pod_od_over_H=d["pod_od"] / H, pod_over_thigh=d["pod_od"] / (2 * P["thigh_r"] * H),
                      pod_over_upper_arm=d["pod_od"] / (2 * P["upper_arm_r"] * H),
                      leg_gimbal_range_deg=d["vec_range_leg"],
                      nozzle_exit_height_mm=nozzle_exit_height(d) * 1000,
                      exhaust_escape_area_ratio=4 * nozzle_exit_height(d) * 0.81 / (0.8 * val(d["edf"]["od_mm"]) / 1000),
                      pod_len_over_forearm=d["pod_len"] / (P["forearm"] * H),
                      shin_pod_gap_mm=(2 * P["hip_half"] * H - d["pod_od"]) * 1000)

    # ---- cost
    n_servo = sum(2 * c["n"] for c in d["servo_choice"].values())
    cost = 4 * val(edf["price_usd"]) + 4 * val(d["esc"]["price_usd"]) + n_ser * n_par * val(bat["price_usd"])
    cost += sum(2 * c["n"] * val(S[c["key"]]["price_usd"]) for c in d["servo_choice"].values())
    cost += 4 * gch["n"] * val(gs["price_usd"]) + 80 + 350   # FC/RX + filament, bearings, wire, connectors

    margin = 4 * T_use / mg
    checks = {
        "thrust_margin_nominal>=1.4": margin >= rules["min_margin"],
        "thrust_margin_target>=1.8": margin >= rules["target_margin"],
        "thrust_margin_worstcase>=1.15": 4 * T_worst / mg >= 1.15,
        "trim_feasible": float(np.abs(alloc["trim_residual"]).max()) < 1e-3 * mg,
        "allocation_rank6": alloc["rank"] == 6,
        "allocation_cond<100": alloc["cond"] < 100,
        "yaw_authority>=2x_reaction_uncert": yaw_auth >= 2 * yaw_disturb,
        "hover_endurance>=60s": t_hover_s >= 60,
        "pack_current_ok": I_hover <= I_pack_cont,
        "pack_current_fullthrottle_ok(info)": I_max <= I_pack_cont * 1.0,
        "esc_current_ok": I_fan_max <= d["esc"]["cont_a"],
        "esc_voltage_ok": cells <= val(d["esc"]["max_cells"]),
        "joint_servos_ok": all(r["ok"] for r in servo_rows),
        "static_stand_ok": stand["stood"] and sup["margin_min"] > 0.01,
        "pods_oversized(>=1.4x thigh)": aesthetics["pod_over_thigh"] >= 1.4,
        "shin_pods_clear(gap>=3mm)": aesthetics["shin_pod_gap_mm"] >= 3,
    }
    return dict(
        name=name, height_m=H, edf=edf["part"], edf_name=edf["name"], cells=cells, mass_kg=m,
        budget_kg={k: v for k, v in sorted(d["budget"].items(), key=lambda kv: -kv[1])},
        pod_od_mm=d["pod_od"] * 1000, pod_len_mm=d["pod_len"] * 1000,
        T_bench_max_N=d["T_bench_max"], T_inst_max_N=T_inst, T_inst_end_N=T_end, T_worst_N=T_worst,
        T_use_N=T_use, current_limited=bool(T_fan_avail < T_inst), hover_pose_deg=tune,
        thrust_margin=margin, thrust_margin_end=4 * min(T_end, T_use) / mg,
        thrust_margin_worst=4 * min(T_worst, T_use * rules["install_eff_band"][0] / eta) / mg,
        hover_throttle_frac=float(T_trim.mean() / T_inst), T_trim_N=T_trim.tolist(),
        P_hover_W=P_hover, I_hover_A=I_hover, I_fullthrottle_A=I_max, I_pack_cont_A=I_pack_cont,
        hover_endurance_s=t_hover_s, pack_temp_rise_K=dT_pack, g_per_W=float(mg / G * 1000 / P_hover),
        yaw_reaction_uncert_Nm=yaw_disturb, yaw_authority_Nm=yaw_auth, gyro_torque_Nm=greq["gyro_nm"],
        allocation=alloc, servos=servo_rows, stand=stand, support=sup,
        inertia_stand=I_stand.tolist(), inertia_hover=I_hov.tolist(), aesthetics=aesthetics,
        cost_usd=cost, checks={k: bool(v) for k, v in checks.items()},
        passed=bool(all(v for k, v in checks.items() if "target" not in k and "info" not in k)),
        model=str(path.relative_to(ROOT)))


def write_bom(res, cat, cand):
    d_edf = cat["edf"][cand["edf"]]
    esc = cat["esc"][cand["esc"]]
    bat = cat["battery"][cand["battery"]]
    rows = [
        ["EDF + motor", d_edf["name"], d_edf["part"], 4, val(d_edf["mass_g"]), val(d_edf["price_usd"]), d_edf["status"], d_edf["source"]],
        ["ESC", esc["name"], esc["part"], 4, esc["mass_g"], val(esc["price_usd"]), esc["status"], esc["source"]],
        ["Battery", bat["name"], "-", cand["pack_series"] * cand["pack_parallel"], bat["mass_g"], val(bat["price_usd"]), bat["status"], bat["source"]],
    ]
    for r in res["servos"]:
        s = cat["servo"][r["servo"]]
        qty = r["n"] * (4 if r["joint"] == "gimbal" else 2)
        rows.append([f"Servo {r['joint']}", r["servo"], r["servo"].upper(), qty, s["mass_g"], val(s["price_usd"]), s["status"], s["source"]])
    fc = cat["fc"]["matek_h743_slim_v3"]
    rows.append(["Flight controller", "Matek H743-SLIM V3", "H743-SLIM-V3", 1, fc["mass_g"], 75, fc["status"], fc["source"]])
    rows.append(["Avionics misc", "RX, 12V BEC, servo bus bridge", "-", 1, val(cat["fc"]["other_avionics_g"]), 80, "assumption", ""])
    rows.append(["Structure", "PETG shells 1.2mm, CF spars, bearings, fasteners", "-", 1,
                 round((res["budget_kg"].get("shell", 0) + res["budget_kg"].get("fastener", 0)) * 1000), 350, "assumption", ""])
    rows.append(["Wiring", "8/10 AWG silicone, AS150/XT90, servo bus", "-", 1, round(res["budget_kg"].get("wiring", 0) * 1000), 0, "assumption", ""])
    p = OUT / f"bom_{res['name']}.csv"
    with p.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["item", "description", "part", "qty", "unit_mass_g", "unit_price_usd", "status", "source"])
        w.writerows(rows)
    return p


def plots(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.2))
    ax = axs[0]
    for r in results:
        col = "#1a7f37" if r["passed"] else "#c62828"
        ax.scatter(r["thrust_margin"], r["hover_endurance_s"], s=r["height_m"] * 400, c=col, alpha=0.75, edgecolor="k")
        ax.annotate(f"{r['name']}\n{r['mass_kg']:.1f} kg, ${r['cost_usd']/1000:.1f}k", (r["thrust_margin"], r["hover_endurance_s"]),
                    textcoords="offset points", xytext=(8, -4), fontsize=8)
    ax.axvline(1.4, ls="--", c="grey")
    ax.axvline(1.8, ls=":", c="grey")
    ax.text(1.405, ax.get_ylim()[1] * 0.95, "min 1.4", fontsize=8, color="grey")
    ax.text(1.805, ax.get_ylim()[1] * 0.95, "target 1.8", fontsize=8, color="grey")
    ax.axhline(60, ls="--", c="grey")
    ax.set_xlabel("installed thrust / weight (nominal)")
    ax.set_ylabel("hover endurance [s] (75% usable energy)")
    ax.set_title("Candidate scales (marker size = height; green = gate pass)")
    ax = axs[1]
    cats = ["joint_servos", "edf", "battery", "shell", "esc", "gimbal_servos", "wiring", "fastener", "avionics"]
    bottom = np.zeros(len(results))
    for c in cats:
        vals = np.array([r["budget_kg"].get(c, 0) for r in results])
        ax.bar([r["name"].split("_")[0] for r in results], vals, bottom=bottom, label=c)
        bottom += vals
    for i, r in enumerate(results):
        ax.text(i, bottom[i] + 0.1, f"T={4 * r['T_use_N'] / G:.1f}kg", ha="center", fontsize=8)
    ax.set_ylabel("mass [kg]")
    ax.set_title("Integrated mass budget vs usable installed thrust (fan- or pack-current-limited)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "pareto_mass.png", dpi=130)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cat = load_catalog()
    cands = load_candidates()
    results = []
    only = sys.argv[1:]
    for name, cand in cands.items():
        if only and not any(name.startswith(o) for o in only):
            continue
        r = analyze(name, cand, cat)
        results.append(r)
        write_bom(r, cat, cand)
        flag = "PASS" if r["passed"] else "FAIL"
        print(f"{name:22s} {flag} m={r['mass_kg']:5.2f}kg T/W={r['thrust_margin']:.2f} (end {r['thrust_margin_end']:.2f}, worst {r['thrust_margin_worst']:.2f}) "
              f"hover={r['hover_endurance_s']:5.0f}s P={r['P_hover_W']/1000:4.1f}kW I={r['I_hover_A']:4.0f}A "
              f"pose={r['hover_pose_deg'] and (round(r['hover_pose_deg']['shoulder_pitch']), round(r['hover_pose_deg']['hip_pitch']))} thr={r['hover_throttle_frac']:.2f} "
              f"rank={r['allocation']['rank']} cond={r['allocation']['cond']:.0f} yaw {r['yaw_authority_Nm']:.2f}/{r['yaw_reaction_uncert_Nm']:.2f}Nm "
              f"${r['cost_usd']:.0f}")
        for k, v in r["checks"].items():
            if not v:
                print(f"      x {k}")
    (OUT / "summary.json").write_text(json.dumps(results, indent=1))
    with (OUT / "candidates.csv").open("w", newline="") as fh:
        keys = ["name", "height_m", "edf", "cells", "mass_kg", "pod_od_mm", "thrust_margin", "thrust_margin_end",
                "thrust_margin_worst", "hover_throttle_frac", "P_hover_W", "I_hover_A", "hover_endurance_s",
                "g_per_W", "yaw_authority_Nm", "yaw_reaction_uncert_Nm", "cost_usd", "passed"]
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k in keys})
    plots(results)
    return results


if __name__ == "__main__":
    main()
