"""Parametric EDF-humanoid design: geometry, component placement and mass budget.

One function, `build_design(candidate)`, is the single source of truth for both the
feasibility budgets (hardware/sizing.py) and the MJCF generator (models/build_mjcf.py),
so the walking, hovering and transition models can never drift apart.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
G = 9.81


def val(x):
    """Catalog values may be bare numbers or {value, status} dicts."""
    return x["value"] if isinstance(x, dict) else x


def load_catalog():
    return yaml.safe_load((ROOT / "configs" / "hardware.yaml").read_text())


def load_candidates():
    return yaml.safe_load((ROOT / "configs" / "candidates.yaml").read_text())


# ----------------------------------------------------------------------------- geometry
# Segment lengths as fractions of overall height H. Human anthropometric ratios
# (Drillis & Contini / Winter) narrowed at waist and hips for the slender silhouette.
PROPORTIONS = dict(
    ankle_h=0.045, shank=0.240, thigh=0.240,
    hip_half=0.072,           # hip joint lateral offset (narrow pelvis, must clear shin pods)
    pelvis_h=0.075, waist_h=0.085, chest_h=0.150,
    shoulder_half=0.112, upper_arm=0.165, forearm=0.150,
    neck=0.020, head_r=0.056,
    foot_len=0.125, foot_w=0.042, heel=0.025,
    thigh_r=0.034, upper_arm_r=0.022,
    waist_rx=0.044, waist_ry=0.056, chest_rx=0.060, chest_ry=0.088, pelvis_rx=0.058, pelvis_ry=0.086,
    arm_pod_len=0.195, leg_pod_len=0.195,   # min housing length / H (limb-length look; fan sets the floor)
)


@dataclass
class Component:
    name: str
    mass: float                  # kg
    kind: str                    # box | cylinder | capsule | ellipsoid | sphere
    size: tuple                  # MuJoCo-style half sizes
    pos: tuple = (0, 0, 0)       # in body frame
    quat: tuple | None = None
    status: str = "verified"
    role: str = "component"      # component | shell | servo | wiring | fastener


@dataclass
class Body:
    name: str
    parent: str | None
    pos: tuple
    joints: list = field(default_factory=list)   # (name, axis, range_deg, servo_key)
    comps: list = field(default_factory=list)


def pod_dims(edf, wall_mm):
    """Pod housing: intake lip section + fan unit + convergent nozzle."""
    d = edf["rotor_d_mm"] / 1000
    od = val(edf["od_lip_mm"]) / 1000 + 2 * (wall_mm / 1000 + 0.003)
    length = val(edf["length_mm"]) / 1000 + 0.30 * d + 0.40 * d
    return od, length


def shell_mass(area_m2, rules):
    return area_m2 * rules["shell_wall_mm"] / 1000 * rules["shell_density_gcc"] * 1000 * rules["shell_rib_factor"]


def cyl_area(r, L):
    return 2 * math.pi * r * L + 2 * math.pi * r * r


def ellipsoid_area(a, b, c):
    p = 1.6075
    return 4 * math.pi * (((a * b) ** p + (a * c) ** p + (b * c) ** p) / 3) ** (1 / p)


def wire_kg_per_m(amps):
    # Silicone-insulated copper, short high-current runs (copper + jacket), conservative.
    if amps <= 60:
        return 0.035   # 12 AWG
    if amps <= 90:
        return 0.055   # 10 AWG
    return 0.085       # 8 AWG


# ----------------------------------------------------------------------------- loads
def joint_torque_requirements(H, m_total, m_leg, m_shank_pod, m_arm, m_forearm_pod,
                              T_pod_max, P, vector_range_deg, pod_r=0.0):
    """First-order quasi-static torque demands (N m) with 1.5x dynamic factor.

    Returns {joint: (continuous_nm, peak_nm, speed_rad_s_needed)}.
    """
    # continuous = RMS over a slow gait cycle (static single-support value x duty factor);
    # peak = static worst case x 1.5 dynamic factor, or landing / hover vectoring loads.
    # joint speed requirements (rad/s) are for slow walking (~0.1 m/s, 1.2 s step period)
    dyn, duty = 1.5, 0.9
    m_upper = m_total - 2 * m_leg
    hip_w = 2 * P["hip_half"] * H
    L_th, L_sh = P["thigh"] * H, P["shank"] * H
    L_ua, L_fa = P["upper_arm"] * H, P["forearm"] * H
    s_vec = math.sin(math.radians(vector_range_deg))
    req = {}
    # No ankle: the shin pod is the shank and stands on its rim. Shifting the centre of
    # pressure across the pod base is done by the knee and hip, so the knee carries the
    # crouch moment plus the rim-COP moment (M g r_pod) as a peak.
    k = (m_total - m_shank_pod) * G * L_th * math.sin(math.radians(15))
    k_land = 2.0 * m_total * G / 2 * L_th * math.sin(math.radians(20))
    k_cop = m_total * G * pod_r
    req["knee"] = (max(k, 0.5 * k_cop) * duty, max(k * dyn, k_land, k_cop * dyn), 2.0)
    hr = m_upper * G * hip_w / 2 + m_leg * G * hip_w
    hr_hover = T_pod_max * s_vec * (L_th + L_sh * 0.5)
    req["hip_roll"] = (hr * duty, max(hr * dyn, hr_hover), 2.0)
    hp = m_leg * G * (L_th + L_sh) / 2 * math.sin(math.radians(25)) + m_upper * G * 0.03 * H
    req["hip_pitch"] = (max(hp * duty, 0.5 * hr_hover, 0.5 * k_cop * duty), max(hp * dyn, hr_hover, k_cop * dyn), 2.0)
    sp = m_arm * G * (L_ua + L_fa * 0.6) * math.sin(math.radians(30))
    sp_hover = T_pod_max * s_vec * (L_ua + L_fa * 0.5)
    req["shoulder_pitch"] = (max(sp, 0.5 * sp_hover), max(sp * dyn, sp_hover), 3.0)
    req["shoulder_roll"] = (max(sp, 0.5 * sp_hover), max(sp * dyn, sp_hover), 3.0)
    ep = m_forearm_pod * G * L_fa * 0.5
    e_h = T_pod_max * s_vec * L_fa * 0.5
    req["elbow"] = (max(ep, 0.5 * e_h), max(ep * dyn, e_h), 3.0)
    return req


def gimbal_requirement(edf, hover_frac, tilt_rate=2.0):
    """Gimbal servo torque: gyroscopic precession of fan rotor + motor bell, plus friction/inertia.

    Rotor inertia is an assumption (aluminium 12-blade rotor + outrunner bell).
    """
    d = edf["rotor_d_mm"] / 1000
    # ASSUMPTION: 3e-5 kg m^2 for a 70 mm Al 12-blade rotor + outrunner bell, scaled ~d^5
    I_rot = 3.0e-5 * (d / 0.07) ** 5
    rpm_max_loaded = edf["kv"] * edf["cells"] * 3.7 * 0.80
    omega = rpm_max_loaded * 2 * math.pi / 60 * math.sqrt(hover_frac)
    gyro = I_rot * omega * tilt_rate
    friction = 0.15
    return dict(I_rotor=I_rot, omega_hover=omega, gyro_nm=gyro, cont_nm=gyro + friction,
                peak_nm=1.6 * gyro + friction, rate=tilt_rate)


def pick_servo(cat, cont, peak, speed, allow_gang=True):
    rules = cat["rules"]
    options = []
    for key, s in cat["servo"].items():
        for n in (1, 2) if allow_gang else (1,):
            c = n * s["stall_nm"] * rules["servo_cont_frac"]
            p = n * s["stall_nm"] * rules["servo_peak_frac"]
            w = s["noload_rpm"] * 2 * math.pi / 60 * rules["servo_speed_at_load"]
            if c >= cont and p >= peak and w >= speed:
                options.append((n * s["mass_g"], n, key))
    if not options:
        best = max(cat["servo"].items(), key=lambda kv: kv[1]["stall_nm"])
        return dict(key=best[0], n=2, ok=False, mass_g=2 * best[1]["mass_g"])
    mass, n, key = min(options)
    return dict(key=key, n=n, ok=True, mass_g=mass)


# ----------------------------------------------------------------------------- design
JOINTS = {
    # name: (axis, range_deg)
    "hip_roll": ((1, 0, 0), (-25, 35)),
    "hip_pitch": ((0, 1, 0), (-100, 30)),
    "knee": ((0, 1, 0), (0, 130)),
    "shoulder_pitch": ((0, 1, 0), (-150, 60)),
    "shoulder_roll": ((1, 0, 0), (-10, 110)),
    "elbow": ((0, 1, 0), (-130, 0)),
}
VEC_AXES = {"vec_x": (1, 0, 0), "vec_y": (0, 1, 0)}


def build_design(cand_name: str, cand: dict, cat: dict, iters: int = 6):
    rules = cat["rules"]
    P = PROPORTIONS
    H = cand["height_m"]
    edf = copy.deepcopy(cat["edf"][cand["edf"]])
    esc = cat["esc"][cand["esc"]]
    bat = cat["battery"][cand["battery"]]
    n_packs = cand["pack_series"] * cand["pack_parallel"]
    vec_axes = cand["vectoring"]          # dict pod->list of axes
    vec_range = cand["vector_range_deg"]

    pod_od, pod_len_fan = pod_dims(edf, rules["shell_wall_mm"])
    pod_len = max(pod_len_fan, P["arm_pod_len"] * H)
    # bench max at loaded reference voltage (first test point is nearest the loaded pack voltage)
    v_ref, a_ref, t_ref, p_ref = edf["test_points"][0]
    T_bench_max = t_ref * G
    T_inst_max = T_bench_max * rules["install_eff"]

    # initial guess of servo masses then iterate on total mass
    servo_choice = {j: dict(key="xh540_w270", n=1, ok=True, mass_g=165) for j in JOINTS}
    gimbal_choice = dict(key="xm430_w210", n=1, ok=True, mass_g=82)
    m_total = 8.0
    for _ in range(iters):
        bodies, budget = _assemble(H, P, edf, esc, bat, n_packs, servo_choice, gimbal_choice,
                                   vec_axes, cat, pod_od, pod_len, a_ref)
        m_total = sum(c.mass for b in bodies.values() for c in b.comps)
        m_leg = sum(c.mass for b in bodies.values() if b.name.startswith("l_") and
                    any(k in b.name for k in ("thigh", "pod_leg", "fan_leg")) for c in b.comps)
        m_shank_pod = sum(c.mass for c in bodies["l_pod_leg"].comps) + sum(c.mass for c in bodies["l_fan_leg"].comps)
        m_arm = sum(c.mass for b in bodies.values() if b.name.startswith("l_") and
                    any(k in b.name for k in ("upper_arm", "forearm", "pod_arm")) for c in b.comps)
        m_fpod = sum(c.mass for c in bodies["l_pod_arm"].comps) + sum(c.mass for c in bodies["l_forearm"].comps)
        req = joint_torque_requirements(H, m_total, m_leg, m_shank_pod, m_arm, m_fpod,
                                        T_inst_max, P, vec_range, pod_r=pod_od / 2)
        new_choice = {j: pick_servo(cat, *req[j]) for j in JOINTS}
        hover_frac = m_total * G / (4 * T_inst_max)
        greq = gimbal_requirement(edf, hover_frac)
        gimbal_choice = pick_servo(cat, greq["cont_nm"], greq["peak_nm"], greq["rate"], allow_gang=False)
        if new_choice == servo_choice:
            break
        servo_choice = new_choice

    return dict(name=cand_name, cand=cand, H=H, pod_len_fan=pod_len_fan, vec_range_leg=leg_gimbal_range(edf, pod_od, rules), edf=edf, esc=esc, bat=bat, n_packs=n_packs,
                bodies=bodies, budget=budget, m_total=m_total, pod_od=pod_od, pod_len=pod_len,
                T_bench_max=T_bench_max, T_inst_max=T_inst_max, test_ref=(v_ref, a_ref, t_ref, p_ref),
                servo_choice=servo_choice, gimbal_choice=gimbal_choice, gimbal_req=greq,
                torque_req=req, P=P, vec_axes=vec_axes, vec_range=vec_range)


def _assemble(H, P, edf, esc, bat, n_packs, servo_choice, gimbal_choice, vec_axes, cat,
              pod_od, pod_len, pod_amps):
    rules = cat["rules"]
    S = cat["servo"]
    bodies: dict[str, Body] = {}

    def servo_comp(joint, pos, prefix=""):
        ch = servo_choice[joint]
        s = S[ch["key"]]
        w, h, d = (x / 2000 for x in s["dims_mm"])
        return Component(f"{prefix}servo_{joint}", ch["mass_g"] / 1000, "box", (w * ch["n"], d, h), pos,
                         role="servo", status="verified")

    # ---- heights
    z_ankle = P["ankle_h"] * H
    z_knee = z_ankle + P["shank"] * H
    z_hip = z_knee + P["thigh"] * H
    z_pelvis = z_hip + 0.02 * H
    z_waist = z_pelvis + P["pelvis_h"] * H
    z_chest = z_waist + P["waist_h"] * H
    z_shoulder = z_chest + P["chest_h"] * H * 0.85
    z_head = z_chest + P["chest_h"] * H + P["neck"] * H + P["head_r"] * H

    # ---- torso (pelvis+waist+chest one rigid body, frame origin at pelvis centre)
    torso = Body("torso", None, (0, 0, z_pelvis))
    pr = (P["pelvis_rx"] * H, P["pelvis_ry"] * H, P["pelvis_h"] * H * 0.75)
    wr = (P["waist_rx"] * H, P["waist_ry"] * H, P["waist_h"] * H * 0.95)
    cr = (P["chest_rx"] * H, P["chest_ry"] * H, P["chest_h"] * H * 0.70)
    torso.comps += [
        Component("pelvis_shell", shell_mass(ellipsoid_area(*pr), rules), "ellipsoid", pr,
                  (0, 0, 0.3 * P["pelvis_h"] * H), role="shell", status="assumption"),
        Component("waist_shell", shell_mass(ellipsoid_area(*wr), rules), "ellipsoid", wr,
                  (0, 0, z_waist - z_pelvis + 0.4 * P["waist_h"] * H), role="shell", status="assumption"),
        Component("chest_shell", shell_mass(ellipsoid_area(*cr), rules), "ellipsoid", cr,
                  (0, 0, z_chest - z_pelvis + 0.45 * P["chest_h"] * H), role="shell", status="assumption"),
        Component("head_shell", shell_mass(4 * math.pi * (P["head_r"] * H) ** 2, rules) + 0.02, "ellipsoid",
                  (P["head_r"] * H * 0.88, P["head_r"] * H * 0.80, P["head_r"] * H * 1.08), (0, 0, z_head - z_pelvis),
                  role="shell", status="assumption"),
        Component("spine_spar", 0.072 * H / 0.7, "capsule", (0.006, (z_shoulder - z_pelvis) / 2), (0, 0, (z_shoulder - z_pelvis) / 2),
                  role="shell", status="assumption"),   # CF tube + Al brackets
    ]
    # battery packs: vertical in chest/waist, side by side
    dims = sorted(bat["dims_mm"])
    bd, bw, bh = dims[0] / 2000, dims[1] / 2000, dims[2] / 2000
    for i in range(n_packs):
        y = (i - (n_packs - 1) / 2) * (2 * bw + 0.004)
        torso.comps.append(Component(f"battery_{i}", bat["mass_g"] / 1000, "box", (bd, bw, bh),
                                     (0.0, y, z_waist - z_pelvis + bh * 0.9), status=bat["status"]))
    torso.comps += [
        Component("fc_avionics", (cat["fc"]["matek_h743_slim_v3"]["mass_g"] + val(cat["fc"]["other_avionics_g"])) / 1000,
                  "box", (0.02, 0.02, 0.01), (-0.03 * H, 0, z_chest - z_pelvis + 0.11 * H), status="assumption"),
        Component("pdb_main_leads", 0.06 + 0.04 * n_packs, "box", (0.02, 0.03, 0.01),
                  (0.02 * H, 0, z_waist - z_pelvis), role="wiring", status="assumption"),
    ]
    # hip and shoulder servos mounted in torso (stator side)
    for side, sy in (("l", 1), ("r", -1)):
        torso.comps.append(servo_comp("hip_roll", (0, sy * P["hip_half"] * H * 0.6, z_hip - z_pelvis + 0.02 * H), f"{side}_"))
        torso.comps.append(servo_comp("shoulder_pitch", (0, sy * (P["shoulder_half"] * H - 0.03 * H), z_shoulder - z_pelvis), f"{side}_"))
    bodies["torso"] = torso

    for side, sy in (("l", 1), ("r", -1)):
        # ---------------- leg
        hip = Body(f"{side}_hip", "torso", (0, sy * P["hip_half"] * H, z_hip - z_pelvis),
                   joints=[(f"{side}_hip_roll", "hip_roll")])
        hip.comps.append(servo_comp("hip_pitch", (0, 0, 0)))
        hip.comps.append(Component("hip_bracket", 0.03 * H / 0.7, "sphere", (0.028 * H,), role="shell", status="assumption"))
        bodies[hip.name] = hip

        Lt = P["thigh"] * H
        thigh = Body(f"{side}_thigh", hip.name, (0, 0, 0), joints=[(f"{side}_hip_pitch", "hip_pitch")])
        thigh.comps += [
            Component("thigh_shell", shell_mass(cyl_area(P["thigh_r"] * H, Lt * 0.8), rules), "capsule",
                      (P["thigh_r"] * H, Lt * 0.38), (0, 0, -Lt * 0.48), role="shell", status="assumption"),
            servo_comp("knee", (0, 0, -Lt * 0.86)),
        ]
        bodies[thigh.name] = thigh

        # shin pod IS the lower leg: knee yoke on the pod's top centre, pod stands on rim pads
        z_kn = z_knee
        legpod, fan = _leg_pod_bodies(side, thigh.name, (0, 0, -Lt), z_kn, H, P, edf, esc, gimbal_choice,
                                      cat, pod_od, vec_axes["leg"], servo_comp)
        bodies[legpod.name] = legpod
        bodies[fan.name] = fan
        # leg wiring: torso -> pod ESC, two conductors
        run = (z_pelvis - z_kn) + 0.06 + 0.08
        wkg = 2 * run * wire_kg_per_m(pod_amps)
        thigh.comps.append(Component("wire_leg_thigh", wkg, "capsule", (0.006, Lt * 0.4), (0.01, 0, -Lt * 0.5),
                                     role="wiring", status="assumption"))

        # ---------------- arm
        Lua, Lfa = P["upper_arm"] * H, P["forearm"] * H
        sh = Body(f"{side}_shoulder", "torso", (0, sy * P["shoulder_half"] * H, z_shoulder - z_pelvis),
                  joints=[(f"{side}_shoulder_pitch", "shoulder_pitch")])
        sh.comps.append(servo_comp("shoulder_roll", (0, 0, 0)))
        sh.comps.append(Component("shoulder_cap", 0.02 * H / 0.7, "sphere", (0.03 * H,), role="shell", status="assumption"))
        bodies[sh.name] = sh
        ua = Body(f"{side}_upper_arm", sh.name, (0, 0, 0), joints=[(f"{side}_shoulder_roll", "shoulder_roll")])
        ua.comps += [
            Component("upper_arm_shell", shell_mass(cyl_area(P["upper_arm_r"] * H, Lua * 0.8), rules), "capsule",
                      (P["upper_arm_r"] * H, Lua * 0.36), (0, 0, -Lua * 0.5), role="shell", status="assumption"),
            servo_comp("elbow", (0, 0, -Lua * 0.85)),
        ]
        run_a = (z_shoulder - z_pelvis) * 0.6 + Lua + Lfa * 0.5
        wka = 2 * run_a * wire_kg_per_m(pod_amps)
        ua.comps.append(Component("wire_arm", wka, "capsule", (0.006, Lua * 0.35), (0, 0, -Lua * 0.5),
                                  role="wiring", status="assumption"))
        bodies[ua.name] = ua
        fa = Body(f"{side}_forearm", ua.name, (0, 0, -Lua), joints=[(f"{side}_elbow", "elbow")])
        fa.comps.append(Component("elbow_yoke", 0.015 * H / 0.7, "sphere", (0.024 * H,), role="shell", status="assumption"))
        bodies[fa.name] = fa
        # forearm pod: coaxial with forearm, starts just below the elbow yoke, extends past wrist (hand inside cap)
        armpod = _pod_body(f"{side}_pod_arm", fa.name, (0, 0, -0.03 * H - pod_len / 2), edf, esc,
                           gimbal_choice, cat, pod_od, pod_len, vec_axes["arm"], side)
        bodies[armpod.name] = armpod

    # fasteners / inserts / bearings as a fraction of dry mass, spread on the torso & limbs pro-rata
    dry = sum(c.mass for b in bodies.values() for c in b.comps)
    for b in bodies.values():
        bm = sum(c.mass for c in b.comps)
        if bm > 0:
            b.comps.append(Component("fasteners_bearings", bm * rules["fastener_frac"], "sphere", (0.005,),
                                     role="fastener", status="assumption"))
    budget = _budget(bodies)
    return bodies, budget


def _pod_body(name, parent, pos, edf, esc, gimbal_choice, cat, pod_od, pod_len, axes, side):
    """Pod body hangs from its limb through vectoring joint(s) at the pod centre.

    Pod frame: +z points from nozzle toward intake (thrust direction when exhausting downward).
    """
    rules = cat["rules"]
    S = cat["servo"]
    joints = [(f"{name}_vec_{ax}", f"vec_{ax}") for ax in axes]
    b = Body(name, parent, pos, joints=joints)
    r = pod_od / 2
    d = edf["rotor_d_mm"] / 1000
    fan_z = pod_len / 2 - 0.30 * d - val(edf["length_mm"]) / 2000
    b.comps += [
        Component("pod_shell", shell_mass(2 * math.pi * r * pod_len, rules) * 1.15, "cylinder", (r, pod_len / 2), (0, 0, 0),
                  role="shell", status="assumption"),
        Component("edf_unit", val(edf["mass_g"]) / 1000, "cylinder", (val(edf["od_mm"]) / 2000, val(edf["length_mm"]) / 2000),
                  (0, 0, fan_z), status=edf["status"] if not isinstance(edf["mass_g"], dict) else "assumption"),
        Component("esc", esc["mass_g"] / 1000, "box", (0.017, 0.010, 0.038), (r * 0.55, 0, fan_z + 0.02), status="verified"),
        Component("inlet_guard", 0.012 * (d / 0.07) ** 2, "cylinder", (r * 0.9, 0.002), (0, 0, pod_len / 2 - 0.004),
                  role="shell", status="assumption"),
        Component("gimbal_ring_bearings", 0.045 * (d / 0.07) ** 2 * len(axes), "cylinder", (r * 1.02, 0.006), (0, 0, 0),
                  role="shell", status="assumption"),
    ]
    for i, ax in enumerate(axes):
        s = S[gimbal_choice["key"]]
        b.comps.append(Component(f"gimbal_servo_{ax}", s["mass_g"] / 1000, "box", (0.014, 0.017, 0.023),
                                 (-r * 0.4 if i == 0 else r * 0.4, 0, pod_len / 2 - 0.035), role="servo", status="verified"))
    return b


POD_STANDOFF = 0.035   # x H: rim pads lift the nozzle exit so exhaust can escape radially at liftoff


def leg_gimbal_range(edf, pod_od, rules):
    """Max tilt (deg) of a lip-less fan cartridge inside the fixed shin shell (bore clearance 1 mm).
    The intake bellmouth is moulded into the shell, so only the bare housing tilts."""
    r_bore = pod_od / 2 - rules["shell_wall_mm"] / 1000 - 0.001
    rf = val(edf["od_mm"]) / 2000
    hl = val(edf["length_mm"]) / 2000
    th = 0.0
    while rf * math.cos(math.radians(th + 0.5)) + hl * math.sin(math.radians(th + 0.5)) <= r_bore and th < 20:
        th += 0.5
    return th


def _leg_pod_bodies(side, parent, pos, z_knee, H, P, edf, esc, gimbal_choice, cat, pod_od, axes, servo_comp):
    """Shin pod = lower leg. Closed duct from the knee yoke down to the nozzle exit just below the
    fan, then an open 4-strut skirt down to the rim pads. Raising the exit plane (instead of
    ducting the jet to the floor) gives the exhaust a large radial escape path at liftoff."""
    rules = cat["rules"]
    S = cat["servo"]
    r = pod_od / 2
    d = edf["rotor_d_mm"] / 1000
    fan_len = val(edf["length_mm"]) / 1000
    yoke = 0.02 * H
    standoff = POD_STANDOFF * H
    fan_z = -yoke - 0.30 * d - fan_len / 2          # fan right under the intake bellmouth
    exit_z = fan_z - fan_len / 2 - 0.25 * d         # short convergent nozzle
    L = -exit_z - yoke                              # closed duct length
    zc = -yoke - L / 2
    z_ring = -(z_knee - standoff)                   # skirt bottom ring, pads below
    skirt_h = exit_z - z_ring
    pod = Body(f"{side}_pod_leg", parent, pos, joints=[(f"{side}_knee", "knee")])
    pod.comps += [
        Component("knee_yoke", 0.04 * H / 0.7, "sphere", (0.026 * H,), (0, 0, -0.004 * H), role="shell", status="assumption"),
        Component("pod_shell", shell_mass(2 * math.pi * r * L, rules) * 1.35, "cylinder", (r, L / 2), (0, 0, zc),
                  role="shell", status="assumption"),   # x1.35: load-bearing shell (it is the shank)
        Component("intake_bellmouth_guard", 0.03 * (d / 0.07) ** 2, "cylinder", (r * 0.9, 0.004), (0, 0, -yoke - 0.004),
                  role="shell", status="assumption"),
        Component("esc", esc["mass_g"] / 1000, "box", (0.017, 0.010, 0.038), (-r * 0.6, 0, fan_z + 0.01), status="verified"),
        Component("skirt_ring", shell_mass(2 * math.pi * r * 0.012, rules) * 2.0, "cylinder", (r, 0.006), (0, 0, z_ring + 0.006),
                  role="shell", status="assumption"),
    ]
    # 4 load-bearing skirt struts at 45/135/225/315 deg -> vents face front, back and both sides
    strut_w = 0.018
    for k in range(4):
        a = math.pi / 4 + k * math.pi / 2
        pod.comps.append(Component(f"skirt_strut_{k}", shell_mass(strut_w * skirt_h * 2, rules) * 2.5, "box",
                                   (0.004, strut_w / 2, skirt_h / 2),
                                   (r * 0.96 * math.cos(a), r * 0.96 * math.sin(a), z_ring + skirt_h / 2),
                                   role="shell", status="assumption"))
    for i, ax in enumerate(axes):
        s = S[gimbal_choice["key"]]
        pod.comps.append(Component(f"gimbal_servo_{ax}", s["mass_g"] / 1000, "box", (0.014, 0.017, 0.023),
                                   (r * 0.62 * (1 if i == 0 else 0), r * 0.62 * (0 if i == 0 else 1), fan_z + 0.02),
                                   role="servo", status="verified"))
    # 8 rubber rim pads (real, finite contact patches, no hidden foot): post + 8 mm hemispherical tip
    tip = 0.008
    for k in range(8):
        a = k * math.pi / 4
        x, y = r * 0.86 * math.cos(a), r * 0.86 * math.sin(a)
        pod.comps.append(Component(f"pad_{k}", 0.004, "box", (0.008, 0.008, (standoff - tip) / 2),
                                   (x, y, z_ring - (standoff - tip) / 2), role="shell", status="assumption"))
        pod.comps.append(Component(f"padtip_{k}", 0.002, "sphere", (tip,), (x, y, z_ring - standoff + tip),
                                   role="shell", status="assumption"))
    fan = Body(f"{side}_fan_leg", pod.name, (0, 0, fan_z), joints=[(f"{side}_pod_leg_vec_{ax}", f"vec_{ax}") for ax in axes])
    fan.comps += [
        Component("edf_unit", val(edf["mass_g"]) / 1000, "cylinder", (val(edf["od_mm"]) / 2000, fan_len / 2),
                  (0, 0, 0), status=edf["status"] if not isinstance(edf["mass_g"], dict) else "assumption"),
        Component("gimbal_ring_bearings", 0.045 * (d / 0.07) ** 2 * len(axes), "cylinder", (val(edf["od_mm"]) / 2000 + 0.003, 0.006),
                  (0, 0, 0), role="shell", status="assumption"),
    ]
    return pod, fan


def nozzle_exit_height(design):
    """Nozzle exit height above ground (m) when standing vertically on the rim pads."""
    H, d = design["H"], design["edf"]["rotor_d_mm"] / 1000
    fan_len = val(design["edf"]["length_mm"]) / 1000
    z_knee = (design["P"]["ankle_h"] + design["P"]["shank"]) * H
    return z_knee - (0.02 * H + 0.30 * d + fan_len + 0.25 * d)


def _budget(bodies):
    out = {}
    for b in bodies.values():
        for c in b.comps:
            key = c.role if c.role != "component" else c.name.split("_")[0]
            if c.name.startswith("edf"):
                key = "edf"
            elif c.name.startswith("esc"):
                key = "esc"
            elif c.name.startswith("battery"):
                key = "battery"
            elif c.name.startswith("fc"):
                key = "avionics"
            elif c.name.startswith("gimbal_servo"):
                key = "gimbal_servos"
            elif c.role == "servo":
                key = "joint_servos"
            out[key] = out.get(key, 0) + c.mass
    return out
