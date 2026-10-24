"""Procedural MJCF generator for the EDF humanoid.

Usage: python models/build_mjcf.py [candidate]   -> models/edf_humanoid_<candidate>.xml
Every physical component from models/design.py becomes a massed geom in its body, so
MuJoCo computes the integrated mass, COM and inertia. Livery geoms are visual only (mass 0).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.design import (JOINTS, VEC_AXES, G, build_design, load_candidates,  # noqa: E402
                           load_catalog, val)

ROOT = Path(__file__).resolve().parents[1]

WHITE = "0.94 0.94 0.92 1"
RED = "0.72 0.07 0.10 1"
BLUE = "0.08 0.18 0.50 1"
STEEL = "0.30 0.32 0.36 1"
DARK = "0.10 0.10 0.12 1"
VISOR = "0.12 0.14 0.20 1"
HAIR = "0.93 0.80 0.45 1"

SPOOL_TAU = {70: 0.12, 64: 0.10, 80: 0.14, 90: 0.16}   # s, first-order EDF spool time constant (assumption)
ARMATURE = {"xm430_w350": 0.012, "xm430_w210": 0.006, "xh540_w270": 0.020, "sts3250": 0.010}   # kg m^2 reflected (assumption)


def f(*xs):
    return " ".join(f"{x:.5g}" for x in xs)


def standing_pose(design):
    """Nominal joint angles (rad). Arms abducted so pods clear the thighs."""
    P, H = design["P"], design["H"]
    r_pod = design["pod_od"] / 2
    # lateral clearance: pod inner edge must clear thigh outer edge by >= 1.5 cm
    need = P["hip_half"] * H + P["thigh_r"] * H + r_pod + 0.015 - P["shoulder_half"] * H
    L = P["upper_arm"] * H + 0.03 * H + design["pod_len"] / 2
    abd = math.asin(max(0.0, min(0.9, need / L))) if need > 0 else 0.0
    abd = max(abd, math.radians(8))
    q = {}
    for s, sy in (("l", 1), ("r", -1)):
        q[f"{s}_shoulder_roll"] = abd           # sign handled by mirrored axis
        # hip and knee cancel so the shin pod stands vertical, flat on its rim pads
        q[f"{s}_hip_pitch"] = math.radians(-8)
        q[f"{s}_knee"] = math.radians(8)
        q[f"{s}_elbow"] = math.radians(-6)
    return q


def hover_pose(design, shoulder_pitch=None, hip_pitch=None):
    """Hover pose: arms hang at the sides (forearm pods lateral, exhaust beside the thighs),
    legs nearly straight. Shoulder and hip pitch are tuned by hardware/sizing.py to balance
    the trim thrust split between forearm and shin pods."""
    q = standing_pose(design)
    sp = design.get("hover_shoulder_pitch", 0.0) if shoulder_pitch is None else shoulder_pitch
    hp = design.get("hover_hip_pitch", math.radians(-4)) if hip_pitch is None else hip_pitch
    for s in ("l", "r"):
        q[f"{s}_shoulder_pitch"] = sp
        q[f"{s}_hip_pitch"] = hp
        q[f"{s}_knee"] = -hp                         # shin pods stay parallel to the torso axis
        q[f"{s}_elbow"] = 0.0
    return q


def build_xml(design, train: bool = False) -> str:
    cat = load_catalog()
    rules = cat["rules"]
    bodies = design["bodies"]
    children = {}
    for b in bodies.values():
        children.setdefault(b.parent, []).append(b)
    H = design["H"]
    edf = design["edf"]
    tau = SPOOL_TAU.get(edf["rotor_d_mm"], 0.15)
    out = []
    w = out.append
    w(f'<mujoco model="edf_humanoid_{design["name"]}">')
    w('  <compiler angle="radian" autolimits="true"/>')
    if train:
        w('  <option timestep="0.002" integrator="implicitfast" iterations="10" ls_iterations="10" gravity="0 0 -9.81">'
          '<flag eulerdamp="disable"/></option>')
    else:
        w('  <option timestep="0.001" integrator="implicitfast" gravity="0 0 -9.81"/>')
    w('  <visual><global offwidth="1600" offheight="1600"/><quality shadowsize="4096"/>'
      '<headlight ambient="0.35 0.35 0.35" diffuse="0.5 0.5 0.5"/></visual>')
    w('  <asset>')
    w('    <texture name="grid" type="2d" builtin="checker" rgb1="0.82 0.84 0.86" rgb2="0.74 0.76 0.79" width="512" height="512"/>')
    w('    <material name="floor" texture="grid" texrepeat="8 8" reflectance="0.05"/>')
    w('    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.86 0.90 0.95" rgb2="0.55 0.62 0.72" width="256" height="256"/>')
    w(f'    <material name="white" rgba="{WHITE}" specular="0.5" shininess="0.6"/>')
    w(f'    <material name="red" rgba="{RED}" specular="0.4" shininess="0.5"/>')
    w(f'    <material name="blue" rgba="{BLUE}" specular="0.4" shininess="0.5"/>')
    w(f'    <material name="steel" rgba="{STEEL}" specular="0.9" shininess="0.9" reflectance="0.2"/>')
    w(f'    <material name="dark" rgba="{DARK}" specular="0.3"/>')
    w(f'    <material name="visor" rgba="{VISOR}" specular="1" shininess="1" reflectance="0.3"/>')
    w(f'    <material name="hair" rgba="{HAIR}" specular="0.2"/>')
    w('    <material name="internal" rgba="0.2 0.6 0.3 0.6"/>')
    w('  </asset>')
    w('  <default>')
    w('    <joint damping="0.4" frictionloss="0.03" limited="true"/>')
    w('    <geom contype="0" conaffinity="0" condim="3" friction="0.8 0.02 0.001" solref="0.005 1"/>')
    w('    <default class="internal"><geom group="3" material="internal"/></default>')
    w('    <default class="shell"><geom group="1" material="white" contype="2" conaffinity="1"/></default>')
    w('    <default class="livery"><geom group="2" mass="0" contype="0" conaffinity="0"/></default>')
    w('    <site group="4" rgba="1 0.5 0 1" size="0.008"/>')
    w('  </default>')
    w('  <worldbody>')
    w('    <light pos="1.5 -1.5 3" dir="-0.5 0.5 -1" diffuse="0.7 0.7 0.7" castshadow="true"/>')
    w('    <light pos="-2 2 2.5" dir="0.6 -0.6 -1" diffuse="0.3 0.3 0.35" castshadow="false"/>')
    w('    <geom name="floor" type="plane" size="0 0 0.05" material="floor" contype="1" conaffinity="1"/>')

    def emit(b, depth):
        ind = "  " * depth
        w(f'{ind}<body name="{b.name}" pos="{f(*b.pos)}">')
        if b.parent is None:
            w(f'{ind}  <freejoint name="root"/>')
            w(f'{ind}  <site name="imu" pos="{f(-0.02 * H, 0, 0.30 * H)}"/>')
        for jname, jkey in b.joints:
            if jkey in VEC_AXES:
                axis = VEC_AXES[jkey]
                v = design["vec_range_leg"] if "pod_leg" in jname else design["vec_range"]
                rng = (-v, v)
                arm = ARMATURE[design["gimbal_choice"]["key"]]
            else:
                axis, rng = JOINTS[jkey]
                arm = ARMATURE[design["servo_choice"][jkey]["key"]] * design["servo_choice"][jkey]["n"]
                if jname.startswith("r_") and jkey in ("hip_roll", "ankle_roll", "shoulder_roll"):
                    axis = tuple(-a for a in axis)        # mirror so +angle is abduction on both sides
            w(f'{ind}  <joint name="{jname}" axis="{f(*axis)}" range="{f(math.radians(rng[0]), math.radians(rng[1]))}" armature="{arm:.4g}"/>')
        for c in b.comps:
            cls = "shell" if c.role == "shell" and c.kind != "sphere" else "internal"
            if c.name in ("hip_bracket", "shoulder_cap", "elbow_yoke", "knee_yoke"):
                cls = "shell"
            mat = ' material="steel"' if c.name in ("hip_bracket", "elbow_yoke", "knee_yoke") else ""
            if c.name == "skirt_ring":
                mat = ' material="steel"'
            if c.name.startswith("pad_"):
                mat = ' material="dark" contype="0" conaffinity="0"'
            if c.name.startswith("padtip_"):
                cls = "shell"
                mat = ' material="dark" friction="0.9 0.02 0.001"'
            if c.name == "shoulder_cap":
                mat = ' material="blue"'
            if train and not c.name.startswith("padtip_"):
                mat += ' contype="0" conaffinity="0"' if "contype" not in mat else ""
            size = f(*c.size)
            w(f'{ind}  <geom name="{b.name}_{c.name}" class="{cls}" type="{c.kind}" size="{size}" '
              f'pos="{f(*c.pos)}" mass="{max(c.mass, 1e-6):.5g}"{mat}/>')
        for line in livery(b, design):
            if train and "<geom" in line:
                continue
            w(f"{ind}  {line}")
        for ch in children.get(b.name, []):
            emit(ch, depth + 1)
        w(f'{ind}</body>')

    emit(bodies["torso"], 2)
    w('  </worldbody>')
    w('  <contact><exclude body1="torso" body2="l_thigh"/><exclude body1="torso" body2="r_thigh"/></contact>')

    # actuators
    w('  <actuator>')
    if train:
        # torque motors: the env applies PD + servo torque-speed envelope + EDF thrust (xfrc)
        for b in bodies.values():
            for jname, jkey in b.joints:
                ch = design["gimbal_choice"] if jkey in VEC_AXES else design["servo_choice"][jkey]
                s = cat["servo"][ch["key"]]
                tmax = ch["n"] * s["stall_nm"] * rules["servo_peak_frac"]
                w(f'    <motor name="{jname}" joint="{jname}" ctrlrange="{-tmax:.4g} {tmax:.4g}"/>')
        w('  </actuator>')
        w('</mujoco>')
        return "\n".join(out)
    for b in bodies.values():
        for jname, jkey in b.joints:
            if jkey in VEC_AXES:
                ch = design["gimbal_choice"]
            else:
                ch = design["servo_choice"][jkey]
            s = cat["servo"][ch["key"]]
            tmax = ch["n"] * s["stall_nm"] * rules["servo_peak_frac"]
            kp = 40 * tmax
            w(f'    <position name="{jname}" joint="{jname}" kp="{kp:.4g}" kv="{0.06 * kp:.4g}" forcerange="{-tmax:.4g} {tmax:.4g}" '
              f'inheritrange="1"/>')
    for pod in ("l_pod_arm", "r_pod_arm", "l_pod_leg", "r_pod_leg"):
        w(f'    <general name="thrust_{pod}" site="thrust_{pod}" gear="0 0 1 0 0 0" dyntype="filter" dynprm="{tau}" '
          f'ctrlrange="0 {design["T_inst_max"]:.4g}" gainprm="1" biasprm="0 0 0"/>')
    w('  </actuator>')
    w('  <sensor>')
    w('    <gyro name="gyro" site="imu" noise="0.005"/>')
    w('    <accelerometer name="accel" site="imu" noise="0.05"/>')
    w('    <framequat name="quat" objtype="site" objname="imu"/>')
    for s in ("l", "r"):
        w(f'    <touch name="{s}_foot_touch" site="{s}_sole"/>')
    w('  </sensor>')
    w('</mujoco>')
    return "\n".join(out)


def livery(b, design):
    """Visual-only USAF-style livery, intake/nozzle detail, face. Also places sites."""
    H, P = design["H"], design["P"]
    w = []
    n = b.name
    r_pod, L_pod = design["pod_od"] / 2, design["pod_len"]
    if n == "torso":
        zc = [c for c in b.comps if c.name == "chest_shell"][0]
        cz = zc.pos[2]
        crx, cry, crz = zc.size
        # blue chest band and red pinstripe under it (horizontal bands, not a ribbon)
        w.append(f'<geom class="livery" type="ellipsoid" size="{f(crx * 1.01, cry * 1.01, crz * 0.16)}" pos="{f(0, 0, cz - crz * 0.25)}" material="blue"/>')
        w.append(f'<geom class="livery" type="ellipsoid" size="{f(crx * 0.97, cry * 0.97, crz * 0.05)}" pos="{f(0, 0, cz - crz * 0.48)}" material="red"/>')
        wc = [c for c in b.comps if c.name == "waist_shell"][0]
        w.append(f'<geom class="livery" type="ellipsoid" size="{f(wc.size[0] * 1.03, wc.size[1] * 1.03, wc.size[2] * 0.12)}" pos="{f(0, 0, wc.pos[2] - wc.size[2] * 0.55)}" material="steel"/>')
        hc = [c for c in b.comps if c.name == "head_shell"][0]
        hx, hy, hz = hc.size
        hz0 = hc.pos[2]
        # wide face-plate visor band (not an eye)
        w.append(f'<geom class="livery" type="ellipsoid" size="{f(hx * 0.42, hy * 0.86, hz * 0.30)}" pos="{f(hx * 0.62, 0, hz0 + hz * 0.05)}" material="visor"/>')
        w.append(f'<geom class="livery" type="ellipsoid" size="{f(hx * 0.30, hy * 0.40, hz * 0.12)}" pos="{f(hx * 0.74, 0, hz0 - hz * 0.48)}" material="steel"/>')
        w.append(f'<geom class="livery" type="capsule" size="{f(0.016 * H, 0.012 * H)}" pos="{f(0, 0, hz0 - hz - 0.008 * H)}" material="steel"/>')
        # short bob hair: visual-only reference feature (excluded from physics)
        w.append(f'<geom class="livery" type="ellipsoid" size="{f(hx * 1.08, hy * 1.12, hz * 0.70)}" pos="{f(-hx * 0.10, 0, hz0 + hz * 0.36)}" material="hair"/>')
        w.append(f'<geom class="livery" type="ellipsoid" size="{f(hx * 0.80, hy * 1.14, hz * 0.62)}" pos="{f(-hx * 0.30, 0, hz0 - hz * 0.12)}" material="hair"/>')
        w.append(f'<geom class="livery" type="ellipsoid" size="{f(hx * 0.30, hy * 0.85, hz * 0.22)}" pos="{f(hx * 0.70, 0, hz0 + hz * 0.52)}" euler="0 0.35 0" material="hair"/>')
        w.append(f'<site name="com_ref" pos="0 0 0"/>')
    elif n.endswith("_thigh") or n.endswith("_upper_arm"):
        sh = [c for c in b.comps if c.name.endswith("_shell")][0]
        r, hl = sh.size
        w.append(f'<geom class="livery" type="cylinder" size="{f(r * 1.04, hl * 0.06)}" pos="{f(sh.pos[0], 0, sh.pos[2] + hl * 0.55)}" material="red"/>')
        w.append(f'<geom class="livery" type="cylinder" size="{f(r * 1.04, hl * 0.06)}" pos="{f(sh.pos[0], 0, sh.pos[2] + hl * 0.35)}" material="blue"/>')
    elif "_pod_" in n:
        shell = [c for c in b.comps if c.name == "pod_shell"][0]
        r, hl = shell.size
        zc = shell.pos[2]
        top, bot = zc + hl, zc - hl
        L = 2 * hl
        # bands: red + blue rings, dark intake grille on top, steel collar
        w.append(f'<geom class="livery" type="cylinder" size="{f(r * 1.012, L * 0.05)}" pos="{f(0, 0, zc + L * 0.22)}" material="red"/>')
        w.append(f'<geom class="livery" type="cylinder" size="{f(r * 1.012, L * 0.025)}" pos="{f(0, 0, zc + L * 0.10)}" material="blue"/>')
        w.append(f'<geom class="livery" type="cylinder" size="{f(r * 0.86, 0.003)}" pos="{f(0, 0, top + 0.001)}" material="dark"/>')
        w.append(f'<geom class="livery" type="cylinder" size="{f(r * 1.03, L * 0.03)}" pos="{f(0, 0, top - L * 0.03)}" material="steel"/>')
        # side intake louvres (inlet must breathe around the elbow/knee mount)
        for k in range(6):
            a = k * math.pi / 3 + math.pi / 6
            w.append(f'<geom class="livery" type="box" size="{f(0.002, r * 0.28, L * 0.07)}" '
                     f'pos="{f(r * math.cos(a), r * math.sin(a), zc + L * 0.36)}" euler="0 0 {a:.4f}" material="dark"/>')
        if n.endswith("_leg"):
            # steel nozzle collar at the duct exit; exhaust vents between the skirt struts
            w.append(f'<geom class="livery" type="cylinder" size="{f(r * 1.02, 0.008)}" pos="{f(0, 0, bot + 0.008)}" material="steel"/>')
            w.append(f'<geom class="livery" type="cylinder" size="{f(r * 0.80, 0.012)}" pos="{f(0, 0, bot - 0.010)}" material="steel"/>')
            w.append(f'<geom class="livery" type="cylinder" size="{f(r * 0.66, 0.002)}" pos="{f(0, 0, bot - 0.022)}" material="dark"/>')
            pads = [c for c in b.comps if c.name.startswith("pad_")]
            zp = pads[0].pos[2] - pads[0].size[2]
            w.append(f'<site name="{n[0]}_sole" type="cylinder" size="{f(r * 1.05, 0.01)}" pos="{f(0, 0, zp)}"/>')
            w.append(f'<site name="nozzle_{n}" pos="{f(0, 0, bot)}"/>')
        else:
            w.append(f'<geom class="livery" type="cylinder" size="{f(r * 0.80, L * 0.06)}" pos="{f(0, 0, bot - L * 0.04)}" material="steel"/>')
            w.append(f'<geom class="livery" type="cylinder" size="{f(r * 0.66, 0.002)}" pos="{f(0, 0, bot - L * 0.10)}" material="dark"/>')
            w.append(f'<site name="nozzle_{n}" pos="{f(0, 0, bot - L * 0.10)}"/>')
        edfc = [c for c in b.comps if c.name == "edf_unit"]
        if edfc:
            w.append(f'<site name="thrust_{n}" pos="{f(*edfc[0].pos)}"/>')
    elif "_fan_" in n:
        # fan cartridge gimballed inside the fixed shin shell; thrust acts here
        w.append(f'<site name="thrust_{n.replace("_fan_", "_pod_")}" pos="0 0 0"/>')
    elif n.endswith("_forearm"):
        pass
    return w


def generate(cand_name: str, out_dir: Path | None = None, train: bool = False):
    design = build_design(cand_name, load_candidates()[cand_name], load_catalog())
    out_dir = out_dir or ROOT / "models"
    path = out_dir / f"edf_humanoid_{cand_name}{'_train' if train else ''}.xml"
    path.write_text(build_xml(design, train=train))
    return path, design


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "D_88cm_jp90_8s"
    p, d = generate(name)
    print(p, f"mass={d['m_total']:.3f} kg")


def pose_qpos(model, pose: dict, ground: bool = True, height: float = 0.0):
    """qpos for a named-joint pose; with ground=True the lowest sole touches z=0."""
    import mujoco
    import numpy as np
    d = mujoco.MjData(model)
    q = model.qpos0.copy()
    for name, a in pose.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            q[model.jnt_qposadr[jid]] = a
    q[2] = 1.0
    d.qpos[:] = q
    mujoco.mj_kinematics(model, d)
    zmin = 1e9
    for g in range(model.ngeom):
        if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
            continue
        if model.geom_bodyid[g] == 0:
            continue
        # lowest corner of the geom's local AABB (exact for boxes/cylinders, conservative otherwise)
        R = d.geom_xmat[g].reshape(3, 3)
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        c, h = model.geom_aabb[g][:3], model.geom_aabb[g][3:]
        corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * h + c
        z = (d.geom_xpos[g][2] + corners @ R[2, :]).min()
        zmin = min(zmin, z)
    q[2] = 1.0 - zmin + (height if not ground else 0.0)
    return q
