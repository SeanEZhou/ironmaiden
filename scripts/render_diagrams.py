"""Engineering diagrams: colour-coded internals and servo motion-axis map.

python scripts/render_diagrams.py [candidate]
  -> results/m0/internals_<cand>.png, results/m0/servo_axes_<cand>.png
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.build_mjcf import generate, pose_qpos, standing_pose  # noqa: E402
from models.design import load_catalog  # noqa: E402

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

CATEGORIES = [  # (label, rgba, name predicate)
    ("EDF fan + motor", (1.00, 0.55, 0.05, 1), lambda n: "edf_unit" in n),
    ("ESC", (0.55, 0.25, 0.85, 1), lambda n: n.endswith("_esc")),
    ("Battery", (0.98, 0.85, 0.10, 1), lambda n: "battery" in n),
    ("Joint servo", (0.10, 0.35, 0.95, 1), lambda n: "servo_" in n and "gimbal" not in n),
    ("Gimbal servo", (0.05, 0.85, 0.90, 1), lambda n: "gimbal_servo" in n),
    ("Wiring / PDB", (0.90, 0.10, 0.15, 1), lambda n: "wire" in n or "pdb" in n),
    ("Flight controller / avionics", (0.10, 0.75, 0.25, 1), lambda n: "fc_" in n),
    ("Structure (spars, rings, yokes, pads)", (0.45, 0.47, 0.50, 1),
     lambda n: any(k in n for k in ("spar", "ring", "yoke", "bracket", "pad_", "strut", "guard", "cap"))),
]

PLANE_COL = {
    "sagittal": ((0.10, 0.35, 0.95, 1), "joint, sagittal plane (pitch: fore/aft swing)"),
    "frontal": ((0.90, 0.12, 0.15, 1), "joint, frontal plane (roll: sideways swing)"),
    "gim_sag": ((0.00, 0.80, 0.85, 1), "fan gimbal, fore/aft tilt (thrust -> Fx / pitch / yaw)"),
    "gim_front": ((0.85, 0.15, 0.85, 1), "fan gimbal, lateral tilt (thrust -> Fy / roll)"),
}


def classify_joint(name, axis_local):
    gimbal = "_vec_" in name
    sag = abs(axis_local[1]) > 0.9
    if gimbal:
        return "gim_sag" if sag else "gim_front"
    return "sagittal" if sag else "frontal"


def font(sz, bold=False):
    try:
        return ImageFont.truetype(FONT_B if bold else FONT, sz)
    except OSError:
        return ImageFont.load_default()


def project(scene, pts, w, h):
    """World points -> pixel coords using the renderer's GL camera."""
    cam = scene.camera[0]
    pos, fwd, up = np.array(cam.pos), np.array(cam.forward), np.array(cam.up)
    right = np.cross(fwd, up)
    out = []
    for p in pts:
        v = np.asarray(p) - pos
        x, y, z = v @ right, v @ up, v @ fwd
        xs = x * cam.frustum_near / z
        ys = y * cam.frustum_near / z
        half_w = cam.frustum_width or (cam.frustum_top - cam.frustum_bottom) / 2 * w / h
        u = (xs - (cam.frustum_center - half_w)) / (2 * half_w)
        vv = (ys - cam.frustum_bottom) / (cam.frustum_top - cam.frustum_bottom)
        out.append((u * w, (1 - vv) * h))
    return out


def add_geom(scene, gtype, size, pos, mat, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, gtype, np.asarray(size, float), np.asarray(pos, float), np.asarray(mat, float).flatten(),
                        np.asarray(rgba, np.float32))
    scene.ngeom += 1


def rot_from_z(z):
    z = z / np.linalg.norm(z)
    a = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0])
    x = np.cross(a, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], 1)


def internals(model, d, cand, H, out):
    m = model
    # shells translucent, livery hidden, internal parts coloured by category
    for g in range(m.ngeom):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        m.geom_matid[g] = -1
        col = None
        for _, rgba, pred in CATEGORIES:
            if pred(n):
                col = rgba
                break
        if col is not None:
            m.geom_rgba[g] = col
            m.geom_group[g] = 3
        elif m.geom_group[g] == 1:
            m.geom_rgba[g] = (0.85, 0.87, 0.90, 0.13)
        elif n == "floor":
            m.geom_rgba[g] = (0.93, 0.94, 0.95, 1)
        elif "fasteners" in n:
            m.geom_rgba[g] = (0, 0, 0, 0)
    W, Hh = 700, 900
    r = mujoco.Renderer(m, Hh, W)
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[0] = opt.geomgroup[1] = opt.geomgroup[3] = 1
    opt.sitegroup[:] = 0
    cam = mujoco.MjvCamera()
    imgs = []
    for az, el in ((180, -8), (90, -8), (225, -18)):
        cam.lookat[:] = [0, 0, H * 0.5]
        cam.distance, cam.azimuth, cam.elevation = H * 1.9, az, el
        r.update_scene(d, cam, opt)
        imgs.append(Image.fromarray(r.render()))
    r.close()
    legend_w = 420
    sheet = Image.new("RGB", (W * 3 + legend_w, Hh + 50), (250, 250, 250))
    dr = ImageDraw.Draw(sheet)
    dr.text((12, 12), f"{cand} internals: massed components by type (shells translucent). SIMULATION MODEL",
            fill=(20, 20, 20), font=font(22, True))
    for i, (im, lb) in enumerate(zip(imgs, ("front", "side", "three-quarter"))):
        sheet.paste(im, (i * W, 50))
        dr.text((i * W + 10, 58), lb, fill=(60, 60, 60), font=font(18))
    x0, y = W * 3 + 20, 90
    dr.text((x0, y), "Component type", fill=(0, 0, 0), font=font(20, True))
    y += 40
    masses = {}
    for g in range(m.ngeom):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        for lb, _, pred in CATEGORIES:
            if pred(n):
                masses[lb] = masses.get(lb, 0) + 0
                break
    for lb, rgba, _ in CATEGORIES:
        c = tuple(int(255 * v) for v in rgba[:3])
        dr.rectangle([x0, y, x0 + 28, y + 28], fill=c, outline=(0, 0, 0))
        dr.text((x0 + 40, y + 3), lb, fill=(20, 20, 20), font=font(18))
        y += 42
    dr.rectangle([x0, y, x0 + 28, y + 28], fill=(225, 228, 232), outline=(0, 0, 0))
    dr.text((x0 + 40, y + 3), "Shells (translucent)", fill=(20, 20, 20), font=font(18))
    p = out / f"internals_{cand}.png"
    sheet.save(p)
    return p


def servo_axes(model, d, cand, H, design, out):
    m = model
    cat = load_catalog()
    W, Hh = 820, 1000
    r = mujoco.Renderer(m, Hh, W)
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[0] = opt.geomgroup[1] = opt.geomgroup[2] = 1
    opt.sitegroup[:] = 0
    # dim the body so the axis markers read clearly
    for g in range(m.ngeom):
        if m.geom_group[g] in (1, 2):
            m.geom_matid[g] = -1
            rgba = m.geom_rgba[g].copy()
            m.geom_rgba[g] = (0.82, 0.83, 0.85, 0.35)
    jrows = []
    for a in range(m.nu):
        if m.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT:
            continue
        j = m.actuator_trnid[a][0]
        jrows.append((mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j), j))
    cam = mujoco.MjvCamera()
    imgs = []
    views = (("front", 180, -5), ("side (right)", 90, -5), ("three-quarter", 215, -20))
    for title, az, el in views:
        cam.lookat[:] = [0, 0, H * 0.5]
        cam.distance, cam.azimuth, cam.elevation = H * 1.75, az, el
        r.update_scene(d, cam, opt)
        labels = []
        for name, j in jrows:
            anchor, axis = d.xanchor[j].copy(), d.xaxis[j].copy()
            kind = classify_joint(name, m.jnt_axis[j])
            if "_r_" in f"_{name}" or name.startswith("r_"):
                pass
            rgba = PLANE_COL[kind][0]
            gim = kind.startswith("gim")
            Lr = 0.075 if not gim else 0.06
            # axis arrow
            add_geom(r.scene, mujoco.mjtGeom.mjGEOM_ARROW, (0.006, 0.006, Lr), anchor, rot_from_z(axis), rgba)
            add_geom(r.scene, mujoco.mjtGeom.mjGEOM_ARROW, (0.006, 0.006, Lr), anchor, rot_from_z(-axis), rgba)
            # rotation arc (plane of motion) around the axis
            R = rot_from_z(axis)
            rad = 0.05 if not gim else 0.04
            for k in range(22):
                t = -0.75 * math.pi + k * 1.5 * math.pi / 21
                pt = anchor + rad * (R[:, 0] * math.cos(t) + R[:, 1] * math.sin(t))
                add_geom(r.scene, mujoco.mjtGeom.mjGEOM_SPHERE, (0.0045, 0, 0), pt, np.eye(3), rgba)
            labels.append((name, anchor, kind))
        img = Image.fromarray(r.render())
        dr = ImageDraw.Draw(img)
        if title == "three-quarter":
            pix = project(r.scene, [a for _, a, _ in labels], W, Hh)
            placed = []
            for (name, _, kind), (u, v) in sorted(zip(labels, pix), key=lambda t: t[1][1]):
                if not name.startswith("l_"):
                    continue
                v -= 10
                while any(abs(v - pv) < 20 and abs(u - pu) < 160 for pu, pv in placed):
                    v += 20
                placed.append((u, v))
                c = tuple(int(255 * x) for x in PLANE_COL[kind][0][:3])
                txt = name[2:].replace("pod_leg_vec_", "shin gimbal ").replace("pod_arm_vec_", "arm gimbal ")
                dr.text((u + 14, v), txt, fill=c, font=font(15, True), stroke_width=2, stroke_fill=(255, 255, 255))
        dr.text((10, 8), title, fill=(50, 50, 50), font=font(18))
        imgs.append(img)
    r.close()
    table_w = 640
    sheet = Image.new("RGB", (W * 3 + table_w, Hh + 60), (250, 250, 250))
    dr = ImageDraw.Draw(sheet)
    dr.text((12, 14), f"{cand}: servo motion axes. Arrow = rotation axis, arc = plane of motion (left-side labels; right side mirrored)",
            fill=(20, 20, 20), font=font(22, True))
    for i, im in enumerate(imgs):
        sheet.paste(im, (i * W, 60))
    x0, y = W * 3 + 16, 80
    for kind, (rgba, desc) in PLANE_COL.items():
        c = tuple(int(255 * v) for v in rgba[:3])
        dr.ellipse([x0, y + 4, x0 + 20, y + 24], fill=c)
        dr.text((x0 + 30, y + 3), desc, fill=(20, 20, 20), font=font(15))
        y += 32
    y += 14
    dr.text((x0, y), "Actuator (per side)        servo          range        motion", fill=(0, 0, 0), font=font(15, True))
    y += 28
    motion = {
        "hip_roll": "leg out/in (abduct)", "hip_pitch": "leg fwd/back", "knee": "shin pod fwd/back",
        "shoulder_pitch": "arm fwd/back", "shoulder_roll": "arm out/in", "elbow": "forearm pod bend",
        "pod_leg_vec_x": "shin fan tilts sideways", "pod_leg_vec_y": "shin fan tilts fwd/back",
        "pod_arm_vec_y": "arm pod tilts fwd/back",
    }
    for name, j in jrows:
        if not name.startswith("l_"):
            continue
        key = name[2:]
        kind = classify_joint(name, m.jnt_axis[j])
        if "_vec_" in name:
            sv = design["gimbal_choice"]
        else:
            sv = design["servo_choice"][key]
        lo, hi = np.degrees(m.jnt_range[j])
        c = tuple(int(255 * v) for v in PLANE_COL[kind][0][:3])
        dr.ellipse([x0, y + 4, x0 + 14, y + 18], fill=c)
        dr.text((x0 + 22, y), f"{key:<16s}", fill=(20, 20, 20), font=font(15))
        dr.text((x0 + 175, y), f"{sv['n']}x {sv['key']}", fill=(20, 20, 20), font=font(15))
        dr.text((x0 + 330, y), f"{lo:+.0f}..{hi:+.0f} deg", fill=(20, 20, 20), font=font(15))
        dr.text((x0 + 450, y), motion.get(key, ""), fill=(60, 60, 60), font=font(13))
        y += 26
    y += 14
    notes = [
        "No ankle: the shin pod is the lower leg (knee on its top centre).",
        "Arm pod gimbal = whole pod pivots at the wrist.",
        "Shin gimbal = fan cartridge tilts inside the fixed shell (+-14 deg cone).",
        "In flight, hip/knee/shoulder also re-aim the pods (slow, large-angle).",
    ]
    for t in notes:
        dr.text((x0, y), t, fill=(60, 60, 60), font=font(14))
        y += 22
    p = out / f"servo_axes_{cand}.png"
    sheet.save(p)
    return p


def main():
    cand = sys.argv[1] if len(sys.argv) > 1 else "E2_88cm_jp90_12s_65c"
    out = ROOT / "results" / "m0"
    path, design = generate(cand)
    H = design["H"]
    for fn in (internals, servo_axes):
        m = mujoco.MjModel.from_xml_path(str(path))
        d = mujoco.MjData(m)
        d.qpos[:] = pose_qpos(m, standing_pose(design))
        mujoco.mj_forward(m, d)
        p = fn(m, d, cand, H, out) if fn is internals else fn(m, d, cand, H, design, out)
        print(p)


if __name__ == "__main__":
    main()
