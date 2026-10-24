"""Render an evaluation rollout (rollout*.npz) to MP4 on the full-livery model.

python scripts/render_rollout.py training/runs/hover_v1/rollout.npz --trial 0 --title "Hover (PPO)"
Exhaust plumes are drawn from each nozzle with length proportional to fan thrust. SIMULATION label.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import imageio.v2 as imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.build_mjcf import generate  # noqa: E402

CAND = "E2_88cm_jp90_12s_65c"
PODS = ("l_pod_arm", "r_pod_arm", "l_pod_leg", "r_pod_leg")
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def add_plume(scene, pos, axis, frac):
    if frac < 0.03 or scene.ngeom + 2 >= scene.maxgeom:
        return
    L = 0.08 + 0.32 * frac
    d = -axis / np.linalg.norm(axis)
    z = d
    a = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0])
    x = np.cross(a, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], 1)
    for k, (rad, alpha, col) in enumerate(((0.034, 0.30, (0.55, 0.75, 1.0)), (0.020, 0.55, (0.75, 0.85, 1.0)))):
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.array([rad * (0.6 + 0.6 * frac), L / 2, 0]),
                            pos + d * L / 2, R.flatten(), np.array([*col, alpha], np.float32))
        scene.ngeom += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--trial", type=int, default=0)
    ap.add_argument("--title", default="")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--w", type=int, default=960)
    ap.add_argument("--h", type=int, default=720)
    ap.add_argument("--azimuth", type=float, default=135)
    ap.add_argument("--tmax", type=float, default=44.6)
    a = ap.parse_args()
    data = np.load(a.npz)
    qpos = data["qpos"][:, a.trial]
    fanT = data["fanT"][:, a.trial] if "fanT" in data else np.zeros((len(qpos), 4))
    done = data["done"][:, a.trial] if "done" in data else np.zeros(len(qpos))
    T = len(qpos)
    if done.any():
        T = min(T, int(np.argmax(done > 0.5)) + 25)
    path, _ = generate(CAND)
    m = mujoco.MjModel.from_xml_path(str(path))
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, a.h, a.w)
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[0] = opt.geomgroup[1] = opt.geomgroup[2] = 1
    opt.sitegroup[:] = 0
    cam = mujoco.MjvCamera()
    sites = [m.site(f"nozzle_{p}").id for p in PODS]
    tsites = [m.site(f"thrust_{p}").id for p in PODS]
    dt_ctrl = 0.02
    stride = max(1, int(round(1 / (a.fps * dt_ctrl))))
    out = a.out or str(Path(a.npz).with_suffix("")) + f"_trial{a.trial}.mp4"
    font = ImageFont.truetype(FONT, 22)
    small = ImageFont.truetype(FONT, 16)
    look = qpos[0, :3].copy()
    writer = imageio.get_writer(out, fps=a.fps, codec="libx264", quality=7, macro_block_size=8)
    for t in range(0, T, stride):
        d.qpos[:] = qpos[t]
        mujoco.mj_forward(m, d)
        look = 0.9 * look + 0.1 * d.qpos[:3]
        cam.lookat[:] = [look[0], look[1], max(look[2], 0.45)]
        cam.distance, cam.azimuth, cam.elevation = 2.4, a.azimuth, -12
        r.update_scene(d, cam, opt)
        for i in range(4):
            axis = d.site_xmat[tsites[i]].reshape(3, 3)[:, 2]
            add_plume(r.scene, d.site_xpos[sites[i]], axis, float(fanT[t, i]) / a.tmax)
        img = Image.fromarray(r.render())
        dr = ImageDraw.Draw(img)
        dr.text((14, 10), a.title, fill=(20, 20, 30), font=font)
        dr.text((14, 40), f"t = {t * dt_ctrl:5.1f} s   z = {d.qpos[2]:.2f} m   thrust = "
                + " ".join(f"{x:4.1f}" for x in fanT[t]) + " N", fill=(30, 30, 40), font=small)
        dr.text((a.w - 190, 10), "SIMULATION", fill=(200, 30, 30), font=font)
        writer.append_data(np.asarray(img))
    writer.close()
    r.close()
    print(out)


if __name__ == "__main__":
    main()
