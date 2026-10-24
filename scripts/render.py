"""Render front/side/back/three-quarter turnaround of a candidate (stand + hover pose).

python scripts/render.py E2_88cm_jp90_12s_65c [--internals]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.build_mjcf import generate, hover_pose, pose_qpos, standing_pose  # noqa: E402


def render_views(model, q, views, w=640, h=860, internals=False, lookat_z=None, dist=None):
    d = mujoco.MjData(model)
    d.qpos[:] = q
    mujoco.mj_forward(model, d)
    r = mujoco.Renderer(model, h, w)
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[0] = 1
    opt.geomgroup[1] = 0 if internals else 1
    opt.geomgroup[2] = 0 if internals else 1
    opt.geomgroup[3] = 1 if internals else 0
    opt.sitegroup[:] = 0
    cam = mujoco.MjvCamera()
    com = d.subtree_com[1]
    imgs = []
    for az, el in views:
        cam.lookat[:] = [com[0], com[1], lookat_z if lookat_z is not None else com[2] + 0.03]
        cam.distance = dist
        cam.azimuth = az
        cam.elevation = el
        r.update_scene(d, cam, opt)
        imgs.append(Image.fromarray(r.render()))
    r.close()
    return imgs


def sheet(imgs, labels, title, path):
    w, h = imgs[0].size
    pad = 44
    out = Image.new("RGB", (w * len(imgs), h + pad), (250, 250, 250))
    dr = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
        small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except OSError:
        font = small = ImageFont.load_default()
    dr.text((10, 10), title, fill=(20, 20, 20), font=font)
    for i, (im, lb) in enumerate(zip(imgs, labels)):
        out.paste(im, (i * w, pad))
        dr.text((i * w + 10, pad + 8), lb, fill=(40, 40, 40), font=small)
    out.save(path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate", nargs="?", default="E2_88cm_jp90_12s_65c")
    ap.add_argument("--out", default=str(ROOT / "results" / "m0"))
    args = ap.parse_args()
    path, d = generate(args.candidate)
    model = mujoco.MjModel.from_xml_path(str(path))
    H = d["H"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mass = model.body_subtreemass[1]
    views = [(180, -5), (90, -5), (0, -5), (225, -15)]
    labels = ["front", "side (right)", "back", "three-quarter"]
    q = pose_qpos(model, standing_pose(d))
    title = f"{args.candidate}  |  H={H:.2f} m  mass={mass:.2f} kg  pod OD={d['pod_od']*1000:.0f} mm  (MuJoCo physics model, SIMULATION)"
    imgs = render_views(model, q, views, lookat_z=H * 0.5, dist=H * 2.25)
    p1 = sheet(imgs, labels, title + "  - standing", out / f"turnaround_stand_{args.candidate}.png")
    q = pose_qpos(model, hover_pose(d), ground=False, height=0.35)
    imgs = render_views(model, q, views, lookat_z=H * 0.5 + 0.35, dist=H * 2.25)
    p2 = sheet(imgs, labels, title + "  - hover pose (gimbals centred)", out / f"turnaround_hover_{args.candidate}.png")
    print(p1, p2, sep="\n")   # colour-coded internals: scripts/render_diagrams.py


if __name__ == "__main__":
    main()
