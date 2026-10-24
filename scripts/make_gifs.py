"""README GIFs from the evaluation rollout videos -> docs/assets/gif_*.gif

python scripts/make_gifs.py
Video frames are 0.04 s apart (render stride 2 of the 50 Hz control loop).
"""
from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "training" / "runs"
OUT = ROOT / "docs" / "assets"
FRAME_DT = 0.04


def failure_frame(npz, trial=0):
    f = np.load(npz)["fanT"][:, trial]                      # [T, 4] at 50 Hz
    dead = int(np.argmin(f[-1]))                            # the failed fan ends at exactly zero thrust
    t = int(np.where(f[:, dead] > 20.0)[0].max())           # last instant it was still producing thrust
    return t // 2


CLIPS = {  # name: (video, start_s, end_s, frame_step)
    "stand": (R / "stand_v2/rollout_trial0.mp4", 2.0, 8.0, 3),
    "walk": (R / "walk_v4/rollout_trial0.mp4", 4.0, 12.0, 3),
    "hover": (R / "hover_v7_wideDR_pos/rollout_trial0.mp4", 0.5, 6.5, 3),
    "transition": (R / "transition_v5_wideDR/rollout_trial0.mp4", 0.0, 24.0, 6),    # played at 2x
    "maneuver": (R / "maneuver_v1/rollout_trial0.mp4", 0.0, 10.0, 3),
    "failsafe": (R / "failsafe_v2_wideDR/rollout_trial0.mp4", None, None, 3),
}


def global_palette(frames, colors=128):
    # One palette for the whole clip: per-frame palettes made the colored pods flicker to grey
    k = max(len(frames) // 16, 1)
    sample = np.concatenate([np.asarray(f) for f in frames[::k]], axis=0)
    return Image.fromarray(sample).quantize(colors=colors, method=Image.MEDIANCUT)


def main(width=720):
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (vid, t0, t1, step) in CLIPS.items():
        rd = imageio.get_reader(vid)
        frames = [f for f in rd]
        if name == "failsafe":
            k = failure_frame(vid.parent / "rollout.npz")
            a, b = max(k - int(1.5 / FRAME_DT), 0), min(k + int(4.5 / FRAME_DT), len(frames))
        else:
            a, b = int(t0 / FRAME_DT), min(int(t1 / FRAME_DT), len(frames))
        sel = frames[a:b:step]
        h = int(width * sel[0].shape[0] / sel[0].shape[1])
        rgb = [Image.fromarray(f).resize((width, h), Image.LANCZOS) for f in sel]
        pal = global_palette(rgb)
        imgs = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in rgb]
        dur = int(1000 * FRAME_DT * min(step, 3))     # real time (transition: 2x speed)
        p = OUT / f"gif_{name}.gif"
        imgs[0].save(p, save_all=True, append_images=imgs[1:], duration=dur, loop=0, optimize=True)
        print(p.name, len(imgs), "frames", f"{p.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
