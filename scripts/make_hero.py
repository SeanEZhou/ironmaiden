"""Hero video: design turnaround -> stand -> walk -> hover -> takeoff/landing -> maneuvers.

python scripts/make_hero.py -> results/hero.mp4  (every clip labelled SIMULATION)
"""
from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "training" / "runs"
W, H, FPS = 960, 720, 25
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_R = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

CLIPS = [  # (path, title card, seconds to keep)
    (R / "stand_v2/rollout_trial0.mp4", "M1 Stand: 30 s with pushes, EDFs off\n100/100 trials", 8),
    (R / "walk_v4/rollout_trial0.mp4", "M2 Walk: 0.135 m/s, 42 mm step clearance, EDFs off\n100/100 trials reach 2 m", 10),
    (R / "hover_v7_wideDR_pos/rollout_trial0.mp4", "M3 Hover: 100/100, altitude RMS 2.3 cm, drift 8 cm\nrobust to -20% thrust (81%), 8.9 kW", 8),
    (R / "transition_v5_wideDR/rollout_trial0.mp4", "M4 Stand > liftoff > hover > land > stand\n92/100 full cycles, unassisted final stand", 26),
    (R / "maneuver_v1/rollout_trial0.mp4", "Stretch: fly fwd / back / sideways, stop, re-hover\n100/100, velocity error 0.07 m/s", 12),
    (R / "failsafe_v2_wideDR/rollout_trial0.mp4", "Failure test: one EDF dies mid-hover\ncontrolled 3-fan landing in 97/100 trials", 22),
]


def card(text, sub="Hardware-constrained MuJoCo simulation. Not a flight test.", secs=2.0):
    img = Image.new("RGB", (W, H), (18, 22, 34))
    d = ImageDraw.Draw(img)
    f, fs = ImageFont.truetype(FONT, 34), ImageFont.truetype(FONT_R, 20)
    y = H // 2 - 60
    for line in text.split("\n"):
        tw = d.textlength(line, font=f)
        d.text(((W - tw) / 2, y), line, fill=(240, 240, 245), font=f)
        y += 48
    tw = d.textlength(sub, font=fs)
    d.text(((W - tw) / 2, y + 20), sub, fill=(200, 80, 80), font=fs)
    return [np.asarray(img)] * int(secs * FPS)


def still(path, secs, title):
    im = Image.open(path).convert("RGB")
    im.thumbnail((W, H - 60))
    canvas = Image.new("RGB", (W, H), (250, 250, 250))
    canvas.paste(im, ((W - im.width) // 2, 60 + (H - 60 - im.height) // 2))
    ImageDraw.Draw(canvas).text((16, 16), title, fill=(20, 20, 30), font=ImageFont.truetype(FONT, 24))
    return [np.asarray(canvas)] * int(secs * FPS)


def main():
    out = ROOT / "results" / "hero.mp4"
    w = imageio.get_writer(out, fps=FPS, codec="libx264", quality=7, macro_block_size=8)
    for fr in card("EDF humanoid: ground-to-flight POC\n88 cm, 10.7 kg, 4 x 90 mm 12S EDFs", secs=3):
        w.append_data(fr)
    m0 = ROOT / "results" / "m0"
    for p, t in ((m0 / "turnaround_stand_E2_88cm_jp90_12s_65c.png", "Design turnaround (physics model)"),
                 (m0 / "internals_E2_88cm_jp90_12s_65c.png", "Every component massed: fans, ESCs, packs, servos, wiring"),
                 (m0 / "servo_axes_E2_88cm_jp90_12s_65c.png", "18 servos: joint axes + 2-axis shin / 1-axis arm fan gimbals")):
        if p.exists():
            for fr in still(p, 3.5, t):
                w.append_data(fr)
    for path, title, secs in CLIPS:
        if not path.exists():
            print("missing", path)
            continue
        for fr in card(title):
            w.append_data(fr)
        rd = imageio.get_reader(path)
        for i, fr in enumerate(rd):
            if i >= secs * FPS:
                break
            w.append_data(np.asarray(Image.fromarray(fr).resize((W, H))))
    w.close()
    print(out)


if __name__ == "__main__":
    main()
