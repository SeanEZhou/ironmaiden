"""Fixed evaluation suite (held-out seeds) for every task, using the milestone pass criteria.

python evaluation/evaluate.py --task hover --params training/runs/hover_v1/params.pkl [--controller classical]
Writes <run>/eval.json and <run>/rollout.npz (qpos + fan thrust of the first trials, for video).
Success criteria (spec section 7):
  stand      : 30 s unsupported, no fall, 100 randomised trials
  walk       : >= 2 m forward at 0.15 m/s command, no fall
  hover      : 20 s, |alt err| <= 0.15 m, roll/pitch <= 10 deg, no ground contact
  transition : stand -> liftoff -> 5 s hover -> descend -> both-pod touchdown (<= 0.6 m/s) -> 10 s stand
  maneuver   : no crash, mean horizontal velocity tracking error <= 0.3 m/s
"""
from __future__ import annotations

import argparse
import functools
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax  # noqa: E402
import jax.numpy as jp  # noqa: E402
import numpy as np  # noqa: E402
from brax.training.acme import running_statistics  # noqa: E402
from brax.training.agents.ppo import networks as ppo_networks  # noqa: E402

from envs.edf_env import make  # noqa: E402

EVAL_SEED = 12345     # held out: training seeds are small integers
EVAL_OVERRIDES = {
    "stand": {"episode_length": 1500, "push.impulse": [0.0, 0.15]},
    "walk": {"episode_length": 1500, "cmd.vx": [0.15, 0.15], "cmd.vy": [0.0, 0.0], "cmd.wz": [0.0, 0.0],
             "cmd.zero_prob": 0.0, "cmd.resample": [100.0, 100.0], "push.impulse": [0.0, 0.15]},
    "hover": {"episode_length": 1000, "push.enable": False},
    "transition": {"episode_length": 1300, "push.enable": False},
    "failsafe": {"episode_length": 1300, "push.enable": False},
    "maneuver": {"episode_length": 1000, "push.enable": False},
}


def load_policy(env, params_path, policy_sizes="512,256,128", value_sizes="512,256,128"):
    with open(params_path, "rb") as fh:
        params = pickle.load(fh)
    obs_size = env.observation_size
    nets = ppo_networks.make_ppo_networks(
        obs_size, env.action_size, preprocess_observations_fn=running_statistics.normalize,
        policy_hidden_layer_sizes=tuple(int(x) for x in policy_sizes.split(",")),
        value_hidden_layer_sizes=tuple(int(x) for x in value_sizes.split(",")),
        policy_obs_key="state", value_obs_key="privileged_state")
    inference = ppo_networks.make_inference_fn(nets)(params, deterministic=True)
    return lambda obs, rng: inference(obs, rng)[0]


def rollout(env, policy, n, steps, seed=EVAL_SEED, keep=4):
    reset = jax.jit(jax.vmap(env.reset))
    step = jax.jit(jax.vmap(env.step))
    pol = jax.jit(jax.vmap(policy))
    rng = jax.random.PRNGKey(seed)
    s = reset(jax.random.split(rng, n))
    rec = {k: [] for k in ("z", "x", "y", "tilt", "alt_err", "vx_head", "contact_l", "contact_r", "vz", "done",
                           "power_kw", "peak", "thermal", "soc", "fanT", "cmd_vx", "cmd_vy", "max_impact", "cmd_z",
                           "max_sat_s")}
    qpos_keep = []
    R_up = lambda s: s.data.xmat[:, 1, 2, 2]
    for t in range(steps):
        rng, k = jax.random.split(rng)
        a = pol(s.obs, jax.random.split(k, n))
        s = step(s, a)
        d = s.data
        pad_z = d.geom_xpos[:, env._pad_geoms, 2] - env._pad_r
        cl = jp.any((pad_z < 0.002) & (env._pad_side == 0), axis=1)
        cr = jp.any((pad_z < 0.002) & (env._pad_side == 1), axis=1)
        R = d.xmat[:, 1]
        yaw = jp.arctan2(R[:, 1, 0], R[:, 0, 0])
        vxh = jp.cos(yaw) * d.qvel[:, 0] + jp.sin(yaw) * d.qvel[:, 1]
        vals = dict(z=d.qpos[:, 2] - env._z_stand, x=d.qpos[:, 0], y=d.qpos[:, 1], tilt=s.metrics["tilt_deg"],
                    alt_err=s.metrics["alt_err"], vx_head=vxh, contact_l=cl, contact_r=cr, vz=d.qvel[:, 2],
                    done=s.done, power_kw=s.metrics["power_kw"], peak=s.info["peak_act_ratio"],
                    thermal=s.metrics["thermal_max"], soc=s.info["soc"], fanT=s.info["fan_T"],
                    cmd_vx=s.info["cmd"][:, 0], cmd_vy=s.info["cmd"][:, 1], max_impact=s.info["max_impact"],
                    cmd_z=s.info["cmd"][:, 3], max_sat_s=s.info["max_sat_s"])
        vals["vel_err"] = s.metrics["vel_err"]
        vals["pod_h"] = jp.stack([jp.min(jp.where(env._pad_side == k, pad_z, 1.0), axis=1) for k in (0, 1)], 1)
        xy = d.geom_xpos[:, env._pad_geoms, 0:2]
        vals["pod_xy"] = jp.stack([jp.mean(jp.where((env._pad_side == k)[None, :, None], xy, 0.0), 1) * 2
                                   for k in (0, 1)], 1)
        vals["wz"] = d.qvel[:, 5]
        vals["path"] = jp.linalg.norm(d.qvel[:, 0:2], axis=-1) * env.dt
        for k_ in ("vel_err", "pod_h", "pod_xy", "wz", "path"):
            rec.setdefault(k_, [])
        for k_, v in vals.items():
            rec[k_].append(np.asarray(v))
        qpos_keep.append(np.asarray(d.qpos[:keep]))
    out = {k: np.stack(v) for k, v in rec.items()}       # [T, N, ...]
    out["qpos_keep"] = np.stack(qpos_keep)
    return out


def first_done(done):
    T = done.shape[0]
    idx = np.where(done.any(0), done.argmax(0), T)
    return idx


def score(task, r, dt):
    T, N = r["done"].shape
    fd = first_done(r["done"] > 0.5)
    alive = np.arange(T)[:, None] < fd[None, :]
    res = dict(trials=int(N), steps=int(T), dt=dt)
    res["no_fall_rate"] = float((fd == T).mean())
    res["peak_actuator_ratio_max"] = float(r["peak"][-1].max())
    res["max_saturation_s_p95"] = float(np.percentile(r["max_sat_s"][-1], 95))   # longest continuous torque-limit
    res["thermal_ratio_max"] = float(np.where(alive, r["thermal"], 0).max())
    if task == "stand":
        succ = fd == T
        sway = np.array([np.std(r["x"][: fd[i], i]) + np.std(r["y"][: fd[i], i]) for i in range(N)])
        res.update(success_rate=float(succ.mean()), sway_cm_mean=float(sway.mean() * 100),
                   max_tilt_deg_p95=float(np.percentile(np.where(alive, r["tilt"], 0).max(0), 95)))
    elif task == "walk":
        dist = np.array([r["x"][max(fd[i] - 1, 0), i] - r["x"][0, i] for i in range(N)])
        succ = (dist >= 2.0) & (fd == T) | ((dist >= 2.0) & (fd > 0))
        succ = np.array([(r["x"][: fd[i], i] - r["x"][0, i] >= 2.0).any() for i in range(N)])
        speed = dist / (np.minimum(fd, T) * dt + 1e-9)
        # gait quality: a shuffle can satisfy the distance metric, so report stepping and slip
        c = r["pod_h"] < 0.003                                   # [T, N, 2]
        lifts = (np.diff(c.astype(int), axis=0) == -1).sum(0)     # [N, 2]
        clear = np.array([[r["pod_h"][: fd[i], i, k].max() for k in (0, 1)] for i in range(N)])
        dxy = np.linalg.norm(np.diff(r["pod_xy"], axis=0), axis=-1)   # [T-1, N, 2]
        slip = (dxy * c[1:]).sum(0).sum(-1)
        travel = dxy.sum(0).sum(-1) + 1e-9
        res.update(success_rate=float(succ.mean()), distance_m_mean=float(dist.mean()),
                   speed_mps_mean=float(speed.mean()), no_fall_rate=float((fd == T).mean()),
                   steps_per_s_per_pod=float(lifts.mean() / (T * dt)), pod_clearance_mm_mean=float(clear.mean() * 1000),
                   slip_fraction_of_travel=float((slip / travel).mean()),
                   yaw_rate_mean=float(np.mean([r["wz"][: fd[i], i].mean() for i in range(N)])),
                   path_length_m_mean=float(np.mean([r["path"][: fd[i], i].sum() for i in range(N)])))
    elif task == "hover":
        alt_ok = (np.abs(r["alt_err"]) <= 0.15) | ~alive
        tilt_ok = (r["tilt"] <= 10.0) | ~alive
        # allow 1 s for the initial transient from the perturbed start
        settle = int(1.0 / dt)
        succ = (fd == T) & alt_ok[settle:].all(0) & tilt_ok[settle:].all(0)
        drift = np.hypot(r["x"][-1] - r["x"][0], r["y"][-1] - r["y"][0])
        res.update(success_rate=float(succ.mean()), alt_err_rms_m=float(np.sqrt((r["alt_err"][settle:] ** 2).mean())),
                   tilt_deg_p95=float(np.percentile(r["tilt"][settle:], 95)), pos_drift_m_mean=float(drift.mean()),
                   hover_power_kw=float(r["power_kw"][settle:].mean()),
                   est_flight_time_s=float(0.75 * 266.4 / max(r["power_kw"][settle:].mean(), 1e-3) * 3.6))
    elif task == "transition":
        z = r["z"]
        cmdz = r["cmd_z"]
        cont = r["contact_l"] & r["contact_r"]
        hold = cmdz >= cmdz.max(0, keepdims=True) - 1e-4
        # 1 s settling allowance after the climb ramp ends (same as the hover eval), then 4 s judged
        hold_start = hold.argmax(0)
        hover_phase = hold & (np.arange(T)[:, None] >= hold_start[None, :] + int(1.0 / dt))
        alt_ok = np.where(hover_phase & alive, np.abs(z - cmdz) <= 0.15, True).all(0)
        lifted = ((~(r["contact_l"] | r["contact_r"])) & alive).any(0)
        last_10s = slice(T - int(10.0 / dt), T)
        landed = cont[last_10s].all(0)
        impact_ok = r["max_impact"][-1] <= 0.6
        # "unsupported" stand: fans must be (near) idle during the final 10 s, <10% of weight in total thrust
        mg = 10.7385 * 9.81
        stand_thrust = r["fanT"][last_10s].sum(-1).mean(0) / mg
        unassisted = stand_thrust < 0.10
        succ = (fd == T) & lifted & alt_ok & landed & impact_ok & unassisted
        res.update(unassisted_stand_rate=float(unassisted.mean()), final_stand_thrust_frac_mean=float(stand_thrust.mean()))
        res.update(success_rate=float(succ.mean()), liftoff_rate=float(lifted.mean()),
                   hover_alt_ok_rate=float(alt_ok.mean()), landed_rate=float(landed.mean()),
                   impact_mps_p95=float(np.percentile(r["max_impact"][-1], 95)), impact_ok_rate=float(impact_ok.mean()))
    elif task == "failsafe":
        # one random fan dies mid-hover: success = no tip-over, both pods down, impact <= 1.5 m/s
        cont = r["contact_l"] & r["contact_r"]
        landed = cont[T - int(5.0 / dt):].all(0)
        impact = r["max_impact"][-1]
        succ = (fd == T) & landed & (impact <= 1.5)
        res.update(success_rate=float(succ.mean()), landed_rate=float(landed.mean()),
                   impact_mps_p50=float(np.median(impact)), impact_mps_p95=float(np.percentile(impact, 95)),
                   soft_landing_rate=float((impact <= 1.5).mean()))
    elif task == "maneuver":
        # exclude a 1 s transient after every command change (step response, not steady tracking)
        chg = np.zeros_like(r["cmd_vx"], bool)
        chg[1:] = (np.abs(np.diff(r["cmd_vx"], axis=0)) + np.abs(np.diff(r["cmd_vy"], axis=0))) > 1e-6
        settle = np.zeros_like(chg)
        for k in range(int(1.0 / dt)):
            settle[k:] |= chg[: T - k]
        ve = np.where(alive & ~settle, r["vel_err"], np.nan)
        mean_ve = np.nanmean(ve, 0)
        succ = (fd == T) & (mean_ve <= 0.3)
        res.update(success_rate=float(succ.mean()), vel_err_mean=float(np.nanmean(mean_ve)),
                   alt_err_rms_m=float(np.sqrt(np.nanmean(np.where(alive, r["alt_err"], np.nan) ** 2))))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--params", default=None)
    ap.add_argument("--controller", default="ppo", help="ppo | classical")
    ap.add_argument("--out", default=None)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--env", default="{}")
    ap.add_argument("--policy", default="512,256,128")
    ap.add_argument("--value", default="512,256,128")
    a = ap.parse_args()
    ov = {**EVAL_OVERRIDES[a.task], **json.loads(a.env), "naconmax": 24 * 256}
    env = make(a.task, **ov)
    if a.controller == "classical":
        from controllers.classical import make_classical
        policy = make_classical(a.task, env)
    else:
        policy = load_policy(env, a.params, a.policy, a.value)
    r = rollout(env, policy, a.trials, env._config.episode_length)
    res = score(a.task, r, env.dt)
    res.update(task=a.task, controller=a.controller, params=a.params, env_overrides=ov)
    out = Path(a.out) if a.out else Path(a.params).parent
    out.mkdir(parents=True, exist_ok=True)
    tag = "" if a.controller == "ppo" else f"_{a.controller}"
    (out / f"eval{tag}.json").write_text(json.dumps(res, indent=1))
    np.savez_compressed(out / f"rollout{tag}.npz", qpos=r["qpos_keep"], fanT=r["fanT"][:, :4], z=r["z"][:, :4],
                        tilt=r["tilt"], alt_err=r["alt_err"], done=r["done"], power_kw=r["power_kw"],
                        vx=r["vx_head"], cmd_vx=r["cmd_vx"], cmd_vy=r["cmd_vy"], cmd_z=r["cmd_z"])
    print(json.dumps({k: v for k, v in res.items() if k != "env_overrides"}, indent=1))


if __name__ == "__main__":
    main()
