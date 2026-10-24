"""Hardware-constrained EDF humanoid environment (MJX / Warp), one physics core for every task.

Tasks: stand | walk | hover | transition | maneuver. All tasks use the SAME robot: identical
mass properties, kinematics, actuator envelopes and propulsion layout (models/design.py).

Physics beyond rigid-body MuJoCo, applied here every substep:
  * servos: PD on the joint target, torque clipped by a linear torque-speed envelope
    (peak = 0.7 x stall, zero torque at no-load speed when driving), thermal RMS tracked
  * EDFs: first-order spool lag, thrust along the actual (gimballed) fan axis, installed
    thrust limit, battery-voltage dependence, pack current limit, P ~ T^1.5 power model,
    residual reaction torque, bounded ground-effect multiplier near the floor
  * battery: 12S pack, linear OCV(SoC), internal resistance sag, SoC drain
Observations for the actor are restricted to plausible sensors (IMU, encoders, fan commands,
pack voltage, pad contact switches, rangefinder altitude and a noisy velocity estimate from
VIO/optical-flow); the critic gets privileged state.
"""
from __future__ import annotations

import functools
import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

try:
    import warp as _wp
    _wp.config.quiet = True
except Exception:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.build_mjcf import generate, hover_pose, pose_qpos, standing_pose  # noqa: E402
from models.design import G, load_catalog, val  # noqa: E402

CANDIDATE = "E2_88cm_jp90_12s_65c"
PODS = ("l_pod_arm", "r_pod_arm", "l_pod_leg", "r_pod_leg")
CMD_SIZE = 10   # [vx, vy, wz, z_ref_rel, vz_ref, ex, ey, flight_flag, ground_flag, phase]


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        task="stand",
        ctrl_dt=0.02,
        sim_dt=0.002,
        episode_length=500,
        impl="warp",
        naconmax=24 * 8192,
        njmax=128,
        action_scale=0.35,           # rad, body joints around the nominal pose
        throttle_center=0.60,        # action 0 -> this fraction of available thrust
        throttle_gain=0.50,
        fans_enabled=True,           # False: EDFs physically off (stand/walk benchmarks)
        noise_level=1.0,
        dr_level=1.0,                # domain randomisation strength
        push=config_dict.create(enable=True, interval=[3.0, 6.0], impulse=[0.0, 0.3]),   # delta-v, m/s
        cmd=config_dict.create(vx=[-0.15, 0.25], vy=[-0.1, 0.1], wz=[-0.4, 0.4],
                               z_hover=[0.35, 0.9], v_air=[-1.0, 1.0], resample=[2.0, 4.0],
                               zero_prob=0.25),
        transition=config_dict.create(stand_pre=[1.5, 3.0], climb=2.0, hover=5.0, descend=3.0, z=[0.4, 0.7],
                                      fail_prob=0.0, fail_descent=0.4),   # failsafe: one fan dies mid-hover
        reward=config_dict.create(scales=config_dict.create()),
        termination_tilt_deg=60.0,
        obs_no_velocity=False,       # ablation: no VIO/mocap - IMU, encoders and rangefinder only
        gait_hz=1.0,
        swing_height=0.04,
        swing_sigma=0.02,
        # M5 robustness sweep knobs (neutral defaults; applied on top of the random DR)
        sweep=config_dict.create(thrust_scale=1.0, fan0_scale=1.0, payload_frac=0.0, tau_extra=0.0,
                                 com_x=0.0, ge_a=-99.0, friction=1.0),
    )


TASK_REWARDS = {
    "stand": dict(alive=1.0, upright=1.0, height=1.0, still=0.5, pose=0.5, action_rate=-0.02, torque=-0.002,
                  thermal=-0.5, pod_flat=0.5, termination=-50.0, thrust_use=-1.0),
    "walk": dict(alive=0.5, track_v=2.0, track_wz=0.8, upright=1.0, height=0.5, air_time=1.5, single_support=0.3,
                 pose=0.2, action_rate=-0.02, torque=-0.002, thermal=-0.5, slip=-1.0, pod_flat=0.3,
                 termination=-50.0, thrust_use=-1.0, lateral_drift=0.0, feet_phase=1.5),
    "hover": dict(alive=1.0, alt=2.0, pos=1.5, vel=0.5, upright=1.0, yaw_rate=0.3, angvel=-0.02, action_rate=-0.02,
                  power=-0.1, gimbal=-0.05, pose=0.3, termination=-50.0, ground=-5.0, thermal=-0.5, thrust_sat=-0.5),
    "transition": dict(alive=1.0, alt=2.5, vz=1.0, pos=1.0, upright=1.5, angvel=-0.02, action_rate=-0.02,
                       power=-0.05, pose=0.3, ground_stand=1.0, impact=-2.0, termination=-50.0, thermal=-0.5,
                       idle_on_ground=0.5, thrust_sat=-0.5, ground_thrust=-4.0, sink_near_ground=0.0),
    "maneuver": dict(alive=1.0, track_v=2.5, alt=1.5, track_wz=0.5, upright=0.5, angvel=-0.02, action_rate=-0.02,
                     power=-0.05, pose=0.2, termination=-50.0, ground=-5.0, thermal=-0.5, thrust_sat=-0.5),
}


class EDFHumanoid(mjx_env.MjxEnv):
    def __init__(self, config: config_dict.ConfigDict = default_config(),
                 config_overrides: Optional[Dict[str, Any]] = None):
        config = config.copy_and_resolve_references()
        if config_overrides:
            config.update_from_flattened_dict(config_overrides)
        task = config.task
        # task defaults, with any explicitly overridden reward scales merged on top
        scales = dict(TASK_REWARDS[task])
        scales.update(config.reward.scales.to_dict())
        config.reward.scales = config_dict.ConfigDict(scales)
        super().__init__(config)
        self.task = task
        path, design = generate(CANDIDATE, train=True)
        self._design = design
        self._xml_path = str(path)
        self._mj_model = mujoco.MjModel.from_xml_path(self._xml_path)
        self._mj_model.opt.timestep = self.sim_dt
        if config.sweep.friction != 1.0:          # M5: pad-floor sliding friction scale
            self._mj_model.geom_friction[:, 0] *= config.sweep.friction
        self._mjx_model = mjx.put_model(self._mj_model, impl=config.impl)
        self._setup_constants()

    # ------------------------------------------------------------------ constants
    def _setup_constants(self):
        m, d = self._mj_model, self._design
        cat = load_catalog()
        rules = cat["rules"]
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in range(m.nu)]
        self.joint_names = names
        jids = [m.actuator_trnid[a][0] for a in range(m.nu)]
        self._qadr = jp.array([m.jnt_qposadr[j] for j in jids])
        self._vadr = jp.array([m.jnt_dofadr[j] for j in jids])
        self._is_gimbal = np.array(["_vec_" in n for n in names])
        self._is_body = ~self._is_gimbal
        tau_pk, w_nl, tau_cont = [], [], []
        for n in names:
            if "_vec_" in n:
                ch = d["gimbal_choice"]
            else:
                key = n[2:]
                ch = d["servo_choice"][key]
            s = cat["servo"][ch["key"]]
            tau_pk.append(ch["n"] * s["stall_nm"] * rules["servo_peak_frac"])
            tau_cont.append(ch["n"] * s["stall_nm"] * rules["servo_cont_frac"])
            w_nl.append(s["noload_rpm"] * 2 * math.pi / 60)
        self._tau_pk = jp.array(tau_pk)
        self._tau_cont = jp.array(tau_cont)
        self._w_nl = jp.array(w_nl)
        self._kp = self._tau_pk / 0.25
        self._kd = 0.04 * self._kp
        lo, hi = m.jnt_range[jids].T
        self._jlo, self._jhi = jp.array(lo), jp.array(hi)

        q_stand = pose_qpos(m, standing_pose(d))
        q_hover = pose_qpos(m, hover_pose(d), ground=False, height=0.5)
        self._q_stand = jp.array(q_stand)
        self._q_hover = jp.array(q_hover)
        self._z_stand = float(q_stand[2])
        self._pose_stand = jp.array(q_stand[np.array(self._qadr)])
        self._pose_hover = jp.array(q_hover[np.array(self._qadr)])
        self._vec_range = jp.array(hi)  # symmetric ranges for gimbals

        # fans
        self._site_thrust = jp.array([m.site(f"thrust_{p}").id for p in PODS])
        self._site_nozzle = jp.array([m.site(f"nozzle_{p}").id for p in PODS])
        self._fan_body = jp.array([m.site_bodyid[m.site(f"thrust_{p}").id] for p in PODS])
        edf = d["edf"]
        v_ref, a_ref, t_ref, p_ref = d["test_ref"]
        self._eta = rules["install_eff"]
        self._T_ref_bench = t_ref * G
        self._P_ref = p_ref
        self._cells = edf["cells"]
        self._v_cell_ref = rules["cell_v_loaded"]
        self._T_max_nom = d["T_inst_max"]                       # installed, 3.7 V/cell
        bat = d["bat"]
        n_par = d["cand"]["pack_parallel"]
        self._I_lim = bat["cap_ah"] * bat["c_cont"] * n_par      # A, pack continuous
        self._cap_ah = bat["cap_ah"] * n_par
        self._R_pack = 0.0025 * self._cells / n_par
        self._aux_w = rules["aux_power_w"]
        self._D_fan = edf["rotor_d_mm"] / 1000
        self._mass = float(m.body_subtreemass[1])
        self._torso_id = 1
        self._pad_geoms = jp.array([g for g in range(m.ngeom)
                                    if "padtip" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "")])
        self._pad_side = jp.array([0 if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(g)).startswith("l_") else 1
                                   for g in np.array(self._pad_geoms)])
        self._pad_r = 0.008
        self._lowbody_ids = jp.array([m.body(n).id for n in ("torso", "l_thigh", "r_thigh", "l_pod_arm", "r_pod_arm")])
        self.nu_joint = m.nu
        self._act_size = m.nu + 4

    # ------------------------------------------------------------------ properties
    @property
    def xml_path(self):
        return self._xml_path

    @property
    def action_size(self):
        return self._act_size

    @property
    def mj_model(self):
        return self._mj_model

    @property
    def mjx_model(self):
        return self._mjx_model

    # ------------------------------------------------------------------ reset
    def _make_data(self, qpos, qvel):
        data = mjx.make_data(self._mj_model, impl=self._config.impl, naconmax=self._config.naconmax,
                             njmax=self._config.njmax)
        return data.replace(qpos=qpos, qvel=qvel, ctrl=jp.zeros(self._mj_model.nu))

    def _sample_dr(self, rng):
        lvl = self._config.dr_level
        k = jax.random.split(rng, 10)
        u = lambda kk, a, b, shape=(): jax.random.uniform(kk, shape, minval=a, maxval=b)
        sw = self._config.sweep
        fan_sweep = jp.array([sw.fan0_scale, 1.0, 1.0, 1.0]) * sw.thrust_scale
        return dict(
            fan_scale=(1.0 + lvl * u(k[0], -0.10, 0.04, (4,))) * fan_sweep,   # installed thrust spread / loss
            ge_a=jp.where(sw.ge_a > -50, sw.ge_a, lvl * u(k[1], -0.15, 0.10)),   # ground effect sign unknown
            react=lvl * u(k[2], 0.0, 0.008),                          # residual reaction torque arm [m]
            tau_up=0.16 + lvl * u(k[3], -0.03, 0.06) + sw.tau_extra,  # spool time constants
            tau_dn=0.20 + lvl * u(k[4], -0.03, 0.08) + sw.tau_extra,
            payload=(lvl * u(k[5], -0.08, 0.10) + sw.payload_frac) * self._mass,   # +-10% mass as gravity force
            com_off=lvl * u(k[6], -0.02, 0.02, (2,)) + jp.array([sw.com_x, 0.0]),  # COM offset of that force (m)
            gyro_bias=lvl * u(k[7], -0.02, 0.02, (3,)),
            kp_scale=1.0 + lvl * u(k[8], -0.15, 0.15, (self.nu_joint,)),
            soc0=u(k[9], 0.55, 1.0),
        )

    def _sample_cmd(self, rng, task):
        c = self._config.cmd
        k = jax.random.split(rng, 6)
        cmd = jp.zeros(CMD_SIZE)
        if task == "walk":
            v = jp.array([jax.random.uniform(k[0], minval=c.vx[0], maxval=c.vx[1]),
                          jax.random.uniform(k[1], minval=c.vy[0], maxval=c.vy[1]),
                          jax.random.uniform(k[2], minval=c.wz[0], maxval=c.wz[1])])
            v = jp.where(jax.random.uniform(k[3]) < c.zero_prob, jp.zeros(3), v)
            cmd = cmd.at[0:3].set(v).at[8].set(1.0)
        elif task in ("hover", "maneuver"):
            z = jax.random.uniform(k[0], minval=c.z_hover[0], maxval=c.z_hover[1])
            cmd = cmd.at[3].set(z).at[7].set(1.0)
            if task == "maneuver":
                v = jp.array([jax.random.uniform(k[1], minval=c.v_air[0], maxval=c.v_air[1]),
                              jax.random.uniform(k[2], minval=c.v_air[0], maxval=c.v_air[1]),
                              jax.random.uniform(k[4], minval=-0.5, maxval=0.5)])
                v = jp.where(jax.random.uniform(k[3]) < c.zero_prob, jp.zeros(3), v)
                cmd = cmd.at[0:3].set(v)
        else:  # stand / transition (transition profile is generated from time)
            cmd = cmd.at[8].set(1.0)
        return cmd

    def reset(self, rng: jax.Array) -> mjx_env.State:
        task = self.task
        rng, k_q, k_v, k_dr, k_cmd, k_push, k_tr = jax.random.split(rng, 7)
        dr = self._sample_dr(k_dr)
        cmd = self._sample_cmd(k_cmd, task)
        nq = self._mj_model.nq
        if task in ("hover", "maneuver"):
            qpos = self._q_hover
            z0 = self._z_stand + cmd[3] + jax.random.uniform(k_q, minval=-0.1, maxval=0.1)
            qpos = qpos.at[2].set(z0)
            # attitude perturbation up to ~10 deg
            ax = jax.random.normal(k_q, (3,)) * jp.array([0.08, 0.08, 0.3])
            quat = _quat_from_rotvec(ax)
            qpos = qpos.at[3:7].set(quat)
            qvel = jax.random.normal(k_v, (self._mj_model.nv,)) * 0.0
            qvel = qvel.at[0:3].set(jax.random.normal(k_v, (3,)) * 0.15).at[3:6].set(jax.random.normal(k_q, (3,)) * 0.2)
            fan_T = jp.full(4, self._mass * G / 4)
        else:
            qpos = self._q_stand
            noise = jax.random.uniform(k_q, (self.nu_joint,), minval=-0.05, maxval=0.05)
            noise = jp.where(self._is_gimbal, 0.0, noise)
            qpos = qpos.at[self._qadr].add(noise)
            qvel = jp.zeros(self._mj_model.nv)
            fan_T = jp.zeros(4)
        data = self._make_data(qpos, qvel)
        data = mjx.forward(self._mjx_model, data)
        z_tr = jax.random.uniform(k_tr, minval=self._config.transition.z[0], maxval=self._config.transition.z[1])
        t_pre = jax.random.uniform(k_tr, minval=self._config.transition.stand_pre[0],
                                   maxval=self._config.transition.stand_pre[1])
        info = dict(
            rng=rng, step=jp.array(0), cmd=cmd, dr=dr, fan_T=fan_T, soc=dr["soc0"],
            v_pack=self._cells * (3.5 + 0.7 * dr["soc0"]), i_pack=jp.array(0.0),
            last_act=jp.zeros(self._act_size), last_last_act=jp.zeros(self._act_size),
            target=jp.where(self.task in ("hover", "maneuver"), self._pose_hover, self._pose_stand),
            tau_sq=jp.zeros(self.nu_joint), push_t=jax.random.uniform(k_push, minval=self._config.push.interval[0],
                                                                     maxval=self._config.push.interval[1]),
            air_time=jp.zeros(2), last_contact=jp.ones(2, bool), xy_ref=qpos[0:2], yaw_ref=jp.array(0.0),
            cmd_t=jp.array(3.0), z_tr=z_tr, t_pre=t_pre, max_impact=jp.array(0.0), was_air=jp.array(False),
            peak_act_ratio=jp.array(0.0), sat_run=jp.array(0.0), max_sat_s=jp.array(0.0),
            gait_phase=jp.array(0.0), pod_xy_prev=self._pod_xy(data),
            fail_idx=jax.random.randint(k_tr, (), 0, 4),
            fail_t=jp.where(jax.random.uniform(jax.random.fold_in(k_tr, 1)) < self._config.transition.fail_prob,
                            t_pre + self._config.transition.climb
                            + jax.random.uniform(jax.random.fold_in(k_tr, 2), minval=0.5, maxval=4.0), 1e6),
        )
        metrics = {k: jp.array(0.0) for k in self._config.reward.scales.keys()}
        metrics.update(dict(alt_err=jp.array(0.0), tilt_deg=jp.array(0.0), power_kw=jp.array(0.0),
                            soc=jp.array(0.0), vel_err=jp.array(0.0), thermal_max=jp.array(0.0)))
        obs = self._obs(data, info, jax.random.PRNGKey(0))
        return mjx_env.State(data, obs, jp.array(0.0), jp.array(0.0), metrics, info)

    # ------------------------------------------------------------------ command schedules
    def _transition_profile(self, info):
        t = info["step"] * self.dt
        tc = self._config.transition
        t1 = info["t_pre"]
        t2 = t1 + tc.climb
        t3 = t2 + tc.hover
        t4 = t3 + tc.descend
        z = info["z_tr"]
        zr = jp.where(t < t1, 0.0,
             jp.where(t < t2, z * (t - t1) / tc.climb,
             jp.where(t < t3, z,
             jp.where(t < t4, z * (1 - (t - t3) / tc.descend), 0.0))))
        vz = jp.where((t >= t1) & (t < t2), z / tc.climb, jp.where((t >= t3) & (t < t4), -z / tc.descend, 0.0))
        flight = ((t >= t1) & (t < t4)).astype(float)
        phase = jp.clip(t / (t4 + 10.0), 0, 1)
        # failsafe: after a fan failure, descend from the commanded altitude at fail_descent m/s
        tf = info["fail_t"]
        failed = t >= tf
        z_at_fail = jp.where(tf < t2, z * (tf - t1) / tc.climb, z)
        zr_f = jp.maximum(z_at_fail - tc.fail_descent * (t - tf), 0.0)
        zr = jp.where(failed, zr_f, zr)
        vz = jp.where(failed, jp.where(zr_f > 0, -tc.fail_descent, 0.0), vz)
        flight = jp.where(failed, (zr_f > 0).astype(float), flight)
        return zr, vz, flight, phase

    def _update_cmd(self, info):
        cmd = info["cmd"]
        if self.task == "walk":
            moving = jp.linalg.norm(cmd[0:3]) > 0.05
            ph = jp.where(moving, info["gait_phase"] + 2 * jp.pi * self._config.gait_hz * self.dt, 0.0)
            ph = jp.mod(ph, 2 * jp.pi)
            info = {**info, "gait_phase": ph}
            cmd = cmd.at[5].set(jp.sin(ph) * moving).at[6].set(jp.cos(ph) * moving)
        if self.task == "transition":
            zr, vz, flight, phase = self._transition_profile(info)
            failed = (info["step"] * self.dt >= info["fail_t"]).astype(float)
            cmd = cmd.at[3].set(zr).at[4].set(vz).at[7].set(flight).at[8].set(1 - flight).at[9].set(phase)
            cmd = cmd.at[5].set(failed)          # fan-failure flag (ESC current / RPM telemetry)
            return cmd, info
        if self.task in ("walk", "maneuver"):
            rng, k1, k2 = jax.random.split(info["rng"], 3)
            t = info["cmd_t"] - self.dt
            new = self._sample_cmd(k1, self.task)
            if self.task == "maneuver":
                new = new.at[3].set(cmd[3])        # keep altitude, change velocity
            resample = t <= 0
            cmd = jp.where(resample, new, cmd)
            t = jp.where(resample, jax.random.uniform(k2, minval=self._config.cmd.resample[0],
                                                      maxval=self._config.cmd.resample[1]), t)
            info = {**info, "rng": rng, "cmd_t": t}
        return cmd, info

    # ------------------------------------------------------------------ step
    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        cfg = self._config
        info = dict(state.info)
        dr = info["dr"]
        rng, k_push, k_obs = jax.random.split(info["rng"], 3)
        info["rng"] = rng
        cmd, info = self._update_cmd(info)
        info["cmd"] = cmd
        action = jp.clip(action, -1.0, 1.0)

        # joint targets
        a_j = action[: self.nu_joint]
        base = jp.where(cmd[7] > 0.5, self._pose_hover, self._pose_stand) if self.task == "transition" else \
            (self._pose_hover if self.task in ("hover", "maneuver") else self._pose_stand)
        tgt_body = base + cfg.action_scale * a_j
        tgt_gim = a_j * self._vec_range
        target = jp.where(self._is_gimbal, tgt_gim, tgt_body)
        target = jp.clip(target, self._jlo, self._jhi)

        # throttle -> requested thrust, limited by pack current
        if cfg.fans_enabled:
            thr = jp.clip(cfg.throttle_center + cfg.throttle_gain * action[self.nu_joint:], 0.0, 1.0)
        else:
            thr = jp.zeros(4)
        v_cell = info["v_pack"] / self._cells
        T_avail = self._T_max_nom * (v_cell / self._v_cell_ref) ** 1.7 * dr["fan_scale"]
        dead = (info["step"] * self.dt >= info["fail_t"]) & (jp.arange(4) == info["fail_idx"])
        T_avail = jp.where(dead, 0.0, T_avail)
        T_req = thr * T_avail
        P_req = self._fan_power(T_req)
        I_req = (P_req.sum() + self._aux_w) / info["v_pack"]
        scale = jp.minimum(1.0, (self._I_lim / jp.maximum(I_req, 1e-3)) ** (1 / 1.5))
        T_cmd = T_req * scale

        # push disturbance (impulse spread over one control step)
        push_t = info["push_t"] - self.dt
        do_push = (push_t <= 0) & cfg.push.enable
        kp1, kp2 = jax.random.split(k_push)
        ang = jax.random.uniform(kp1, minval=0, maxval=2 * jp.pi)
        mag = jax.random.uniform(kp2, minval=cfg.push.impulse[0], maxval=cfg.push.impulse[1]) * self._mass
        push_f = jp.where(do_push, mag / self.dt, 0.0) * jp.array([jp.cos(ang), jp.sin(ang), 0.0])
        info["push_t"] = jp.where(do_push, jax.random.uniform(kp1, minval=cfg.push.interval[0],
                                                              maxval=cfg.push.interval[1]), push_t)

        data = state.data
        fan_T = info["fan_T"]
        tau_sq = info["tau_sq"]
        kp = self._kp * dr["kp_scale"]

        def substep(carry, _):
            data, fan_T, tau_sq, peak = carry
            q = data.qpos[self._qadr]
            qd = data.qvel[self._vadr]
            tau = kp * (target - q) - self._kd * qd
            driving = tau * qd > 0
            lim = jp.where(driving, self._tau_pk * jp.clip(1 - jp.abs(qd) / self._w_nl, 0.0, 1.0), self._tau_pk)
            tau = jp.clip(tau, -lim, lim)
            peak = jp.maximum(peak, jp.max(jp.abs(tau) / self._tau_pk))
            tau_sq = tau_sq + (tau ** 2 - tau_sq) * self.sim_dt / 20.0      # 20 s thermal time constant
            # fan spool dynamics
            tc = jp.where(T_cmd > fan_T, dr["tau_up"], dr["tau_dn"])
            fan_T = fan_T + (T_cmd - fan_T) * self.sim_dt / tc
            xfrc = self._thrust_wrench(data, fan_T, dr) + self._ext_wrench(data, dr, push_f)
            data = data.replace(ctrl=tau, xfrc_applied=xfrc)
            data = mjx.step(self._mjx_model, data)
            return (data, fan_T, tau_sq, peak), None

        (data, fan_T, tau_sq, peak), _ = jax.lax.scan(substep, (data, fan_T, tau_sq, jp.array(0.0)), (),
                                                       self.n_substeps)

        # battery
        P_fans = self._fan_power(fan_T)
        I = (P_fans.sum() + self._aux_w) / info["v_pack"]
        soc = jp.clip(info["soc"] - I * self.dt / 3600 / self._cap_ah, 0, 1)
        v_pack = self._cells * (3.5 + 0.7 * soc) - I * self._R_pack
        sat_run = jp.where(peak > 0.98, info["sat_run"] + self.dt, 0.0)   # continuous torque-limit time
        info.update(fan_T=fan_T, soc=soc, v_pack=v_pack, i_pack=I, tau_sq=tau_sq, target=target,
                    peak_act_ratio=jp.maximum(info["peak_act_ratio"], peak), sat_run=sat_run,
                    max_sat_s=jp.maximum(info["max_sat_s"], sat_run))

        # contacts (pad tip within 2 mm of floor)
        pad_z = data.geom_xpos[self._pad_geoms, 2] - self._pad_r
        pad_c = pad_z < 0.002
        contact = jp.array([jp.any(pad_c & (self._pad_side == 0)), jp.any(pad_c & (self._pad_side == 1))])

        reward_terms, done, info = self._rewards(data, action, info, contact, pad_z, P_fans, thr, scale)
        rew = sum(jp.asarray(reward_terms[k], jp.float32) * cfg.reward.scales[k]
                  for k in reward_terms if k in cfg.reward.scales)
        rew = rew * self.dt

        info["last_last_act"] = info["last_act"]
        info["last_act"] = action
        info["step"] = info["step"] + 1
        obs = self._obs(data, info, k_obs)
        metrics = dict(state.metrics)
        for k in metrics:
            if k in reward_terms:
                metrics[k] = jp.asarray(reward_terms[k], jp.float32)
        metrics.update(self._extra_metrics(data, info, P_fans))
        done = done.astype(float)
        return mjx_env.State(data, obs, rew, done, metrics, info)

    def _pod_xy(self, data):
        xy = data.geom_xpos[self._pad_geoms, 0:2]
        return jp.stack([jp.mean(jp.where((self._pad_side == k)[:, None], xy, 0.0), 0) * 2 for k in (0, 1)])

    def _pod_h(self, data):
        z = data.geom_xpos[self._pad_geoms, 2] - self._pad_r
        return jp.stack([jp.min(jp.where(self._pad_side == k, z, 1.0)) for k in (0, 1)])

    # ------------------------------------------------------------------ physics helpers
    def _fan_power(self, T):
        Tb = jp.maximum(T, 0.0) / self._eta
        return self._P_ref * (Tb / self._T_ref_bench) ** 1.5

    def _thrust_wrench(self, data, fan_T, dr):
        xfrc = jp.zeros((self._mj_model.nbody, 6))
        pos = data.site_xpos[self._site_thrust]
        axis = data.site_xmat[self._site_thrust][:, :, 2]
        noz_h = jp.maximum(data.site_xpos[self._site_nozzle][:, 2], 0.0)
        ge = 1.0 + dr["ge_a"] * jp.exp(-noz_h / self._D_fan)
        T = fan_T * ge
        F = T[:, None] * axis
        com = data.xipos[self._fan_body]
        tau = jp.cross(pos - com, F) - (dr["react"] * T)[:, None] * axis   # all fans spin the same way
        xfrc = xfrc.at[self._fan_body, 0:3].add(F)
        xfrc = xfrc.at[self._fan_body, 3:6].add(tau)
        return xfrc

    def _ext_wrench(self, data, dr, push_f):
        xfrc = jp.zeros((self._mj_model.nbody, 6))
        f = jp.array([0.0, 0.0, -dr["payload"] * G]) + push_f
        arm = jp.array([dr["com_off"][0], dr["com_off"][1], 0.0])
        xfrc = xfrc.at[self._torso_id, 0:3].add(f)
        xfrc = xfrc.at[self._torso_id, 3:6].add(jp.cross(arm, f))
        return xfrc

    # ------------------------------------------------------------------ observations
    def _obs(self, data, info, rng):
        lvl = self._config.noise_level
        k = jax.random.split(rng, 8)
        R = data.xmat[self._torso_id]
        grav = R.T @ jp.array([0.0, 0.0, -1.0])
        gyro = data.qvel[3:6]
        v_world = data.qvel[0:3]
        yaw = jp.arctan2(R[1, 0], R[0, 0])
        cy, sy = jp.cos(yaw), jp.sin(yaw)
        v_head = jp.array([cy * v_world[0] + sy * v_world[1], -sy * v_world[0] + cy * v_world[1], v_world[2]])
        q = data.qpos[self._qadr]
        qd = data.qvel[self._vadr]
        z = data.qpos[2] - self._z_stand
        cmd = info["cmd"]
        # position error to the hold setpoint, heading frame (VIO / mocap estimate)
        exy = info["xy_ref"] - data.qpos[0:2]
        e_head = jp.array([cy * exy[0] + sy * exy[1], -sy * exy[0] + cy * exy[1]])
        n = lambda kk, s, shape: lvl * s * jax.random.normal(kk, shape)
        pad_z = data.geom_xpos[self._pad_geoms, 2] - self._pad_r
        contact = jp.array([jp.any((pad_z < 0.002) & (self._pad_side == 0)),
                            jp.any((pad_z < 0.002) & (self._pad_side == 1))]).astype(float)
        state = jp.concatenate([
            gyro + info["dr"]["gyro_bias"] + n(k[0], 0.05, (3,)),
            grav + n(k[1], 0.03, (3,)),
            (v_head + n(k[2], 0.08, (3,))) * (0.0 if self._config.obs_no_velocity else 1.0),   # VIO / flow
            jp.array([z]) + n(k[3], 0.02, (1,)),                # rangefinder / baro altitude
            (e_head + n(k[4], 0.05, (2,))) * (0.0 if self._config.obs_no_velocity else 1.0),
            q - jp.where(self._is_gimbal, 0.0, self._pose_stand) + n(k[5], 0.01, (self.nu_joint,)),
            qd * 0.1 + n(k[6], 0.05, (self.nu_joint,)),
            info["last_act"],
            info["fan_T"] / self._T_max_nom,
            jp.array([info["v_pack"] / (self._cells * 3.7) - 1.0]),
            contact,
            cmd,
        ])
        priv = jp.concatenate([
            state,
            gyro, v_world, data.qpos[0:3] - jp.array([0.0, 0.0, self._z_stand]),
            data.qpos[3:7], q, qd * 0.1,
            info["fan_T"] / self._T_max_nom, pad_z, jp.array([info["dr"]["ge_a"], info["dr"]["react"] * 100,
                                                            info["dr"]["payload"] / self._mass]),
            info["dr"]["fan_scale"], jp.sqrt(info["tau_sq"]) / self._tau_cont,
        ])
        return {"state": state, "privileged_state": priv}

    # ------------------------------------------------------------------ rewards
    def _rewards(self, data, action, info, contact, pad_z, P_fans, thr, scale):
        task = self.task
        cmd = info["cmd"]
        R = data.xmat[self._torso_id]
        up = R[2, 2]
        tilt = jp.arccos(jp.clip(up, -1, 1))
        z = data.qpos[2] - self._z_stand
        v = data.qvel[0:3]
        w = data.qvel[3:6]
        yaw = jp.arctan2(R[1, 0], R[0, 0])
        cy, sy = jp.cos(yaw), jp.sin(yaw)
        v_head = jp.array([cy * v[0] + sy * v[1], -sy * v[0] + cy * v[1], v[2]])
        q = data.qpos[self._qadr]
        body_err = jp.where(self._is_gimbal, 0.0, q - jp.where(cmd[7] > 0.5, self._pose_hover, self._pose_stand))
        thermal = jp.max(jp.sqrt(info["tau_sq"]) / self._tau_cont)
        act_rate = jp.sum((action - info["last_act"]) ** 2)
        P_kw = P_fans.sum() / 1000
        r = {}
        r["alive"] = 1.0
        r["upright"] = jp.exp(-(tilt / 0.2) ** 2)
        r["action_rate"] = act_rate
        r["torque"] = 0.0
        r["thermal"] = jp.maximum(thermal - 1.0, 0.0)
        r["pose"] = jp.exp(-jp.sum(body_err ** 2) / 0.3)
        r["angvel"] = jp.sum(w ** 2)
        r["thrust_sat"] = jp.mean(thr > 0.98) + (1 - scale)
        pod_axes = data.site_xmat[self._site_nozzle[2:]][:, 2, 2]          # shin pod vertical alignment
        r["pod_flat"] = jp.mean(jp.exp(-((1 - pod_axes) / 0.01) ** 2) * contact)
        r["thrust_use"] = jp.mean(info["fan_T"]) / self._T_max_nom

        fallen = (up < jp.cos(jp.deg2rad(self._config.termination_tilt_deg))) | (data.qpos[2] < 0.25)
        low_body = jp.any(data.xpos[self._lowbody_ids, 2] < 0.06)
        nan = jp.any(jp.isnan(data.qpos)) | jp.any(jp.abs(data.qvel) > 200)
        done = fallen | low_body | nan

        if task == "stand":
            r["height"] = jp.exp(-(z / 0.05) ** 2)
            r["still"] = jp.exp(-jp.sum(v ** 2) / 0.02)
        if task == "walk":
            ev = v_head[0:2] - cmd[0:2]
            r["track_v"] = jp.exp(-jp.sum(ev ** 2) / 0.01)
            r["track_wz"] = jp.exp(-(w[2] - cmd[2]) ** 2 / 0.1)
            r["height"] = jp.exp(-((z + 0.02) / 0.05) ** 2)
            first = contact & ~info["last_contact"]
            air = info["air_time"] + self.dt
            moving = jp.linalg.norm(cmd[0:3]) > 0.05
            r["air_time"] = jp.sum((jp.clip(air, 0, 0.5) - 0.2) * first) * moving
            r["single_support"] = (jp.sum(contact) == 1) * moving + (jp.sum(contact) == 2) * (1 - moving)
            pod_xy = self._pod_xy(data)
            pod_v = jp.linalg.norm(pod_xy - info["pod_xy_prev"], axis=-1) / self.dt
            info["pod_xy_prev"] = pod_xy
            r["slip"] = jp.sum(pod_v * contact)                       # m/s of sliding while in contact
            # gait clock: left swings when sin(phase) > 0.3, right half a cycle later
            ph = info["gait_phase"]
            sw = jp.array([jp.sin(ph), jp.sin(ph + jp.pi)])
            h_des = self._config.swing_height * jp.clip((sw - 0.3) / 0.7, 0.0, 1.0) * moving
            r["feet_phase"] = jp.exp(-jp.sum((self._pod_h(data) - h_des) ** 2) / self._config.swing_sigma ** 2)
            r["lateral_drift"] = 0.0
            info["air_time"] = jp.where(contact, 0.0, air)
            info["last_contact"] = contact
            done = done
        if task in ("hover", "maneuver"):
            alt_err = z - cmd[3]
            # fine + coarse kernels: keep a gradient when far from the setpoint (hover_v1 drifted 3.8 m)
            r["alt"] = 0.7 * jp.exp(-(alt_err / 0.08) ** 2) + 0.3 * jp.exp(-(alt_err / 0.4) ** 2)
            exy = info["xy_ref"] - data.qpos[0:2]
            r["pos"] = 0.6 * jp.exp(-jp.sum(exy ** 2) / 0.05) + 0.4 * jp.exp(-jp.sum(exy ** 2) / 1.0)
            r["vel"] = jp.exp(-jp.sum(v ** 2) / 0.1)
            r["yaw_rate"] = jp.exp(-w[2] ** 2 / 0.1)
            r["power"] = P_kw / 10
            gim = jp.where(self._is_gimbal, q / self._vec_range, 0.0)
            r["gimbal"] = jp.sum(gim ** 2)
            r["ground"] = jp.any(pad_z < 0.01).astype(float)
            done = done | (pad_z.min() < 0.0) | (jp.abs(alt_err) > 1.5)
            if task == "maneuver":
                ev = v_head[0:2] - cmd[0:2]
                r["track_v"] = jp.exp(-jp.sum(ev ** 2) / 0.05)
                r["track_wz"] = jp.exp(-(w[2] - cmd[2]) ** 2 / 0.1)
                # moving setpoint for position: integrate command so stopping returns to hold
                xy_ref = data.qpos[0:2]
                info["xy_ref"] = xy_ref
                done = (up < jp.cos(jp.deg2rad(70.0))) | low_body | nan | (pad_z.min() < 0.0) | (jp.abs(alt_err) > 1.5)
        if task == "transition":
            alt_err = z - cmd[3]
            r["alt"] = 0.7 * jp.exp(-(alt_err / 0.08) ** 2) + 0.3 * jp.exp(-(alt_err / 0.4) ** 2)
            r["vz"] = jp.exp(-(v[2] - cmd[4]) ** 2 / 0.05)
            exy = info["xy_ref"] - data.qpos[0:2]
            r["pos"] = 0.6 * jp.exp(-jp.sum(exy ** 2) / 0.05) + 0.4 * jp.exp(-jp.sum(exy ** 2) / 1.0)
            r["power"] = P_kw / 10
            ground_phase = cmd[8] > 0.5
            both = jp.all(contact)
            r["ground_stand"] = ground_phase * both * r["upright"]
            r["idle_on_ground"] = ground_phase * both * (1 - jp.mean(info["fan_T"]) / self._T_max_nom)
            # after the landing phase the stand must be unsupported: penalise residual thrust (fraction of weight)
            post_land = (cmd[9] > 0) & ground_phase & (info["step"] * self.dt > info["t_pre"] + 1.0)
            r["ground_thrust"] = post_land * jp.sum(info["fan_T"]) / (self._mass * G)
            airborne = ~jp.any(contact)
            touchdown = jp.any(contact) & info["was_air"]
            impact = jp.where(touchdown, jp.abs(v[2]), 0.0)
            r["impact"] = jp.maximum(impact - 0.3, 0.0) ** 2
            # soft-landing shaping: in the last 15 cm above the floor while descending, sink rate <= 0.3 m/s
            pod_low = jp.min(pad_z)
            r["sink_near_ground"] = ((pod_low < 0.15) & airborne & (cmd[4] < 0)) * jp.maximum(-v[2] - 0.3, 0.0)
            info["max_impact"] = jp.maximum(info["max_impact"], impact)
            info["was_air"] = airborne
        r["termination"] = done.astype(float)
        return r, done, info

    def _extra_metrics(self, data, info, P_fans):
        R = data.xmat[self._torso_id]
        tilt = jp.rad2deg(jp.arccos(jp.clip(R[2, 2], -1, 1)))
        z = data.qpos[2] - self._z_stand
        yaw = jp.arctan2(R[1, 0], R[0, 0])
        cy, sy = jp.cos(yaw), jp.sin(yaw)
        v = data.qvel[0:2]
        v_head = jp.array([cy * v[0] + sy * v[1], -sy * v[0] + cy * v[1]])     # commands are heading-frame
        return dict(alt_err=jp.abs(z - info["cmd"][3]), tilt_deg=tilt, power_kw=P_fans.sum() / 1000,
                    soc=info["soc"], vel_err=jp.linalg.norm(v_head - info["cmd"][0:2]),
                    thermal_max=jp.max(jp.sqrt(info["tau_sq"]) / self._tau_cont))


def _quat_from_rotvec(v):
    a = jp.linalg.norm(v) + 1e-9
    ax = v / a
    return jp.concatenate([jp.array([jp.cos(a / 2)]), ax * jp.sin(a / 2)])


def make(task: str, **overrides):
    cfg = default_config()
    failsafe = task == "failsafe"
    if failsafe:
        task = "transition"
        cfg.transition.fail_prob = 1.0
    cfg.task = task
    lengths = dict(stand=500, walk=1000, hover=1000, transition=1300, maneuver=1000)
    cfg.episode_length = lengths[task]
    if task in ("stand", "walk"):
        cfg.fans_enabled = False
    if task == "transition":
        cfg.throttle_center, cfg.throttle_gain = 0.5, 0.5
    if task in ("hover", "maneuver", "transition"):
        cfg.push.impulse = [0.0, 0.5]
    for k, v in overrides.items():
        keys = k.split(".")
        node = cfg
        for kk in keys[:-1]:
            node = node[kk]
        node[keys[-1]] = v
    return EDFHumanoid(cfg)
