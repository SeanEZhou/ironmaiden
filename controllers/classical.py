"""Classical baselines (no learning), using only the deployable observation vector.

stand      : hold the nominal pose; hip pitch/roll feedback on IMU tilt.
hover      : cascaded PD. Outer loop -> desired body-frame force (gimbals give Fx/Fy directly, so the
             body stays level; this robot is fully actuated in 6 DOF), inner attitude PD -> torque;
             weighted pseudo-inverse of the 6x10 allocation Jacobian (4 thrusts + 6 gimbals) at trim.
transition : FSM on the commanded profile: ground -> idle fans + stand; flight -> hover controller
             tracking z_ref with vz feed-forward; touchdown (both pods in contact, z_ref = 0) -> idle.
maneuver   : hover controller with a velocity-tracking outer loop.
Stateless (no integrators): steady-state altitude error under thrust loss is a known limitation.
"""
from __future__ import annotations

import jax.numpy as jp
import numpy as np

from controllers.allocation import Allocator, PODS

# observation layout (envs/edf_env.py::_obs, actor "state")
GYRO, GRAV, VEL, Z, EXY = slice(0, 3), slice(3, 6), slice(6, 9), 9, slice(10, 12)
QERR = slice(12, 30)
VPACK, CONTACT, CMD = 74, slice(75, 77), slice(77, 87)


def _allocation(env):
    A = Allocator(env.mj_model, np.asarray(env._q_hover), env._T_max_nom)
    u0, _ = A.trim()
    J = A.jacobian(u0)                      # world == body frame at the level hover pose
    lo, hi = A.bounds()
    W = np.diag((hi - lo) ** 2)             # range-normalised minimum-effort pseudo-inverse
    M = W @ J.T @ np.linalg.inv(J @ W @ J.T)
    wtrim = A.wrench(u0)
    gim_act = [env.joint_names.index(n) for n in A.gimbal_names]
    lo = lo.copy()
    lo[: A.n_t] = 0.05 * env._T_max_nom
    return jp.array(u0), jp.array(M), jp.array(wtrim), jp.array(gim_act), A.n_t, jp.array(lo), jp.array(hi)


def make_classical(task, env):
    nj = env.nu_joint
    m = env._mass
    g = 9.81
    center, gain = env._config.throttle_center, env._config.throttle_gain
    rng_vec = env._vec_range
    is_g = env._is_gimbal
    joint = {n: i for i, n in enumerate(env.joint_names)}

    if task == "stand" or task == "walk":
        hp = jp.array([joint["l_hip_pitch"], joint["r_hip_pitch"]])
        hr = jp.array([joint["l_hip_roll"], joint["r_hip_roll"]])

        def stand(obs, rng):
            o = obs["state"]
            gb = o[GRAV]
            pitch, roll = gb[0], -gb[1]
            w = o[GYRO]
            a = jp.zeros(env.action_size)
            # lean forward -> hips extend (pitch the torso back over the pods)
            a = a.at[hp].set(jp.clip(1.5 * pitch + 0.15 * w[1], -1, 1))
            a = a.at[hr].set(jp.clip(jp.array([1.0, -1.0]) * (1.0 * roll + 0.1 * w[0]), -1, 1))
            a = a.at[nj:].set(-1.0)
            return a
        return stand

    u0, M, wtrim, gim_act, n_t, lo, hi = _allocation(env)
    I_diag = jp.array([0.50, 0.42, 0.10])
    wn_att, zeta = 5.0, 0.8
    Kp_att, Kd_att = I_diag * wn_att ** 2, I_diag * 2 * zeta * wn_att
    Kp_z, Kd_z = 12.0, 6.0
    Kp_xy, Kd_xy = 2.0, 2.8

    def hover_core(o, z_ref, vz_ff, v_ref_xy, pos_gain):
        gb = o[GRAV]
        tilt = jp.array([-gb[1], gb[0]])                 # roll, pitch (small angle)
        w = o[GYRO]
        v = o[VEL]
        exy = o[EXY]
        az = Kp_z * (z_ref - o[Z]) + Kd_z * (vz_ff - v[2])
        axy = pos_gain * Kp_xy * exy + Kd_xy * (v_ref_xy - v[0:2])
        F = m * jp.array([axy[0], axy[1], az + g])
        # gravity is not along body z when tilted: rotate the world-z component into the body frame
        F = F.at[0:2].add(-m * g * jp.array([gb[0], gb[1]]) * 0.0)
        tau = jp.array([-Kp_att[0] * tilt[0] - Kd_att[0] * w[0],
                        -Kp_att[1] * tilt[1] - Kd_att[1] * w[1],
                        -Kd_att[2] * 3.0 * w[2]])
        # prioritised allocation: torque + lift first, horizontal force scaled into the remaining range
        w_att = jp.concatenate([jp.zeros(2), F[2:3], tau]) - wtrim.at[0:2].set(0.0)
        w_xy = jp.concatenate([F[0:2], jp.zeros(4)]) - jp.concatenate([wtrim[0:2], jp.zeros(4)])
        u_att = jp.clip(u0 + M @ w_att, lo, hi)
        du_xy = M @ w_xy
        room = jp.where(du_xy > 0, (hi - u_att) / jp.maximum(du_xy, 1e-9), (lo - u_att) / jp.minimum(du_xy, -1e-9))
        sxy = jp.clip(jp.min(jp.where(jp.abs(du_xy) > 1e-9, room, 1.0)), 0.0, 1.0)
        u = u_att + sxy * du_xy
        T = u[:n_t]
        gang = u[n_t:]
        v_cell_ratio = o[VPACK] + 1.0
        T_av = env._T_max_nom * v_cell_ratio ** 1.7
        a = jp.zeros(env.action_size)
        a = a.at[gim_act].set(jp.clip(gang / rng_vec[gim_act], -1, 1))
        # PODS order == thrust action order (l_arm, r_arm, l_leg, r_leg)
        a = a.at[nj:].set(jp.clip((T / T_av - center) / gain, -1, 1))
        return a

    if task == "hover":
        def hover(obs, rng):
            o = obs["state"]
            c = o[CMD]
            return hover_core(o, c[3], 0.0, jp.zeros(2), 1.0)
        return hover

    if task == "maneuver":
        def man(obs, rng):
            o = obs["state"]
            c = o[CMD]
            return hover_core(o, c[3], 0.0, c[0:2], 0.0)
        return man

    if task == "transition":
        def trans(obs, rng):
            o = obs["state"]
            c = o[CMD]
            flight = c[7] > 0.5
            both = (o[CONTACT].sum() > 1.5)
            fly = hover_core(o, c[3], c[4], jp.zeros(2), 1.0)
            idle = jp.zeros(env.action_size).at[nj:].set(-1.0)
            # stay on the ground until commanded; after descent, cut fans once both pods touch
            landed = (~flight) & both
            return jp.where(landed | ((~flight) & (o[Z] < 0.03)), idle, fly)
        return trans
    raise ValueError(task)
