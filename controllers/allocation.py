"""Thrust-vector allocation analysis on the actual MuJoCo kinematics.

Inputs u = [T_1..T_4, g_1..g_k]: installed thrust per pod (N) and gimbal angles (rad).
Wrench about the whole-body COM: w = [Fx, Fy, Fz, tau_x, tau_y, tau_z] in the world frame.
"""
from __future__ import annotations

import numpy as np
import mujoco
from scipy.optimize import least_squares, linprog

PODS = ("l_pod_arm", "r_pod_arm", "l_pod_leg", "r_pod_leg")


class Allocator:
    def __init__(self, model: mujoco.MjModel, qpos: np.ndarray, t_max: float, t_min_frac: float = 0.05):
        self.m = model
        self.d = mujoco.MjData(model)
        self.q0 = qpos.copy()
        self.t_max = t_max
        self.t_min = t_min_frac * t_max
        self.sites = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"thrust_{p}") for p in PODS]
        self.gimbals = [j for j in range(model.njnt) if "_vec_" in mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)]
        self.gimbal_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in self.gimbals]
        self.n_t, self.n_g = len(PODS), len(self.gimbals)
        self.mass = float(model.body_subtreemass[1])

    def bounds(self):
        lo = [self.t_min] * self.n_t + [self.m.jnt_range[j][0] for j in self.gimbals]
        hi = [self.t_max] * self.n_t + [self.m.jnt_range[j][1] for j in self.gimbals]
        return np.array(lo), np.array(hi)

    def geometry(self, g):
        q = self.q0.copy()
        for j, a in zip(self.gimbals, g):
            q[self.m.jnt_qposadr[j]] = a
        self.d.qpos[:] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        com = self.d.subtree_com[1].copy()
        pos = np.array([self.d.site_xpos[s] for s in self.sites])
        axis = np.array([self.d.site_xmat[s].reshape(3, 3)[:, 2] for s in self.sites])
        return com, pos, axis

    def wrench(self, u):
        T, g = u[: self.n_t], u[self.n_t:]
        com, pos, axis = self.geometry(g)
        F = (T[:, None] * axis).sum(0)
        tau = np.cross(pos - com, T[:, None] * axis).sum(0)
        return np.concatenate([F, tau])

    def trim(self):
        target = np.array([0, 0, self.mass * 9.81, 0, 0, 0])
        lo, hi = self.bounds()
        u0 = np.concatenate([np.full(self.n_t, self.mass * 9.81 / self.n_t), np.zeros(self.n_g)])
        scale = np.array([1, 1, 1, 10, 10, 10.0])
        # regulariser: equalise thrusts (throttle headroom) and keep gimbals centred, in range-normalised units
        rng = hi - lo
        res = least_squares(lambda u: np.concatenate([(self.wrench(u) - target) * scale,
                                                      0.05 * self.mass * (u - u0) / rng]),
                            np.clip(u0, lo, hi), bounds=(lo, hi))
        return res.x, self.wrench(res.x) - target

    def jacobian(self, u, eps=1e-4):
        J = np.zeros((6, len(u)))
        for i in range(len(u)):
            du = np.zeros_like(u)
            du[i] = eps
            J[:, i] = (self.wrench(u + du) - self.wrench(u - du)) / (2 * eps)
        return J

    def analyze(self, H):
        u, resid = self.trim()
        J = self.jacobian(u)
        lo, hi = self.bounds()
        mg = self.mass * 9.81
        # normalise: inputs by their full range, outputs by mg and mg*0.1H
        Sin = np.diag(hi - lo)
        Sout = np.diag(1 / np.array([mg, mg, mg, mg * 0.1 * H, mg * 0.1 * H, mg * 0.1 * H]))
        Jn = Sout @ J @ Sin
        sv = np.linalg.svd(Jn, compute_uv=False)
        rank = int((sv > 1e-6 * sv[0]).sum())
        # attainable pure wrench per axis (linearised, others held at zero)
        names = ["Fx", "Fy", "Fz", "tau_x", "tau_y", "tau_z"]
        attain = {}
        for k, nm in enumerate(names):
            vals = []
            for sgn in (1, -1):
                c = np.zeros(len(u) + 1)
                c[-1] = -1
                A_eq = np.hstack([J, -sgn * np.eye(6)[:, [k]]])
                b_eq = np.zeros(6)
                bnds = [(l - x, h - x) for l, h, x in zip(lo, hi, u)] + [(0, None)]
                r = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bnds, method="highs")
                vals.append(r.x[-1] if r.success else 0.0)
            attain[nm] = (vals[0], vals[1])
        com, pos, axis = self.geometry(u[self.n_t:])
        return dict(u_trim=u.tolist(), trim_residual=resid.tolist(), J=J.tolist(), sv_norm=sv.tolist(),
                    rank=rank, cond=float(sv[0] / max(sv[-1], 1e-12)), attainable=attain,
                    pod_pos_rel_com=(pos - com).tolist(), pod_axes=axis.tolist(), com=com.tolist(),
                    gimbals=self.gimbal_names)
