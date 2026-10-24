"""PPO training (Brax) on the hardware-constrained EDF humanoid.

python training/train.py --task hover --name hover_v1 --steps 150e6 [--restore runs/x/params.pkl]
Writes training/runs/<name>/{config.json, progress.csv, params.pkl, train.log, eval.json}
"""
from __future__ import annotations

import argparse
import csv
import functools
import json
import os
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax  # noqa: E402
from brax.training.agents.ppo import networks as ppo_networks  # noqa: E402
from brax.training.agents.ppo import train as ppo  # noqa: E402
from mujoco_playground import wrapper  # noqa: E402

from envs.edf_env import make  # noqa: E402

RUNS = ROOT / "training" / "runs"


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--steps", type=float, default=100e6)
    ap.add_argument("--num_envs", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--entropy", type=float, default=3e-3)
    ap.add_argument("--lr_schedule", default="ADAPTIVE_KL", help="NONE | ADAPTIVE_KL")
    ap.add_argument("--desired_kl", type=float, default=0.01)
    ap.add_argument("--discount", type=float, default=0.97)
    ap.add_argument("--unroll", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--minibatches", type=int, default=32)
    ap.add_argument("--updates", type=int, default=4)
    ap.add_argument("--policy", default="512,256,128")
    ap.add_argument("--value", default="512,256,128")
    ap.add_argument("--evals", type=int, default=12)
    ap.add_argument("--restore", default=None, help="params.pkl to warm-start the policy (same obs/action)")
    ap.add_argument("--env", default="{}", help="JSON dict of env config overrides, dotted keys")
    ap.add_argument("--init_std", type=float, default=1.0)
    return ap.parse_args()


def main():
    a = parse()
    out = RUNS / a.name
    out.mkdir(parents=True, exist_ok=True)
    overrides = json.loads(a.env)
    env = make(a.task, **overrides)
    eval_env = make(a.task, **overrides)
    (out / "config.json").write_text(json.dumps({**vars(a), "env_config": env._config.to_dict()}, indent=1, default=str))

    net = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=tuple(int(x) for x in a.policy.split(",")),
        value_hidden_layer_sizes=tuple(int(x) for x in a.value.split(",")),
        policy_obs_key="state", value_obs_key="privileged_state", init_noise_std=a.init_std,
    )
    restore = None
    if a.restore:
        with open(a.restore, "rb") as fh:
            restore = pickle.load(fh)
    t0 = time.time()
    prog = out / "progress.csv"
    fields = None
    best = {"reward": -1e18, "step": -1}
    last = {"params": None}

    def progress(step, metrics):
        nonlocal fields
        row = {"step": int(step), "wall_s": round(time.time() - t0, 1)}
        for k, v in metrics.items():
            try:
                row[k] = float(v)
            except (TypeError, ValueError):
                pass
        new = fields is None
        if new:
            fields = list(row.keys())
        with prog.open("a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(row)
        rew = row.get("eval/episode_reward", float("nan"))
        # brax calls policy_params_fn just before progress_fn: keep the best evaluated checkpoint
        if last["params"] is not None and rew == rew and rew > best["reward"]:
            best.update(reward=rew, step=int(step))
            with open(out / "params_best.pkl", "wb") as fh:
                pickle.dump(last["params"], fh)
        ln = row.get("eval/avg_episode_length", float("nan"))
        print(f"[{a.name}] step={step:,} t={row['wall_s']:.0f}s eval_reward={rew:.2f} ep_len={ln:.0f}", flush=True)

    def save_params(step, make_policy, params):
        last["params"] = params
        with open(out / "params_last.pkl", "wb") as fh:
            pickle.dump(params, fh)

    make_inference, params, _ = ppo.train(
        environment=env,
        eval_env=eval_env,
        # full_reset: the default auto-reset only restores data/obs, NOT info, so battery SoC,
        # step counters, commands and domain-randomisation draws leaked across episodes
        wrap_env_fn=functools.partial(wrapper.wrap_for_brax_training, full_reset=True),
        num_timesteps=int(a.steps),
        num_evals=a.evals,
        reward_scaling=1.0,
        episode_length=env._config.episode_length,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=a.unroll,
        num_minibatches=a.minibatches,
        num_updates_per_batch=a.updates,
        discounting=a.discount,
        learning_rate=a.lr,
        entropy_cost=a.entropy,
        num_envs=a.num_envs,
        batch_size=a.batch,
        max_grad_norm=1.0,
        clipping_epsilon=0.2,
        network_factory=net,
        seed=a.seed,
        num_eval_envs=256,
        progress_fn=progress,
        policy_params_fn=save_params,
        restore_params=restore,
        restore_value_fn=False,
        deterministic_eval=True,
        learning_rate_schedule=a.lr_schedule,
        desired_kl=a.desired_kl,
    )
    with open(out / "params_last.pkl", "wb") as fh:
        pickle.dump(params, fh)
    # params.pkl (what evaluation and warm starts use) = best deterministic-eval checkpoint
    src = out / "params_best.pkl" if (out / "params_best.pkl").exists() else out / "params_last.pkl"
    (out / "params.pkl").write_bytes(src.read_bytes())
    (out / "best.json").write_text(json.dumps(best))
    print(f"[{a.name}] DONE in {time.time() - t0:.0f}s best_eval={best['reward']:.2f} @ {best['step']:,}", flush=True)


if __name__ == "__main__":
    main()
