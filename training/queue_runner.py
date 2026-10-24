"""Sequential GPU job queue: exactly one training job on the GPU at a time.

Jobs are JSON files in training/queue/pending/ (processed in filename order):
  {"name": "hover_v1", "task": "hover", "train": ["--steps", "150e6"], "eval": true}
Lifecycle: pending/ -> running/ -> done/ | failed/. Logs: training/runs/<name>/train.log
Run: nohup python training/queue_runner.py &   (a pidfile guards against duplicates)
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
Q = ROOT / "training" / "queue"
RUNS = ROOT / "training" / "runs"
PY = str(ROOT / ".venv" / "bin" / "python")
PID = Q / "runner.pid"
STATUS = Q / "status.json"
MAX_RETRIES = 1


def env():
    e = dict(os.environ)
    libs = ":".join(str(p) for p in sorted((ROOT / ".venv/lib/python3.10/site-packages/nvidia").glob("*/lib")))
    e["LD_LIBRARY_PATH"] = libs + ":/usr/lib/wsl/lib"
    e["MUJOCO_GL"] = "egl"
    e["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    e["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.85"
    return e


def write_status(**kw):
    STATUS.write_text(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), **kw}, indent=1))


def run(cmd, log):
    with open(log, "a") as fh:
        fh.write(f"\n$ {' '.join(cmd)}\n")
        fh.flush()
        p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT, env=env())
        return p.wait()


def main():
    for d in ("pending", "running", "done", "failed"):
        (Q / d).mkdir(parents=True, exist_ok=True)
    if PID.exists():
        try:
            os.kill(int(PID.read_text()), 0)
            print("runner already alive", file=sys.stderr)
            return
        except (OSError, ValueError):
            pass
    PID.write_text(str(os.getpid()))
    # a job left in running/ means the previous runner died mid-job: requeue it once
    for f in (Q / "running").glob("*.json"):
        job = json.loads(f.read_text())
        job["retries"] = job.get("retries", 0) + 1
        dest = Q / ("pending" if job["retries"] <= MAX_RETRIES else "failed") / f.name
        dest.write_text(json.dumps(job, indent=1))
        f.unlink()
    while True:
        jobs = sorted((Q / "pending").glob("*.json"))
        if not jobs:
            write_status(state="idle")
            time.sleep(30)
            continue
        f = jobs[0]
        job = json.loads(f.read_text())
        run_f = Q / "running" / f.name
        shutil.move(str(f), run_f)
        name, task = job["name"], job["task"]
        out = RUNS / name
        out.mkdir(parents=True, exist_ok=True)
        log = out / "train.log"
        write_status(state="running", job=name, started=time.strftime("%H:%M:%S"))
        t0 = time.time()
        rc = 0
        if job.get("train", None) is not None:
            rc = run([PY, "-u", "training/train.py", "--task", task, "--name", name, *job["train"]], log)
        if rc == 0 and job.get("eval", True):
            ev = [PY, "-u", "evaluation/evaluate.py", "--task", task, "--params", str(out / "params.pkl"),
                  *job.get("eval_args", [])]
            if job.get("train") is None and job.get("controller"):
                ev = [PY, "-u", "evaluation/evaluate.py", "--task", task, "--controller", job["controller"],
                      "--out", str(out), *job.get("eval_args", [])]
            rc = run(ev, log)
        job.update(returncode=rc, minutes=round((time.time() - t0) / 60, 1), finished=time.strftime("%H:%M:%S"))
        if rc != 0 and job.get("retries", 0) < MAX_RETRIES:
            job["retries"] = job.get("retries", 0) + 1
            (Q / "pending" / f.name).write_text(json.dumps(job, indent=1))
        else:
            (Q / ("done" if rc == 0 else "failed") / f.name).write_text(json.dumps(job, indent=1))
        run_f.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
