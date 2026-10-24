"""Queue a job: python training/submit.py <order> <name> <task> [train args...] [--noeval]"""
import json
import sys
from pathlib import Path

Q = Path(__file__).resolve().parent / "queue" / "pending"
Q.mkdir(parents=True, exist_ok=True)
order, name, task, *rest = sys.argv[1:]
ev = "--noeval" not in rest
rest = [r for r in rest if r != "--noeval"]
eval_args = []
if "--eval_args" in rest:
    i = rest.index("--eval_args")
    eval_args = rest[i + 1:]
    rest = rest[:i]
(Q / f"{order}_{name}.json").write_text(json.dumps({"name": name, "task": task, "train": rest, "eval": ev,
                                                    "eval_args": eval_args}, indent=1))
print(f"queued {order}_{name}")
