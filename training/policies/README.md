# Reference policies

Brax PPO parameters (normalizer + policy + value), MLP 512-256-128, trained on the E2 hardware model.

| File | Run | Task | Eval command (env overrides must match training) |
|---|---|---|---|
| `stand.pkl` | stand_v2 | stand | `python evaluation/evaluate.py --task stand --params training/policies/stand.pkl` |
| `walk.pkl` | walk_v4 | walk | `... --task walk --params training/policies/walk.pkl --env '{"gait_hz": 0.7, "swing_height": 0.05, "swing_sigma": 0.02}'` |
| `hover.pkl` | hover_v7_wideDR_pos | hover | `... --task hover --params training/policies/hover.pkl` |
| `transition.pkl` | transition_v5_wideDR | transition | `... --task transition --params training/policies/transition.pkl` |
| `maneuver.pkl` | maneuver_v1 | maneuver | `... --task maneuver --params training/policies/maneuver.pkl` |
| `failsafe.pkl` | failsafe_v2_wideDR | failsafe | `... --task failsafe --params training/policies/failsafe.pkl` |

`evaluate.py` writes `eval.json` + `rollout.npz` next to the params (use `--out DIR` to redirect); render with
`python scripts/render_rollout.py DIR/rollout.npz`. Full training settings: `training/runs/<run>/config.json`.
