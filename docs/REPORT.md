# Engineering report: EDF humanoid ground-to-flight POC (simulation)

**Status:** simulation POC, 2026-10-24. Every result below comes from one hardware-constrained MuJoCo model, design E2: 0.88 m tall, 10.74 kg, four JP Hobby 90 mm 12S EDFs. **Nothing here is flight-validated.**

## 1. What was demonstrated

All results are on held-out seed 12345, 100 randomized trials each. The randomization covers:
- installed thrust −10%/+4% per fan
- payload ±10% with a ±2 cm COM offset
- spool lag 0.13–0.26 s
- servo gain ±15%
- gyro bias, sensor noise and pushes
- an unknown ground-effect sign (±15%)

| Milestone | Spec target | Result | Policy |
|---|---|---|---|
| M1 stand, 30 s, EDFs off | ≥95/100 | **100/100**, sway 0.9 cm | `stand_v2` |
| M2 walk ≥2 m, EDFs off | ≥80/100 | **100/100**, 0.135 m/s, **42 mm pod clearance**, 13% slip | `walk_v4` |
| M3 hover 20 s, ±0.15 m, ±10° | ≥90/100 | **100/100 on 3 seeds**, altitude RMS 1.1–1.4 cm, tilt p95 2.0–2.4°. Recommended `hover_v7` (wide randomization): 100/100, drift 0.08 m, 81% at −20% thrust | `hover_v5` s0/s1/s2, **`hover_v7`** |
| M4 stand → liftoff → 5 s hover → land → **unassisted** 10 s stand | ≥80/100 | **94/100** (`v4`). Recommended **`v5`** (wide randomization): 92/100 nominal, 91% at +10% mass, 95% at −10% thrust | `transition_v4`, **`transition_v5`** |
| Stretch: fly fwd / back / side, stop, re-hover | – | **100/100**, steady velocity error 0.07 m/s, altitude RMS 2.4 cm | `maneuver_v1` |
| Extra: single-fan failure mid-hover | quantify | **97/100 controlled 3-fan landings**, never tips, impact median 0.17 m/s, p95 0.85 m/s (0/100 without training) | `failsafe_v2_wideDR` |

**Classical baselines on the same suite:**

| Task | Controller | Success |
|---|---|---|
| Stand | PD + IMU | 40% |
| Hover | Cascaded PD + prioritized 6×10 allocation | 26% |
| Transition | FSM | 0% (tips at single-pod touchdown) |

## 2. Hardware feasibility summary (M0)

| | |
|---|---|
| Mass | 10.74 kg, every part massed: servos 2.43, shells 2.36, battery 2.00, EDFs 1.80, ESCs 0.72 kg… |
| Installed thrust | 4 × 43.2 N, limited by pack current, not the fan. Thrust/weight 1.64 nominal, 1.37 worst case (target 1.8 not met). |
| Hover | about 60% throttle, **8.7–8.9 kW**, about 190 A at 44 V, about 1.25 g/W |
| Endurance | about 80 s (75% usable energy), probably about 60 s if pack heating is the limit |
| Cost | about $9.2k BOM (`results/m0/bom_*.csv`, with sources) |
| Controllability | rank 6, cond 6. Roll ±12, pitch ±4.7, yaw ±3.9 N·m. Yaw authority is 6.8× the reaction-torque uncertainty. |

**Design decisions driven by the analysis:**
- **Size:** fan size sets the minimum scale. Below about 0.85 m, no catalog EDF gives ≥1.4 installed thrust/weight.
- **Leg pods:** the shin pod *is* the lower leg (per the concept art). The knee sits on its top centre; there is no ankle and no foot; it stands on 8 rim pads.
- **Shin-fan gimbal:** these pods are coaxial with the leg, so pitch can't come from differential thrust. The shin fans therefore use a 2-axis cartridge gimbal (±14°, set by bore clearance). The arm pods use a 1-axis fore/aft gimbal.
- **Shin exhaust:** the fan sits high with a vented 4-strut skirt below. The nozzle exits 84 mm above the floor, and the radial escape area is 3.7× the nozzle area (it was 1.1× with a full-length duct).

## 3. What remains unvalidated, ranked by risk

1. **Installed thrust.** The 0.85 install efficiency (side intake, guard, cartridge gap) is assumed. M5 shows hover success drops to 66–83% at 80% thrust.
2. **Ground effect and recirculation at liftoff and touchdown.** It is modeled only as a bounded ±15% multiplier, and the policies tolerate both signs. The real flow (jet impingement through the skirt vents) needs a thrust-stand test against a ground plane.
3. **EDF dynamics.** The 0.16 s first-order spool and the 25% residual reaction torque are assumptions. Policies tolerate +100 ms of extra lag.
4. **Servo thermal limits.** Walking reaches 86% of the assumed continuous rating (35% of stall). Stall is a momentary rating, and the real continuous torque needs a bench test.
5. **Battery.** IR (2.5 mΩ/cell) and the thermal rise (+47 K over a full hover) are assumed. The endurance claim rests on them.
6. **Rigid-body contact.** The rubber pads are modeled as rigid spheres, so landing impulses and compliance are approximate.
7. **State estimation.** The flight policies use a noisy velocity estimate and rangefinder altitude, and noise tolerance is 2× nominal at 100% and 3× at 92%. The ablation `hover_v8` (IMU, encoders and rangefinder only) still passes the M3 altitude and attitude criteria, but **drifts about 10 m/s**. **A downward optical-flow and range sensor is required** for position hold (add it to the BOM, about 10 g). Full mocap is not required. No estimator was implemented.

## 4. Failure modes found

| Failure | Behavior | Fix or mitigation |
|---|---|---|
| Single fan failed outright | **Unrecoverable in hover** (0%) | Trained descent lands 97% (p95 impact 0.85 m/s) |
| Mass +20% or thrust −20% | v5 policy fails 34–69% | Wide-randomization policy v6: 62–83% |
| Pushes >0.45 m/s while standing | Fans off: **falls** (0%) | Fan bursts: 0.6 m/s at 99% |
| Hard-landing tail | p95 impact 0.97 m/s in M4, 1.31 m/s at 1 m altitude | Near-ground sink-rate shaping did **not** help (v6). Likely the attitude transient at first-pod contact; still open. |
| Thrust-assisted "standing" | transition_v2 kept 30% of weight on the fans after landing | Caught by a stricter criterion and fixed (v3/v4) |
| Degenerate walking | walk_v1 shuffled with 6–9 mm clearance and about 45% slip | Gait clock plus true slip penalty, and removing the pod-flat term (v4) |

## 5. What RL contributed beyond classical control

- **Stand:** RL scored 100% vs 40% for pose-hold PD. The policy uses stepping and hip motion to recover.
- **Hover:** RL scored 100% vs 26%. The stateless classical controller has no integral action, so randomized thrust spread and COM offsets become steady errors. The policy compensates implicitly.
- **Transitions:** RL scored 94% vs 0%. The key behavior is contact-aware touchdown: throttling down as the first pod lands, then levelling. That's hard to hand-design without ankles.
- **Fan failure:** a learned emergency descent, 97% vs 0%.
- **Caveat:** the classical baselines are simple, with no integrators or MPC. A stronger model-based controller might close much of the gap; that remains open.

## 6. Path to a tethered physical prototype

These are the spec §10 steps, updated with what the simulation flagged.
1. **Thrust stand, one pod.** Use the JPH6007-104 with a FlyFun 110A HV on 12S, inside the printed shin-pod shell. Measure installed thrust vs throttle, power, spool step response, reaction torque, and **thrust against a ground plane at 30–120 mm exit height** (risks 1–3).
2. **Shin cartridge gimbal rig.** Measure ±14° travel, slew rate under gyroscopic load (0.65 N·m at 2 rad/s), leakage loss and the fan-failure telemetry flag.
3. **One leg, then two legs.** Bench the knee and hip XH540 pairs at the walking duty cycle and check the thermal margin (risk 4).
4. **Pack characterization:** IR, sag at 190 A, temperature rise over 60 s.
5. **Update the simulation** with the measured curves, retrain, and re-run this exact evaluation suite.
6. **Tethered hover only.** Use a tethered or protected rig with a remote cutoff and an exclusion zone, after an independent flight-safety review. Never an untethered flight near people in the first iteration.

## 7. Reproduce

See `README.md`. The experiment history, including every dead end and its lesson, is in `docs/experiments.md`.
