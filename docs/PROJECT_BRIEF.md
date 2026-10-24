# Project specification: EDF humanoid ground-to-flight proof of concept

**Audience:** Autonomous coding agent / robotics engineer  
**Project type:** Simulation-first embodied robotics research and design feasibility study  
**Primary simulator:** MuJoCo (Python); alternate tools allowed when justified  
**End goal:** A reproducible portfolio-grade demonstration of standing, walking, hovering, takeoff and landing with one hardware-plausible humanoid design.

## 1. Vision and non-negotiable requirements

Design an **original**, compact, slender and graceful humanoid with a slim waist, narrow hips, slim upper arms and thighs, and **large cylindrical forearm and lower-leg housings** containing electric ducted fans (EDFs). The cylinders should visually resemble powerful propulsion pods strapped onto an otherwise elegant, slender robot. Its design language is retro high-visibility USAF: white fuselage-like shells, red/blue stripes, silver or dark metallic joints. Blonde hair and a face are reference-art features, **not** required in physics simulation. Avoid gold joints, red chest ribbon, or distinctive character-specific headgear. The physical design should retain a recognizable visual lineage with slender ground-based humanoids.

The agent may choose scale, overall height, DOF count, and mechanical architecture, **but must preserve the silhouette**. It must explicitly document and justify deviations from the reference, including actuator bulges, foot geometry, air intakes, and fan orientation. The forearm and shin pods must be noticeably larger than the adjacent upper arm and thigh; torso/hip must not be enlarged into a disguised quadcopter fuselage. The robot must have two arms, two legs, and no hidden rotors, wings, tail, or external flight frame in the final free-flight design. Small functional feet are allowed; prefer narrow feet and report stability implications.

**Critical:** The same robot mass properties, kinematic model, actuator envelopes and propulsion layout must be used across standing, walking, hovering and transitions. Do not quietly swap to a lighter or different flying model.

## 2. Core hypotheses

H1. A slender humanoid with oversized EDF pods can statically stand and execute slow controlled walking using commercially available servos.

H2. Four commercially available EDF assemblies can generate sufficient **installed** thrust (including duct and intake losses) to lift the complete robot with useful control authority and thermal/electrical margin.

H3. Vectoring the four thrust axes and modulating their magnitudes provides enough independent force and torque authority to stabilize all six rigid-body degrees of freedom around a hover configuration, subject to servo and fan response limits.

H4. A hybrid controller can lift off from standing, stabilize in hover, descend and regain foot contact without falling or exceeding contact-force/attitude constraints.

**An infeasible result is a valid outcome.** If no plausible commercial hardware meets the silhouette and mass/power constraints, report that clearly and identify the minimum design change needed. Never silently increase thrust or reduce mass.

## 3. Deliverables

1. Parametric robot model (MJCF preferred), mesh or primitive visualization with front/side/back views and a side-by-side reference-silhouette comparison.
2. Reproducible hardware BOM: exact manufacturer, part number, supplier link, dimensions, mass, voltage, current, rated thrust/torque, response assumptions, availability/price at time checked, and citations/datasheets. Separate verified specs from assumptions.
3. Mass, inertia, center-of-mass, power and thrust budgets; powertrain sizing and thermal/runtime estimates.
4. Four executable simulation environments/scenarios: standing, walking, hovering, and ground-to-flight-to-ground transitions.
5. Baseline controllers, optional learned policies, training/evaluation scripts, configuration files, seeds, and checkpoints.
6. Quantitative results with plots, failure cases, robustness sweeps, and short annotated videos for each scenario.
7. Engineering report explaining what was demonstrated, what remains unvalidated, and an actionable path to a tethered physical prototype.
8. Clean GitHub-ready README with hero render, architecture diagram, reproducibility commands and results table.

## 4. Hardware-first design workflow (before training)

### 4.1 Select a feasible scale

Evaluate at least three candidate heights, for example 30, 45 and 60 cm (illustrative only). For each, estimate the actual packaging volume and mass of EDFs, ESCs, batteries, servos, structural shells, wiring, flight controller and sensors. Choose the scale that maximizes feasibility and preserves the intended aesthetic, not the one that makes simulation easiest by ignoring hardware.

**Important:** Four EDFs capable of lifting the robot may be physically too large for a cute small robot. Treat this as a key research finding rather than forcing a predetermined height. Account for the fact that a fan's inlet must breathe unobstructed air; a solid-looking cylindrical housing cannot block the inlet. Document plausible side/forward inlet paths and exhaust clearance. The forearm and shin housings may require substantially more diameter than their neighboring limbs.

### 4.2 Commercial components only

- **Propulsion:** Four purchasable EDF assemblies with brushless motors, compatible ESCs and documented test data. Prefer manufacturer thrust curves for the exact voltage, fan and inlet/exhaust setup. Do not treat free-stream bench thrust as guaranteed installed thrust.
- **Actuation:** Commercial hobby/robotic servos or integrated actuators with actual torque-speed curves, weight, dimensions, backlash and voltage limits. Select load-bearing hip/knee/ankle actuators and independent thrust-vectoring actuators. Servos must have enough continuous torque at their required speed; stall torque is not a continuous operating rating.
- **Power:** Realistic battery cells/pack and wiring/connector/ESC current limits; include battery sag, usable capacity, discharge heating and reserve. No massless power supply in final validation. External power may be used only in explicitly labeled diagnostic experiments.
- **Control:** Realistic IMU, optional barometer/range sensor, MCU/flight-controller mass and update rates. Assume no perfect position/velocity observation in the final policy unless a realizable external motion-capture setup is explicitly part of the prototype plan.
- **Structure:** Printable shells and commercial fasteners/bearings/shafts. Include a plausible wall thickness, fasteners, mounts, strain relief and service access. Structural and cable mass must not be zero.

Provide a **procurement gate** before any claim of physical feasibility: at least one exact commercially available candidate per critical part, supplier evidence, and an integrated mass/power budget. If current product information cannot be retrieved, record components as *unverified candidates* and mark hardware feasibility unresolved.

### 4.3 First-order calculations

Let total loaded mass be `m`, gravity `g`, individual installed thrusts `T_i`, and vertical components `T_i cos(theta_i)`.

- Hover requires `sum_i T_i cos(theta_i) >= m*g`.
- Define installed hover thrust margin `M = (sum_i T_i,max,installed)/(m*g)`; target approximately **1.8 or greater** as an initial design objective, not a universal pass criterion. Re-evaluate against actual control authority and operating efficiency.
- Compute full 6D wrench mapping from each fan's thrust direction and lever arm: `F = sum_i T_i d_i`, `tau = sum_i (r_i - r_COM) x (T_i d_i) + tau_reaction,i`.
- Check local controllability / allocation Jacobian rank and conditioning for the **actual** arm and leg geometry, thrust-vectoring ranges and rate limits. Do not assume four thrust magnitudes alone provide six independent control axes.
- Check achievable roll/pitch/yaw torque under asymmetric thrust, servo rate limits, and reaction torque uncertainty.
- Estimate electrical power at hover from **measured fan curves** rather than ideal momentum theory alone; estimate conservative flight duration from usable pack energy, reserve, and auxiliary loads.
- Include actuator self-weight, housing mass, battery position, changing inertia with arm pose, and ground reaction loads.

For hover, include both static trim and dynamic tests with motor lag. A mathematical hover trim without sufficient yaw authority, feedback bandwidth or runtime is not a validated aircraft.

## 5. Parametric mechanical architecture

Start with a low-complexity robot and add DOFs only when necessary. Suggested minimum: torso rigid body; two arms with shoulder pitch/roll and optional elbow; two legs with hip pitch/roll, knee pitch and ankle pitch (additional yaw/roll as needed). Preserve plausible proportions and joint ranges. Place EDF thrust application at the actual fan axis within each oversized forearm/shin pod; place realistic masses/inertias there too.

**Thrust vectoring must be mechanically plausible.** Model a nozzle or pivoting propulsion assembly with a commercial servo, finite angular travel and slew rate. An EDF does not automatically behave like a high-temperature jet with a vectoring nozzle; assess whether a gimbaled fan or suitable aerodynamic vane arrangement is mechanically feasible and quantify efficiency losses. No instantaneous, unlimited-direction thrust vectors.

Ensure legs can stand and walk without the exhaust contacting the ground or causing unacceptable ground clearance problems. Simulate feet with contact/friction, even if small. If slim-foot standing proves unstable, quantify the smallest aesthetic compromise that makes it work.

## 6. Simulation architecture

Recommended repository:

```text
edf-humanoid/
  README.md
  configs/               # geometry, hardware, training, eval
  assets/                # generated meshes, optional reference renders
  models/                # MJCF and procedural generation
  hardware/              # BOM, datasheets, provenance, sizing notebooks
  envs/                  # stand, walk, hover, transition
  controllers/           # PD, LQR/MPC, allocator, FSM, learned policies
  training/              # PPO/SAC scripts and checkpoints
  evaluation/            # batch tests, robustness sweeps, metrics
  scripts/               # render, smoke tests, demo capture
  results/               # tables, plots, videos
  docs/                  # design decisions, limitations, prototype roadmap
```

Use MuJoCo rigid-body physics and contact for ground tasks. Add an explicit custom thrust/torque model using measured EDF curves and a first-order or empirically derived spool-up/down dynamic. Model battery voltage/charge approximately and constrain total current. MuJoCo alone does **not** validate complex inlet flow, ground effect, recirculation or fan-to-fan interference. Add bounded uncertainties for these and identify where CFD or thrust-stand testing is required.

Maintain two model tiers:
- **Ideal diagnostic model:** unlimited sensing, simplified thrust and/or external support, clearly marked for debugging only.
- **Hardware-constrained model:** real mass, power, actuator response, control limits, noise, and contacts. **Only this tier may count toward project success.**

## 7. Milestone sequence and pass criteria

### M0 — Feasibility gate (must precede RL)

Deliver parametric silhouette, exact candidate hardware BOM, estimated integrated mass and center of mass, thrust margin, battery endurance, servo torque margins, and control-allocation analysis. Show a Pareto comparison of candidate scales. Flag any unsatisfied constraints.

**Pass:** At least one plausible purchasable configuration fits the geometry and has static hover and ground support margins; if not, produce a documented infeasibility report and revised design proposal. Do not proceed claiming physical plausibility without this gate.

### M1 — Standing

Use a conventional joint-space PD controller and/or whole-body controller first; RL optional later. Initialize near nominal standing pose, then perturb orientation and center of mass. Test on level ground with EDFs off.

**Suggested success:** Stand unsupported for 30 s in at least 95/100 randomized trials, with no fall, joint limit violations or actuator saturation lasting beyond specified tolerance. Record foot contact forces, torque-speed envelopes, sway, and recovery limits. Vary friction and mass properties.

### M2 — Walking

Implement simple gait baseline (trajectory optimization, gait generator or finite-state machine), then consider PPO for robustness. EDFs off in the primary walking benchmark; a separate thrust-assisted walking ablation is allowed.

**Suggested success:** Walk at least 2 m at a specified slow speed in at least 80/100 randomized trials, without falling, and report speed, distance, energy, step width and peak/continuous servo loading. Measure turning separately if feasible. Narrow feet must remain narrow in this test.

### M3 — Hover

Begin with a free rigid body and fixed limb pose, then articulated limbs. Derive a baseline thrust allocator plus PID/LQR/MPC attitude and position loops. Compare against a learned controller if useful. Check motor/servo bandwidth, thrust-vector angles and yaw authority.

**Suggested success:** From a near-hover initial condition, maintain altitude within +/-0.15 m and roll/pitch within +/-10 degrees for 20 s in at least 90/100 randomized trials, with no ground contact, using installed thrust limits and battery power model. Report yaw drift, position drift, controller effort, saturation, recovery from perturbations and estimated flight duration. These tolerances are provisional and should scale with robot size.

### M4 — Takeoff, landing, and ground-to-flight transitions

Use a hybrid state machine with phases `STAND -> PREPARE -> THRUST_RAMP -> LIFTOFF -> HOVER -> DESCEND -> TOUCHDOWN -> STAND`. Allow distinct subcontrollers initially; then evaluate a shared or hierarchical RL policy. Implement contact-aware transitions and safe aborts.

**Suggested success:** At least 80/100 trials complete a full cycle: unsupported stand, liftoff, 5 s stable hover, controlled descent, both-foot touchdown and 10 s unsupported stand, without falling or exceeding impact/servo/attitude limits. Define impact speed and acceptable foot-contact impulse quantitatively based on the selected structure. Test failures and emergency motor shutdown behavior in simulation.

### M5 — Robustness and sim-to-real readiness

Sweep total mass (+/-10%), COM offset, EDF thrust loss (e.g. 10-20%), motor lag, vectoring servo lag/backlash, IMU noise, joint friction, ground friction, battery sag, single-EDF degradation and fan asymmetry. Report success distributions and failure modes. Single-fan failure may be unrecoverable; quantify rather than assume recovery.

A prototype-readiness review should explicitly state which parameters require measurement on a thrust stand, servo bench or constrained rig, and which remain uncertain due to aerodynamics.

## 8. Control and RL design

Use classical control baselines to prevent wasting compute on learning basic stabilization from scratch. Recommended strategy:

- **Standing/walking:** PD or model-based stabilization; optional PPO locomotion policy.
- **Hover:** model-based attitude control and constrained thrust allocation first; optional PPO residual control or end-to-end policy benchmark.
- **Transitions:** hybrid state machine + learned phase-specific policies, or a hierarchical RL controller conditioned on mode and desired altitude.

Possible observation vector: joint positions/velocities, IMU orientation and angular rates, estimated vertical velocity/altitude, commanded body pose, contact estimates, previous EDF commands and battery state. Do not feed simulator-only world-state values into the final deployable policy without identifying a sensor or estimator.

Possible action vector: servo joint targets or torques, four EDF commands, vectoring servo targets. Clip and rate-limit all commands to hardware specifications.

Reward components may include pose/altitude tracking, low angular velocity, stable foot contacts, low impact speed, moderate power usage, avoidance of actuator saturation and successful mode completion. Penalize unsafe contact, falls and oscillation. Keep terminal success metrics separate from reward to avoid reward hacking.

For reproducibility: multiple seeds, held-out test seeds, training curves, fixed evaluation suite and ablations. Compare (a) classical only, (b) learned only where sensible, and (c) hybrid. Record wall-clock and sample efficiency.

## 9. Key risks / no-cheating checklist

- No massless battery, ESCs, wiring, mounts or fan housings.
- No fictional EDF thrust or zero-lag motor response.
- No weightless limb articulation or servos operating continuously at stall torque.
- No magical 6-DOF control from an underactuated thruster layout.
- No hidden rotor, external stabilizer or permanently attached tether in final benchmark.
- No idealized infinite-friction feet, enlarged invisible collision feet or frictionless ground contact.
- No swapping hardware, mass or geometry between flight and walking benchmarks.
- No treating MuJoCo as a CFD solver or claiming actual flightworthiness from simulation.
- No substituting render-only aesthetics for a geometrically plausible CAD/physics design.

## 10. Physical prototype roadmap (not part of this initial POC)

1. Buy and characterize **one** EDF + ESC + battery combination on an instrumented, guarded thrust stand. Measure thrust, electrical power, thermal rise and spool response at multiple throttle points.
2. Validate one thrust-vectoring mechanism on a constrained rig. Measure vector angle, thrust loss, servo bandwidth and failure modes.
3. Build a representative forearm/shin pod to verify actual inlet area, assembly tolerances, cooling, stiffness and mass.
4. Bench-test one leg and then a two-leg stand/walk prototype, using measured servo performance.
5. Only after constrained testing and independent flight-safety review, evaluate a tethered/protected hover rig with remote emergency cutoff and exclusion zone. Never use an indoor occupied space for open high-power EDF testing.
6. Update the simulation with measured curves before considering untethered operation.

**Safety:** Spinning EDFs can cause severe injury and high-power LiPo packs can fail dangerously. Add fan guards where feasible, secure test mounts, emergency disarm, thermal/current protection and safe charging. Do not plan an untethered human-proximity flight demonstration in the first iteration.

## 11. Portfolio presentation

Make the result easy for a robotics hiring manager to assess in under two minutes:

- One hero video: design turnaround -> standing -> walking -> hover -> takeoff/landing (with explicit labels identifying simulation).
- One comparison of concept silhouette vs simplified physics model.
- A quantitative table with standing, walking, hover and transition success rates and uncertainty ranges.
- A hardware feasibility table with total mass, installed thrust, thrust-to-weight ratio, electrical power, runtime and cost.
- An honest failures section, especially yaw controllability, EDF response, ground effect, and small-foot stability.
- A short explanation of what RL contributed beyond a conventional controller.

## 12. Agent execution instructions

**Work autonomously in stages.** Begin with M0 feasibility and a runnable minimal MJCF; do not spend days on detailed meshes. Prioritize a working simulation with correct physics over visual polish. Commit at each milestone and run automated smoke tests. If product data is unavailable, maintain explicit assumptions and request a sourcing review before calling the design hardware-realistic. Prefer conservative estimates. Surface any design contradiction early and propose 2-3 tradeoffs, including aesthetic consequences.

**First output expected from the coding agent:** (1) candidate scale comparison; (2) preliminary exact-part BOM with source links; (3) mass/thrust/torque feasibility spreadsheet or script; (4) MJCF skeleton and rendered front/side view; (5) risk register; (6) implementation plan. Proceed to controller development only after M0's constraints are evaluated.

**Definition of done:** All four target behaviors are demonstrated and evaluated with the **same hardware-constrained robot**, or the project produces a reproducible, quantitative infeasibility result identifying the smallest realistic change that could make them possible. A beautiful animation without this evidence is not a successful POC.
