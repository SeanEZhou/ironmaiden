# M5 robustness sweeps

One perturbation at a time, layered on top of the standard domain randomization. 100 trials each on held-out seed 12345.

## hover_v5

| condition | success | no-fall | alt_err_rms_m | tilt_deg_p95 | pos_drift_m_mean | hover_power_kw |
|---|---|---|---|---|---|---|
| com3cm | 1.00 | 1.00 | 0.0127 | 2.22 | 1.04 | 8.92 |
| com5cm | 1.00 | 1.00 | 0.0128 | 2.15 | 1.08 | 8.92 |
| fan0_0 | 0.00 | 0.00 | 1.49 | 171 | 4.04 | 7.55 |
| fan0_50 | 0.66 | 0.90 | 0.369 | 142 | 1.38 | 8.79 |
| fan0_70 | 0.93 | 0.99 | 0.0664 | 2.76 | 1.06 | 8.91 |
| fan0_85 | 1.00 | 1.00 | 0.0211 | 2.16 | 0.986 | 8.92 |
| lag100 | 1.00 | 1.00 | 0.0137 | 2.39 | 0.992 | 8.92 |
| lag50 | 1.00 | 1.00 | 0.0132 | 2.38 | 0.995 | 8.92 |
| mass110 | 0.81 | 0.89 | 0.323 | 16.7 | 1.21 | 10.1 |
| mass120 | 0.31 | 0.50 | 0.904 | 164 | 2.27 | 10 |
| noise2x | 1.00 | 1.00 | 0.0257 | 3.18 | 0.379 | 8.91 |
| noise3x | 0.92 | 1.00 | 0.0481 | 4.66 | 1.15 | 8.9 |
| nominal | 1.00 | 1.00 | 0.0154 | 0.941 | 1.14 | 8.68 |
| thrust80 | 0.66 | 0.77 | 0.575 | 153 | 1.71 | 8.38 |
| thrust90 | 0.93 | 0.97 | 0.0973 | 3.11 | 1.13 | 8.88 |

## transition_v3

| condition | success | no-fall | liftoff_rate | landed_rate | impact_mps_p95 | unassisted_stand_rate |
|---|---|---|---|---|---|---|
| fan0_70 | 0.93 | 0.98 | 1 | 0.98 | 0.703 | 0.98 |
| ge_minus15 | 0.96 | 1.00 | 1 | 1 | 0.503 | 1 |
| ge_plus10 | 0.94 | 1.00 | 1 | 1 | 0.612 | 1 |
| lag50 | 0.95 | 1.00 | 1 | 0.99 | 0.571 | 1 |
| mass110 | 0.85 | 0.96 | 1 | 0.98 | 0.861 | 0.96 |
| nominal | 0.91 | 1.00 | 1 | 0.95 | 0.588 | 1 |
| thrust90 | 0.91 | 1.00 | 1 | 1 | 0.862 | 1 |

## stand_v2

| condition | success | no-fall | sway_cm_mean | max_tilt_deg_p95 |
|---|---|---|---|---|
| fric04 | 1.00 | 1.00 | 0.96 | 3.08 |
| fric06 | 1.00 | 1.00 | 0.966 | 3.19 |
| fric13 | 1.00 | 1.00 | 1.01 | 3.25 |

## walk_v4

| condition | success | no-fall | speed_mps_mean | pod_clearance_mm_mean | slip_fraction_of_travel |
|---|---|---|---|---|---|
| fric04 | 1.00 | 1.00 | 0.131 | 41.6 | 0.129 |
| fric06 | 0.99 | 1.00 | 0.132 | 41.7 | 0.129 |
| fric13 | 1.00 | 0.99 | 0.139 | 41.6 | 0.132 |

## stand_v2 pushes (no fans)

| condition | success | no-fall | sway_cm_mean |
|---|---|---|---|
| 02 | 1.00 | 1.00 | 6.64 |
| 03 | 1.00 | 1.00 | 15.1 |
| 045 | 0.00 | 0.00 | 20.7 |
| 06 | 0.00 | 0.00 | 15.1 |

## stand_v3 pushes (thrust-assist)

| condition | success | no-fall | sway_cm_mean |
|---|---|---|---|
| 045 | 1.00 | 1.00 | 21.4 |
| 060 | 0.99 | 0.99 | 30.7 |
| 080 | 0.59 | 0.59 | 47.9 |
