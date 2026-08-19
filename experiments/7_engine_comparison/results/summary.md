# Engine comparison — generated summary

(regenerate with plot_results.py; GT bags: ['fast_experiment0', 'fast_experiment1', 'calibrate'], holdout: motors0)

## Headline: all configs at 3 s windows

| config | GT-mean [m] | holdout (motors0) [m] |
|---|---|---|
| agx anisotropic | 0.376 | 0.248 |
| agx tuned | 0.333 | 0.263 |
| ostrich anisotropic | 0.261 | 0.272 |
| chrono SCM (phi=20) | 0.302 | 0.272 |
| agx defaults | 0.386 | 0.334 |
| ostrich tuned | 0.432 | 0.349 |
| ostrich defaults | 0.485 | 0.426 |
| chrono tuned | 0.406 | 0.493 |
| chrono defaults | 0.458 | 0.540 |

(Config selection came from the 15 s-window sweeps; this table re-scores those configs under the current window.)

## Axis A+B: sim-to-real accuracy (sweeps, scored at 15 s windows)

| engine | defaults [m] | tuned best [m] | best params | holdout def [m] | holdout best [m] |
|---|---|---|---|---|---|
| ostrich | 5.256 | 4.110 | `{'mu_front': 1.0, 'mu_rear': 0.3, 'compliance_contact': 1e-05, 'mu_rolling': 0.3, 'dt': 0.02}` | 3.928 | 3.207 |
| agx | 4.389 | 3.651 | `{'mu_front': 0.7, 'mu_rear': 0.2}` | 2.937 | 2.301 |
| chrono | inf | 4.142 | `{'mu_front': 1.0, 'mu_rear': 0.7, 'solver': 'psor', 'iterations': 500, 'rolling_resistance': 0.09}` | 3.892 | 3.577 |

### Solver sensitivity (spread across swept configs)

| engine | stable configs | unstable | min [m] | median [m] | max [m] |
|---|---|---|---|---|---|
| ostrich | 21 | 0 | 4.110 | 5.028 | 5.294 |
| agx | 16 | 9 | 3.651 | 4.389 | 4.440 |
| chrono | 11 | 14 | 4.142 | 4.291 | 6.754 |

## Wheel-terrain extensions (real turn gain alpha ~ 2)

| variant | GT-mean [m] | holdout [m] | alpha (1,3) | alpha mean | note |
|---|---|---|---|---|---|
| agx + anisotropic friction | 3.073 | 1.543 | 2.21 | 2.03 | `{'mu_front': 0.7, 'mu_rear': 0.4, 'mu_lat_front': 0.3, 'mu_lat_rear': 0.1, 'oriented_friction': True}` |
| ostrich + anisotropic friction | 3.225 | 1.720 | 2.84 | 2.75 | `{'mu_front': 0.1, 'mu_rear': 0.05, 'mu_long_front': 1.6, 'mu_long_rear': 0.6, 'mu_rolling': 0.3, 'ground_mu': 0.2}` |
| chrono SCM (phi=20) | 3.740 | 2.370 | 3.74 | 3.63 |  |
| chrono SCM (phi=30) | 4.116 | 2.430 | 3.74 | 3.67 |  |
| agxTerrain dirt_1 + TerrainWheel | 4.483 | 2.849 | 13.13 | 13.48 | diverges mid-bag |

## Axis C: speed / timestep (fast_experiment1)

![speed](speed_dt.png)

| engine | config | hardware | largest stable dt | RTF there | err there [m] |
|---|---|---|---|---|---|
| ostrich | defaults | gpu+cudagraph | 0.1 | 10.7x | 4.133 |
| ostrich | best | gpu+cudagraph | 0.1 | 10.4x | 3.104 |
| agx | defaults | cpu | 0.1 | 670.3x | 3.250 |
| agx | best | cpu | 0.1 | 685.8x | 3.255 |
| chrono | defaults | cpu | 0.05 | 622.7x | 2.938 |
| chrono | best | cpu | 0.05 | 177.3x | 3.189 |

## Axis D: behavioral scenarios (best configs, 2 reps)

### step16 — 16 cm step climb (real robot: climbs)

| engine | config | cleared | t_clear [s] | max pitch [deg] |
|---|---|---|---|---|
| ostrich | defaults | True | 4.9 | 13 |
| ostrich | defaults | True | 4.9 | 13 |
| ostrich | best | True | 5.0 | 12 |
| ostrich | best | True | 5.0 | 12 |
| agx | defaults | True | 5.8 | 14 |
| agx | defaults | True | 5.8 | 14 |
| agx | best | True | 6.1 | 14 |
| agx | best | True | 6.1 | 14 |
| chrono | defaults | True | 5.1 | 13 |
| chrono | defaults | True | 5.1 | 13 |
| chrono | best | True | 5.0 | 13 |
| chrono | best | True | 5.0 | 13 |

### turn gain alpha (real robot on this surface: ~2)

| engine | config | (1,3) | (1.5,3.5) | (2,4) | (0.5,3.5) |
|---|---|---|---|---|---|
| ostrich | defaults | 14.24 | 13.41 | 7.82 | 6.67 |
| ostrich | best | 4.80 | 4.62 | 4.24 | 3.67 |
| agx | defaults | 4.45 | 4.48 | 4.51 | 3.94 |
| agx | best | 4.36 | 4.38 | 4.38 | 2.91 |
| chrono | defaults | 1.05 | 1.07 | 1.07 | 1.05 |
| chrono | best | 1.09 | 1.14 | 1.19 | 1.08 |

### rock_field — 3x3 loose rocks

| engine | config | success | x_final [m] | lateral RMS [m] | wall [s] |
|---|---|---|---|---|---|
| ostrich | defaults | False | 1.7 | 0.73 | 3.6 |
| ostrich | defaults | False | 2.3 | 0.00 | 3.5 |
| ostrich | best | False | 1.8 | 0.00 | 7.8 |
| ostrich | best | False | 1.9 | 0.00 | 6.9 |
| agx | defaults | True | 7.0 | 0.02 | 0.3 |
| agx | defaults | True | 7.0 | 0.02 | 0.3 |
| agx | best | True | 6.7 | 0.01 | 0.3 |
| agx | best | True | 6.7 | 0.01 | 0.3 |
| chrono | defaults | True | 9.7 | 0.02 | 0.4 |
| chrono | defaults | True | 9.7 | 0.02 | 0.4 |
| chrono | best | True | 9.6 | 0.00 | 2.5 |
| chrono | best | True | 9.6 | 0.00 | 2.7 |
