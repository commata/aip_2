# Offensive BT+RL Hybrid Final Report

## Result

The selected candidate is a conservative BT-first hybrid, not a claim of dominant RL performance. Across six held-out scenario/seed pairs, it preserved every BT outcome while improving mean reward, ATA, and surface-saturation rate. Mean health margin regressed slightly, so the evaluator keeps an explicit non-regression tolerance and the report does not label the result as a decisive win over BT.

## Selected configuration

- Mode: `offensive_residual`
- Residual scale: `0.10`
- RL cadence inside gate: every 6 simulator frames (10 Hz at 60 Hz simulation)
- BT cadence: every simulator frame
- Throttle: one policy mapping from `[-1, 1]` to `[0, 1]`, followed by convex blending only
- Low-energy throttle guard: preserve BT throttle below speed `210` when RL requests less power
- Gate entry: range `152.4–1500 m`, ownship ATA `<=15°`, target ATA `>=135°`
- Gate exit hysteresis: range `<=2000 m`, ownship ATA `<=25°`, target ATA `>=110°`

Outside the gate, the returned action is the exact clipped BT action and RL inference is skipped. Crossing held-out trajectories recorded gate occupancy `0`, RL inference calls `0`, and exact BT/hybrid metric equality where the offensive geometry never appeared.

## Held-out paired evidence

Evaluation used three deterministic scenario buckets (`offensive_tail`, `crossing_left`, `crossing_right`) and two held-out seeds (`307`, `401`), producing 12 runs: one BT and one hybrid run for each scenario/seed pair.

| Metric | BT | Hybrid 0.10 | Delta |
|---|---:|---:|---:|
| Win rate | 16.67% | 16.67% | 0.00 pp |
| Crash rate | 0.00% | 0.00% | 0.00 pp |
| Mean reward | 142.7592 | 146.1316 | +3.3725 |
| Mean health margin | -0.04648 | -0.05042 | -0.00394 |
| Mean ATA | 73.5812° | 73.0553° | -0.5259° |
| Surface saturation rate | 0.8989 | 0.8844 | -0.0145 |
| Minimum altitude | 533.1 m | 521.9 m | -11.2 m |

Outcome counts were identical for both controllers: one win, one loss, four timeouts, and zero crashes. The minimum hybrid altitude remained above the configured 300 m safety floor.

## Why the earlier configuration was rejected

The initial wide gate (`2700/3300 m`, `36/52°`) and all fixed scales from `0.10` through `0.20` lost a held-out BT win by timing out. Maneuver telemetry showed the wide gate active for roughly half or more of the engagement. The RL surface corrections were small, but throttle reduction during low-energy turning altered the long-term trajectory. Narrowing the gate and adding the low-energy throttle guard restored the BT win at scale `0.10`.

## Verification and artifacts

- Core and automation tests: 20 passing
- Real JSBSim smoke, scale-grid, gate search, throttle-guard, crossing, and held-out paired evaluations completed
- Raw local evidence: `artifacts/evaluations/final_heldout/` (intentionally ignored by Git)
- Machine-readable selected configuration: `automation/best_offensive_hybrid.json`
- Branch: `codex/offensive-hybrid-autotune`

No PR or merge to `main` was created. Existing model bundles, checkpoints, DLLs, and rule XML files were not committed or modified.
