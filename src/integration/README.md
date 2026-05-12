# Integration Module — Week 12 Demo

> **Scope**: This module is the exclusive responsibility of **Team Member D**.
> Other team members should not modify files here without coordination.

---

## Purpose

Week 12 integration demo: a PyBullet simulation that runs EXP-PATH-001,
comparing three evacuation scenarios where a drone swarm guides PersonAgents
out of a burning building.

The demo pipeline is:

1. Load a trained model (ConvLSTM or PI-FNO) from `checkpoints/`.
2. Run inference on a test FDS scenario to produce fire predictions.
3. Convert predictions to a `DynamicRiskMap` (from `src/risk_map/`).
4. Plan evacuation paths with `EvacuationPlanner` (from `src/path_planning/`).
5. Spawn a PyBullet environment with the building geometry and 20 PersonAgents.
6. Run three scenarios (S1 / S2 / S3) and collect per-person metrics.
7. Log cumulative FED, evacuation time, and casualty rate to W&B.

---

## Three Scenarios (EXP-PATH-001)

| Scenario | Guidance | RiskMap source | Validates |
|----------|----------|---------------|-----------|
| **S1 — Fixed sign baseline** | Static evacuation signs pre-placed in the building; persons individually query RiskMap and navigate toward the nearest non-hazardous sign | FDS | Baseline |
| **S2 — FDS drone swarm** | Drone swarm (Boids/APF) detects isolated persons and guides them via dynamic waypoints computed by Weighted A* | FDS | H6 |
| **S3 — PI-FNO drone swarm** | Same drone swarm and A* as S2 | PI-FNO | H5 |

S1 vs S2 → H6 (system effectiveness): does drone guidance reduce cumulative FED by ≥ 30%?  
S2 vs S3 → H5 (risk map fidelity): does the PI-FNO risk map deliver guidance quality on par with FDS?

All three scenarios share the same building, the same 20 PersonAgents, and the same `DynamicRiskMap` interface.

---

## PersonAgent Specification

- **Speed**: constant 1.2 m/s
- **Wall collision**: avoidance via PyBullet contact detection
- **State transitions**: `alive → evacuated` (reached an exit) / `alive → dead` (cumulative FED ≥ 0.3 or tenability threshold exceeded)
- **Excluded**: panic speed increase, social force model, crowd density effects

---

## Metrics Collected (per scenario)

Collected by `src/evaluation/metrics.py`:

| Metric | Description |
|--------|-------------|
| `evacuation_success_rate` | Fraction of persons evacuated within 300 s |
| `mean_evacuation_time` | Mean time to exit for evacuated persons (s) |
| `danger_zone_exposure_time` | Mean cumulative seconds spent in risk ≥ 0.5 zone |
| `casualty_rate` | Fraction of persons with status `dead` |
| `cumulative_FED` | Mean cumulative FED per person (primary metric for H6) |

---

## Dependencies

```python
import pybullet              # Simulation
import gym_pybullet_drones   # Drone swarm dynamics (multi-agent)
```

These are **week-12-only** dependencies. Do not import them in any other module.

---

## Files (to be created in Week 12)

| File | Description |
|------|-------------|
| `env_setup.py` | PyBullet scene setup: floor, walls, exits, lighting |
| `person_agent.py` | PersonAgent: constant 1.2 m/s movement, wall avoidance, alive/evacuated/dead states |
| `drone_swarm.py` | Drone swarm controller: Boids/APF collision avoidance, PSO mission assignment |
| `scenario_runner.py` | S1/S2/S3 scenario execution and per-person metric collection |
| `metrics_logger.py` | W&B logging, CSV export, EXP-PATH-001 result aggregation |

---

## Notes

- Drone swarm (multi-agent), not a single drone. (D-NEW-001)
- The demo uses pre-computed FDS data — no live CFD.
- Coordinate system is identical to the rest of the project (see `docs/coordinate_convention.md`).
- Decision history: D-NEW-001 (drone swarm adopted), D-NEW-002 (3-scenario expansion), D-NEW-003 (simplified PersonAgent movement model).
