"""PyBullet integration for EXP-PATH-001 (D-025, Week 12).

Scope per CLAUDE.md "EXP-PATH-001 - Three Scenarios" and
``docs/decisions.md::D-025``:

* :mod:`scene`       — PyBullet world + ground plane + camera.
* :mod:`urdf_builder`— STL -> URDF for the building.
* :mod:`person_agent`— Simplified 1.2 m/s occupant with
                       ``alive`` -> ``evacuated`` / ``dead`` lifecycle.
* :mod:`drone_swarm` — Multi-agent Crazyflie swarm using
                       :class:`~src.path_planning.planners.EvacuationPlanner`
                       for active waypoint guidance.
* :mod:`metrics`     — 5-metric collector (success_rate /
                       mean_evac_time / danger_zone_exposure /
                       casualty_rate / cumulative_FED).
* :mod:`scenarios`   — S1 fixed sign / S2 FDS swarm / S3 PI-FNO swarm.
* :mod:`run_exp_path_001` — Sweep entry point.

Only :mod:`scene` and the dataclass helpers in :mod:`metrics` /
:mod:`person_agent` are functional in the current commit; the rest are
skeletons with frozen signatures so the Week-12 implementation can be
split across milestones / contributors.
"""
