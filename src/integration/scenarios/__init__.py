"""EXP-PATH-001 PyBullet scenarios (D-025).

Three scenarios share the same Scene + 20 :class:`PersonAgent`
instances and differ only in the *guidance system* and the
:class:`RiskMap` that backs the planner:

* ``s1_fixed_sign``  — static exit signs, no drones, FDS truth risk.
* ``s2_fds_swarm``   — drone swarm guided by FDS-derived RiskMap.
* ``s3_fno_swarm``   — drone swarm guided by PI-FNO predictions.

Each scenario module exposes ``run(...) -> ScenarioMetrics``.
"""

from src.integration.scenarios import s1_fixed_sign, s2_fds_swarm, s3_fno_swarm

__all__ = ["s1_fixed_sign", "s2_fds_swarm", "s3_fno_swarm"]
