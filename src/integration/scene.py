"""PyBullet scene setup for EXP-PATH-001 (D-025 Week 12).

This module owns the PyBullet world: physics client, ground, optional
building URDF, camera, and pre-step bookkeeping. Higher-level scenario
modules (``scenarios/s1_fixed_sign``, ``scenarios/s2_fds_swarm``,
``scenarios/s3_fno_swarm``) compose a :class:`Scene` plus a population
of :class:`~src.integration.person_agent.PersonAgent` and (for S2/S3) a
:class:`~src.integration.drone_swarm.DroneSwarm`.

Coordinate conventions follow ``docs/coordinate_convention.md``: Z-up,
metres, world origin at corner ``(0, 0, 0)`` matching the SLCF region
``[0, 30] × [0, 20] × [0, 3] m``. PyBullet's default Y-up is therefore
overridden via the gravity vector and per-camera basis.

This implementation is intentionally minimal in the first commit:

* connect (DIRECT for headless tests / GUI for demos),
* set gravity,
* spawn a plane at ``z=0`` and an axis marker,
* expose :meth:`Scene.step` that advances physics by a configurable
  ``dt`` and increments wall-clock ``Scene.t``.

The building URDF, fire visualisation, and persons/drones are wired in
follow-up commits as the other ``src/integration/`` modules land.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pybullet as p
import pybullet_data

from src.shared.constants import DOMAIN_SIZE_M


# ─── Config ───────────────────────────────────────────────────────────────
@dataclass
class SceneConfig:
    """Top-level scene knobs.

    Attributes:
        connection_mode: ``"DIRECT"`` (headless, fast) or ``"GUI"`` (slow,
            human-visible). Default DIRECT so tests pass on CI.
        gravity_mps2: Z-down gravity magnitude. Default 9.81.
        dt_s: Physics step in seconds. The evacuation sim's outer loop
            calls :meth:`Scene.step` once per dt.
        building_urdf: Optional path to ``assets/building.urdf``. If
            ``None`` the scene is empty (used for unit tests).
        draw_origin_axes: Render small RGB axis lines at origin for
            debugging in GUI mode.
    """

    connection_mode: str = "DIRECT"
    gravity_mps2: float = 9.81
    dt_s: float = 1.0
    building_urdf: Optional[Path] = None
    draw_origin_axes: bool = True


# ─── Scene ────────────────────────────────────────────────────────────────
@dataclass
class Scene:
    """Owns the PyBullet physics client and the static world geometry.

    Construct with :meth:`Scene.create` (preferred); the dataclass init
    exists for advanced cases that need to inject an externally managed
    physics client.

    Attributes:
        client: PyBullet physics client id.
        config: :class:`SceneConfig`.
        plane_id: Body id of the ground plane.
        building_id: Body id of the building URDF, or ``None``.
        t: Wall-clock simulation time (s).
    """

    client: int
    config: SceneConfig
    plane_id: int = -1
    building_id: Optional[int] = None
    t: float = 0.0
    _axis_lines: list[int] = field(default_factory=list)

    # ─── Lifecycle ─────────────────────────────────────────────────────
    @classmethod
    def create(cls, config: Optional[SceneConfig] = None) -> "Scene":
        """Connect to PyBullet and populate the world.

        Args:
            config: :class:`SceneConfig`. Defaults to a headless DIRECT
                scene with no building URDF.

        Returns:
            A ready-to-step :class:`Scene`.

        Raises:
            RuntimeError: If PyBullet connection fails.
            FileNotFoundError: If ``config.building_urdf`` is set but
                does not exist.
            ValueError: If ``config.connection_mode`` is unknown.
        """
        cfg = config or SceneConfig()
        mode = cfg.connection_mode.upper()
        if mode == "DIRECT":
            cid = p.connect(p.DIRECT)
        elif mode == "GUI":
            cid = p.connect(p.GUI)
        else:
            raise ValueError(
                f"connection_mode must be DIRECT or GUI, got {cfg.connection_mode!r}"
            )
        if cid < 0:
            raise RuntimeError("pybullet.connect returned negative id")

        # pybullet_data ships urdf/plane.urdf — use it as the floor.
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)
        p.setGravity(0, 0, -cfg.gravity_mps2, physicsClientId=cid)

        plane_id = p.loadURDF("plane.urdf", physicsClientId=cid)

        building_id: Optional[int] = None
        if cfg.building_urdf is not None:
            urdf_path = Path(cfg.building_urdf)
            if not urdf_path.exists():
                p.disconnect(cid)
                raise FileNotFoundError(
                    f"building URDF not found: {urdf_path}"
                )
            building_id = p.loadURDF(
                str(urdf_path),
                basePosition=[0.0, 0.0, 0.0],
                useFixedBase=True,
                physicsClientId=cid,
            )

        scene = cls(
            client=cid,
            config=cfg,
            plane_id=plane_id,
            building_id=building_id,
            t=0.0,
        )

        if cfg.draw_origin_axes:
            scene._draw_origin_axes()

        return scene

    def close(self) -> None:
        """Disconnect the physics client."""
        try:
            p.disconnect(physicsClientId=self.client)
        except p.error:
            pass

    # Context-manager sugar for tests.
    def __enter__(self) -> "Scene":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ─── Stepping ──────────────────────────────────────────────────────
    def step(self) -> None:
        """Advance physics by ``config.dt_s`` and increment :attr:`t`.

        Higher-level scenario loops call this once per outer tick.
        PyBullet integrates internally at the engine's fixed timestep
        (default 1/240 s), but the *exposed* clock here advances by
        ``dt_s`` so it stays aligned with the SLCF frame rate
        (10 s by default for fire data).
        """
        # PyBullet's stepSimulation advances by its internal timestep;
        # for our slow walking sim, calling it once per outer dt is
        # sufficient to update collision / sensor state.
        p.stepSimulation(physicsClientId=self.client)
        self.t += float(self.config.dt_s)

    # ─── Debug helpers ─────────────────────────────────────────────────
    def _draw_origin_axes(self, length: float = 1.0) -> None:
        """Draw RGB axis lines at world origin for visual debugging."""
        ox, oy, oz = 0.0, 0.0, 0.01  # slight lift so axes aren't z-fighting
        rgb = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
        ends = ((length, 0, 0), (0, length, 0), (0, 0, length))
        for c, e in zip(rgb, ends):
            lid = p.addUserDebugLine(
                lineFromXYZ=[ox, oy, oz],
                lineToXYZ=[ox + e[0], oy + e[1], oz + e[2]],
                lineColorRGB=c,
                lineWidth=2.0,
                physicsClientId=self.client,
            )
            self._axis_lines.append(lid)

    def world_extents(self) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        """Return ``((x_min, x_max), (y_min, y_max), (z_min, z_max))`` for the SLCF region."""
        lx, ly, lz = DOMAIN_SIZE_M
        return ((0.0, lx), (0.0, ly), (0.0, lz))


# ─── Self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("scene.py self-test")
    print("=" * 60)

    errors: list[str] = []

    # ── 1. DIRECT mode connect + ground + step ──────────────────────────
    print("\n[1] Scene.create() in DIRECT mode")
    scene = Scene.create(SceneConfig(connection_mode="DIRECT", dt_s=1.0))
    if scene.client < 0:
        errors.append("client id < 0")
    if scene.plane_id < 0:
        errors.append("plane_id < 0 (plane.urdf load failed)")
    print(f"  client={scene.client}  plane_id={scene.plane_id}  t={scene.t}")

    # ── 2. step() advances t ────────────────────────────────────────────
    print("\n[2] step() advances Scene.t by dt_s")
    scene.step()
    scene.step()
    if not math.isclose(scene.t, 2.0):
        errors.append(f"t after 2 steps = {scene.t}, expected 2.0")
    print(f"  t after 2 steps = {scene.t}")

    # ── 3. world_extents ────────────────────────────────────────────────
    print("\n[3] world_extents matches SLCF region")
    ext = scene.world_extents()
    if ext != ((0.0, 30.0), (0.0, 20.0), (0.0, 3.0)):
        errors.append(f"world_extents wrong: {ext}")
    print(f"  extents = {ext}")

    # ── 4. close() disconnects ──────────────────────────────────────────
    print("\n[4] close() releases the client")
    scene.close()
    # After disconnect, getBodyInfo should raise pybullet.error.
    try:
        p.getBodyInfo(scene.plane_id, physicsClientId=scene.client)
    except p.error:
        print("  PASS: client disconnected (getBodyInfo raised)")
    else:
        errors.append("client still alive after close()")

    # ── 5. Context manager ──────────────────────────────────────────────
    print("\n[5] Context-manager closes on exit")
    with Scene.create(SceneConfig(connection_mode="DIRECT")) as s2:
        cid = s2.client
        if cid < 0:
            errors.append("ctx-mgr client < 0")
    try:
        p.getBodyInfo(0, physicsClientId=cid)
    except p.error:
        print("  PASS: client released after with-block")
    else:
        errors.append("client still alive after with-block")

    # ── 6. Bad connection_mode rejected ─────────────────────────────────
    print("\n[6] Bad connection_mode raises ValueError")
    try:
        Scene.create(SceneConfig(connection_mode="HEADLESS"))
    except ValueError:
        print("  PASS")
    else:
        errors.append("bad connection_mode did not raise")

    # ── 7. Missing URDF raises FileNotFoundError ────────────────────────
    print("\n[7] Missing building URDF raises FileNotFoundError")
    try:
        Scene.create(
            SceneConfig(
                connection_mode="DIRECT",
                building_urdf=Path("assets/__no_such_urdf__.urdf"),
            )
        )
    except FileNotFoundError:
        print("  PASS")
    else:
        errors.append("missing URDF did not raise")

    # ── Verdict ────────────────────────────────────────────────────────
    if errors:
        print("\nFAIL")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print("\nPASS: Scene minimal smoke validated")
