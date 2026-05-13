"""STL → URDF conversion for the building (D-025 Week 12 Milestone 1).

The project's building geometry is authored in PyroSim (proprietary
``.pyrogeom`` format) and exported as STL. PyBullet does not load STL
directly as a kinematic body — it expects URDF + mesh references. This
module converts the building STL into a single-link, fixed-base URDF
with both collision and visual meshes pointing at the same STL file.

Pipeline::

    data/raw/s_000/*.pyrogeom
        │  (PyroSim GUI export, 1-time manual step — outside this module)
        ▼
    assets/building.stl
        │  build_building_urdf(...)
        ▼
    assets/building.urdf

After the URDF is on disk :class:`~src.integration.scene.Scene` loads
it via ``config.building_urdf=Path("assets/building.urdf")``.

Conventions enforced (failures here cause weeks of evacuation-sim bugs,
per ``docs/pybullet_integration_spec.md`` §6):

* STL units in **metres** (not mm). If exported as mm, this module
  detects the bounding box magnitude > 1e3 and applies an explicit
  ``scale="0.001 0.001 0.001"`` to the mesh tag.
* Origin alignment: STL geometry must lie inside the SLCF region
  ``[0, 30] × [0, 20] × [0, 3.2]`` (Z up). A bounding-box check is
  performed before writing the URDF.
* Single fixed-base link — building does not move under gravity.

**Status: skeleton.** Conversion + bounding-box validation pending.
The STL itself is not yet checked into the repo (handled by Member A
or by an outsourced PyBullet contributor per Spec §3.4).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from src.shared.constants import DOMAIN_SIZE_M


_URDF_TEMPLATE = """<?xml version="1.0" ?>
<robot name="{name}">
  <link name="base">
    <inertial>
      <mass value="0.0"/>
      <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
    </inertial>
    <visual>
      <geometry>
        <mesh filename="{mesh_filename}" scale="{scale_x} {scale_y} {scale_z}"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <mesh filename="{mesh_filename}" scale="{scale_x} {scale_y} {scale_z}"/>
      </geometry>
    </collision>
  </link>
</robot>
"""


# ─── Config ───────────────────────────────────────────────────────────────
@dataclass
class UrdfBuildResult:
    """Outcome of one URDF build.

    Attributes:
        urdf_path: Destination URDF file.
        mesh_path: STL referenced inside the URDF (relative).
        scale: 3-tuple actually applied to the mesh.
        bounding_box: ``((x_min, x_max), (y_min, y_max), (z_min, z_max))``
            of the STL after scaling — m.
        warnings: Soft warnings raised during the build (e.g. mesh
            slightly outside SLCF region, mm-to-m auto-scaling applied).
    """

    urdf_path: Path
    mesh_path: Path
    scale: Tuple[float, float, float]
    bounding_box: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]
    warnings: list


# ─── Public API ───────────────────────────────────────────────────────────
def build_building_urdf(
    stl_path: Path,
    out_urdf_path: Path,
    name: str = "building",
    auto_mm_to_m: bool = True,
    strict_bounds: bool = False,
) -> UrdfBuildResult:
    """Wrap ``stl_path`` in a single-link fixed-base URDF.

    Args:
        stl_path: Path to the building STL.
        out_urdf_path: Where to write the URDF. Parent directories are
            created. The URDF references ``stl_path`` by a path
            **relative** to ``out_urdf_path``'s parent (PyBullet's URDF
            loader resolves meshes relative to the URDF).
        name: ``<robot name=...>`` attribute.
        auto_mm_to_m: If the STL bounding box on any axis exceeds 1e3,
            assume mm units and apply ``scale=0.001`` to the mesh tag.
        strict_bounds: If True, raise on out-of-SLCF-region geometry.
            Default False emits a warning instead (the building may
            legitimately overhang the buffer zone in Z by 0.2 m, per
            D-015).

    Returns:
        :class:`UrdfBuildResult`.

    Raises:
        FileNotFoundError: If ``stl_path`` does not exist.
        ValueError: If ``strict_bounds`` and the STL extends outside
            the SLCF region.
        NotImplementedError: Pending the next implementation commit
            (needs ``trimesh`` for bounding-box read).
    """
    raise NotImplementedError(
        "Week 12 M1: implement STL bounding-box probe (trimesh) + "
        "URDF emission via _URDF_TEMPLATE. After implementation, "
        "verify with: python -m src.integration.scene with "
        "building_urdf=assets/building.urdf."
    )


def slcf_bounds() -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """SLCF region (the *learnable* bounds) — building must fit inside (Z may exceed by 0.2 m)."""
    lx, ly, lz = DOMAIN_SIZE_M
    return ((0.0, lx), (0.0, ly), (0.0, lz))


# ─── Self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("urdf_builder.py - skeleton (Week 12 M1 implementation pending)")
    print(f"  slcf_bounds = {slcf_bounds()}")
    print("SKIP")
