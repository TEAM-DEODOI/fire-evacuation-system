"""STL -> URDF conversion for the building (D-025 Week 12 Milestone 1).

This module provides two URDF-construction paths:

1. :func:`build_building_urdf` -- wraps an exported PyroSim STL in a
   single-link fixed-base URDF. **Skeleton** (needs ``trimesh`` for the
   bounding-box probe). Real building geometry will flow through this
   path once Member A finishes the STL export.

2. :func:`build_placeholder_urdf` -- **functional**. Emits a URDF made
   of axis-aligned ``<box>`` primitives (no mesh files) that
   approximates the L-shape building outer walls + a couple of interior
   partitions, with 3 gaps at the canonical exit locations
   (``exit_west`` / ``exit_north`` / ``exit_east`` per
   :mod:`src.shared.building`). This lets the rest of the Week-12
   pipeline (PersonAgent collision, drone swarm A* over a non-empty
   world) be developed in parallel before the real STL lands.

The placeholder is *not* a model of the real building -- it is the
minimum sufficient geometry to exercise PyBullet collision against the
SLCF region boundary.

Conventions enforced (failures here cause weeks of evacuation-sim bugs,
per ``docs/pybullet_integration_spec.md`` §6):

* STL units in **metres** (not mm). If exported as mm, this module
  detects the bounding box magnitude > 1e3 and applies an explicit
  ``scale="0.001 0.001 0.001"`` to the mesh tag.
* Origin alignment: STL geometry must lie inside the SLCF region
  ``[0, 30] × [0, 20] × [0, 3.2]`` (Z up). A bounding-box check is
  performed before writing the URDF.
* Single fixed-base link -- building does not move under gravity.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from src.shared.constants import DOMAIN_SIZE_M


# ─── URDF templates ───────────────────────────────────────────────────────
_STL_URDF_TEMPLATE = """<?xml version="1.0" ?>
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

_PLACEHOLDER_URDF_HEAD = """<?xml version="1.0" ?>
<robot name="{name}">
  <link name="base">
    <inertial>
      <mass value="0.0"/>
      <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
    </inertial>
"""

_PLACEHOLDER_BOX_BLOCK = """    <collision>
      <origin xyz="{cx} {cy} {cz}" rpy="0 0 0"/>
      <geometry><box size="{sx} {sy} {sz}"/></geometry>
    </collision>
    <visual>
      <origin xyz="{cx} {cy} {cz}" rpy="0 0 0"/>
      <geometry><box size="{sx} {sy} {sz}"/></geometry>
      <material name="wall_{idx}">
        <color rgba="{r} {g} {b} {a}"/>
      </material>
    </visual>
"""

_PLACEHOLDER_URDF_TAIL = """  </link>
</robot>
"""


# ─── Dataclasses ──────────────────────────────────────────────────────────
@dataclass(frozen=True)
class BoxSpec:
    """Axis-aligned box for a placeholder wall segment.

    Attributes:
        center: ``(x, y, z)`` world centre in metres.
        size:   ``(x, y, z)`` full extents (not half-extents) in metres.
        label:  Short descriptor for debugging ("N_wall_left" etc.).
    """

    center: Tuple[float, float, float]
    size: Tuple[float, float, float]
    label: str = ""


@dataclass
class UrdfBuildResult:
    """Outcome of one URDF build.

    Attributes:
        urdf_path: Destination URDF file.
        mesh_path: STL referenced inside the URDF (``None`` for the
            primitive-only placeholder).
        scale: 3-tuple actually applied to the mesh. ``(1, 1, 1)`` for
            primitive-only URDFs.
        bounding_box: ``((x_min, x_max), (y_min, y_max), (z_min, z_max))``
            covering all geometry the URDF defines (after any scaling).
        warnings: Soft warnings raised during the build.
    """

    urdf_path: Path
    mesh_path: Optional[Path]
    scale: Tuple[float, float, float]
    bounding_box: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]
    warnings: List[str] = field(default_factory=list)


# ─── Placeholder building geometry ────────────────────────────────────────
# Exits (XY centre, gap half-width) -- gaps are cut from the outer walls.
_EXIT_W_Y_RANGE = (4.0, 6.0)         # exit_west @ (0, 5)
_EXIT_N_X_RANGE = (7.0, 9.0)         # exit_north @ (8, 18)
_EXIT_E_Y_RANGE = (12.0, 14.0)       # exit_east @ (30, 13)

_BUILDING_H_M = 3.2                  # STL height per D-015
_WALL_THICK_M = 0.2

# Outer wall extent (centre line just inside the SLCF region so the
# building fits in [0, 30] × [0, 20] × [0, 3.2]).
_LX, _LY, _ = DOMAIN_SIZE_M
_NEAR_W_X = _WALL_THICK_M / 2          # 0.1
_NEAR_E_X = _LX - _WALL_THICK_M / 2    # 29.9
_NEAR_S_Y = _WALL_THICK_M / 2          # 0.1
_NEAR_N_Y = _LY - _WALL_THICK_M / 2    # 19.9
_WALL_CZ = _BUILDING_H_M / 2           # 1.6


def _default_placeholder_boxes() -> List[BoxSpec]:
    """Return 9 boxes approximating the L-shape building.

    Outer walls are split into segments around the three exit gaps;
    two interior partitions break the rectangle into rough "rooms".
    """
    boxes: List[BoxSpec] = []

    # ── North wall (y ≈ 20), split by exit_north @ x∈[7, 9] ─────────
    nw_a_len = _EXIT_N_X_RANGE[0]
    nw_a_cx = nw_a_len / 2
    boxes.append(BoxSpec(
        center=(nw_a_cx, _NEAR_N_Y, _WALL_CZ),
        size=(nw_a_len, _WALL_THICK_M, _BUILDING_H_M),
        label="N_wall_left",
    ))
    nw_b_len = _LX - _EXIT_N_X_RANGE[1]
    nw_b_cx = _EXIT_N_X_RANGE[1] + nw_b_len / 2
    boxes.append(BoxSpec(
        center=(nw_b_cx, _NEAR_N_Y, _WALL_CZ),
        size=(nw_b_len, _WALL_THICK_M, _BUILDING_H_M),
        label="N_wall_right",
    ))

    # ── South wall (y ≈ 0), no exit ────────────────────────────────
    boxes.append(BoxSpec(
        center=(_LX / 2, _NEAR_S_Y, _WALL_CZ),
        size=(_LX, _WALL_THICK_M, _BUILDING_H_M),
        label="S_wall",
    ))

    # ── West wall (x ≈ 0), split by exit_west @ y∈[4, 6] ───────────
    ww_a_len = _EXIT_W_Y_RANGE[0]
    ww_a_cy = ww_a_len / 2
    boxes.append(BoxSpec(
        center=(_NEAR_W_X, ww_a_cy, _WALL_CZ),
        size=(_WALL_THICK_M, ww_a_len, _BUILDING_H_M),
        label="W_wall_bottom",
    ))
    ww_b_len = _LY - _EXIT_W_Y_RANGE[1]
    ww_b_cy = _EXIT_W_Y_RANGE[1] + ww_b_len / 2
    boxes.append(BoxSpec(
        center=(_NEAR_W_X, ww_b_cy, _WALL_CZ),
        size=(_WALL_THICK_M, ww_b_len, _BUILDING_H_M),
        label="W_wall_top",
    ))

    # ── East wall (x ≈ 30), split by exit_east @ y∈[12, 14] ────────
    ew_a_len = _EXIT_E_Y_RANGE[0]
    ew_a_cy = ew_a_len / 2
    boxes.append(BoxSpec(
        center=(_NEAR_E_X, ew_a_cy, _WALL_CZ),
        size=(_WALL_THICK_M, ew_a_len, _BUILDING_H_M),
        label="E_wall_bottom",
    ))
    ew_b_len = _LY - _EXIT_E_Y_RANGE[1]
    ew_b_cy = _EXIT_E_Y_RANGE[1] + ew_b_len / 2
    boxes.append(BoxSpec(
        center=(_NEAR_E_X, ew_b_cy, _WALL_CZ),
        size=(_WALL_THICK_M, ew_b_len, _BUILDING_H_M),
        label="E_wall_top",
    ))

    # ── Interior partition: vertical at x=15, y∈[0, 10] ────────────
    boxes.append(BoxSpec(
        center=(15.0, 5.0, _WALL_CZ),
        size=(_WALL_THICK_M, 10.0, _BUILDING_H_M),
        label="interior_v_partition",
    ))
    # ── Interior partition: horizontal at y=10, x∈[15, 30] ─────────
    boxes.append(BoxSpec(
        center=(22.5, 10.0, _WALL_CZ),
        size=(15.0, _WALL_THICK_M, _BUILDING_H_M),
        label="interior_h_partition",
    ))

    return boxes


def _box_bounding_box(
    boxes: List[BoxSpec],
) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """Union bounding box across a list of axis-aligned boxes."""
    xs: List[float] = []
    ys: List[float] = []
    zs: List[float] = []
    for b in boxes:
        cx, cy, cz = b.center
        sx, sy, sz = b.size
        xs.extend([cx - sx / 2, cx + sx / 2])
        ys.extend([cy - sy / 2, cy + sy / 2])
        zs.extend([cz - sz / 2, cz + sz / 2])
    return ((min(xs), max(xs)), (min(ys), max(ys)), (min(zs), max(zs)))


# ─── Public API ───────────────────────────────────────────────────────────
def build_building_urdf(
    stl_path: Path,
    out_urdf_path: Path,
    name: str = "building",
    auto_mm_to_m: bool = True,
    strict_bounds: bool = False,
) -> UrdfBuildResult:
    """Wrap ``stl_path`` in a single-link fixed-base URDF.

    Reads the STL via :mod:`trimesh` to probe units + bounds, then
    emits a URDF whose ``<mesh>`` tag references ``stl_path`` by a
    relative path. PyBullet's URDF loader resolves meshes relative to
    the URDF file's parent directory, so the URDF and STL should live
    in the same folder (typically ``assets/``).

    Args:
        stl_path: Path to the building STL.
        out_urdf_path: Where to write the URDF. The URDF references
            ``stl_path`` by ``stl_path.name`` only (assumes they share
            a parent). Parent of ``out_urdf_path`` is created if missing.
        name: ``<robot name=...>`` attribute.
        auto_mm_to_m: If the STL bounding box max-extent exceeds 1e3,
            assume mm units and apply ``scale=0.001`` to the mesh tag
            (L-010 lesson). Set to ``False`` to keep raw STL units.
        strict_bounds: If ``True``, raise on out-of-SLCF-region geometry
            after any auto-scaling. Default ``False`` emits a warning
            instead (Z can legitimately overhang to 3.2 m per D-015,
            and small XY edge gaps are normal for architectural STLs).

    Returns:
        :class:`UrdfBuildResult` populated with the actual scale used
        and the post-scaling bounding box.

    Raises:
        FileNotFoundError: If ``stl_path`` does not exist.
        ValueError: If ``strict_bounds`` is set and any axis of the
            scaled bounding box lies outside the SLCF region (with a
            small 0.2 m Z tolerance per D-015).
        ImportError: If ``trimesh`` is not installed.
    """
    stl_path = Path(stl_path)
    out_urdf_path = Path(out_urdf_path)
    if not stl_path.exists():
        raise FileNotFoundError(f"STL not found: {stl_path}")

    # Lazy import: avoid forcing trimesh on every caller of this module.
    try:
        import trimesh
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "trimesh is required for build_building_urdf; "
            "install with `pip install trimesh`."
        ) from exc

    mesh = trimesh.load(str(stl_path), force="mesh")
    if not hasattr(mesh, "bounds"):
        raise ValueError(
            f"trimesh did not return a single mesh for {stl_path} "
            f"(got {type(mesh).__name__}); STL may contain multiple "
            f"disconnected scenes"
        )
    raw_bounds = mesh.bounds  # (2, 3) -- min and max corners
    raw_extents = raw_bounds[1] - raw_bounds[0]
    max_extent = float(raw_extents.max())

    warnings: List[str] = []
    # L-010: PyroSim STLs are usually mm.
    if auto_mm_to_m and max_extent > 1e3:
        scale = (0.001, 0.001, 0.001)
        scaled_bounds = (
            (
                raw_bounds[0, 0] * 0.001,
                raw_bounds[1, 0] * 0.001,
            ),
            (
                raw_bounds[0, 1] * 0.001,
                raw_bounds[1, 1] * 0.001,
            ),
            (
                raw_bounds[0, 2] * 0.001,
                raw_bounds[1, 2] * 0.001,
            ),
        )
        warnings.append(
            f"auto mm->m: raw max extent {max_extent:.1f} -> scale 0.001 "
            f"applied"
        )
    else:
        scale = (1.0, 1.0, 1.0)
        scaled_bounds = (
            (float(raw_bounds[0, 0]), float(raw_bounds[1, 0])),
            (float(raw_bounds[0, 1]), float(raw_bounds[1, 1])),
            (float(raw_bounds[0, 2]), float(raw_bounds[1, 2])),
        )

    # SLCF bounds check.
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = scaled_bounds
    lx, ly, _lz = DOMAIN_SIZE_M
    z_overhang_tol = 0.2  # D-015: STL building may extend to 3.2 m
    z_max_allowed = _lz + z_overhang_tol
    oob_msgs: List[str] = []
    if xmin < -1e-3 or xmax > lx + 1e-3:
        oob_msgs.append(f"X [{xmin:.3f},{xmax:.3f}] outside [0,{lx}]")
    if ymin < -1e-3 or ymax > ly + 1e-3:
        oob_msgs.append(f"Y [{ymin:.3f},{ymax:.3f}] outside [0,{ly}]")
    if zmin < -1e-3 or zmax > z_max_allowed + 1e-3:
        oob_msgs.append(
            f"Z [{zmin:.3f},{zmax:.3f}] outside [0,{z_max_allowed}]"
        )
    if oob_msgs:
        if strict_bounds:
            raise ValueError(
                "STL bounds outside SLCF region after scaling: "
                + "; ".join(oob_msgs)
            )
        warnings.extend(oob_msgs)

    # Emit URDF -- mesh filename is relative to out_urdf_path.parent.
    mesh_filename = stl_path.name
    out_urdf_path.parent.mkdir(parents=True, exist_ok=True)
    if stl_path.parent.resolve() != out_urdf_path.parent.resolve():
        warnings.append(
            f"STL ({stl_path}) and URDF ({out_urdf_path}) live in different "
            f"directories; URDF references mesh by name '{mesh_filename}' "
            f"and PyBullet's loader resolves relative to URDF parent"
        )

    urdf_text = _STL_URDF_TEMPLATE.format(
        name=name,
        mesh_filename=mesh_filename,
        scale_x=scale[0], scale_y=scale[1], scale_z=scale[2],
    )
    out_urdf_path.write_text(urdf_text, encoding="utf-8")

    return UrdfBuildResult(
        urdf_path=out_urdf_path,
        mesh_path=stl_path,
        scale=scale,
        bounding_box=scaled_bounds,
        warnings=warnings,
    )


def build_placeholder_urdf(
    out_urdf_path: Path,
    boxes: Optional[List[BoxSpec]] = None,
    name: str = "placeholder_building",
    color_rgba: Tuple[float, float, float, float] = (0.65, 0.65, 0.70, 0.6),
) -> UrdfBuildResult:
    """Emit a primitive-box placeholder URDF approximating the L-shape building.

    The URDF has one fixed-base link with one ``<collision>`` +
    ``<visual>`` pair per :class:`BoxSpec`. PyBullet treats the union
    of all collision shapes as the link's contact body, so
    :class:`PersonAgent` capsules cannot pass through walls.

    Args:
        out_urdf_path: Destination URDF path. Parent directories are
            created. Existing file is overwritten.
        boxes: List of wall segments. ``None`` uses the canonical
            9-box L-shape approximation with gaps at the 3 exits.
        name: ``<robot name=...>``.
        color_rgba: Wall RGBA tuple ∈ [0, 1]^4 applied to every visual.
            Alpha < 1 gives a semi-transparent look helpful in GUI debug.

    Returns:
        :class:`UrdfBuildResult` with ``mesh_path=None``,
        ``scale=(1.0, 1.0, 1.0)``, and the union bounding box across
        the boxes.

    Raises:
        ValueError: If ``boxes`` is supplied but empty, or any
            ``BoxSpec`` has non-positive size, or the resulting
            bounding box lies entirely outside the SLCF region.
    """
    out = Path(out_urdf_path)
    if boxes is None:
        boxes = _default_placeholder_boxes()
    if not boxes:
        raise ValueError("boxes must contain at least one BoxSpec")
    for b in boxes:
        if any(s <= 0 for s in b.size):
            raise ValueError(f"box {b.label!r} has non-positive size {b.size}")

    bbox = _box_bounding_box(boxes)
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bbox
    if xmax <= 0 or ymax <= 0 or xmin >= _LX or ymin >= _LY:
        raise ValueError(
            f"placeholder boxes lie outside SLCF region "
            f"[0,{_LX}]x[0,{_LY}]: bbox={bbox}"
        )

    warnings: List[str] = []
    if xmax > _LX + 1e-6:
        warnings.append(f"bbox xmax {xmax:.3f} exceeds SLCF Lx {_LX}")
    if ymax > _LY + 1e-6:
        warnings.append(f"bbox ymax {ymax:.3f} exceeds SLCF Ly {_LY}")
    if zmax > _BUILDING_H_M + 1e-6:
        warnings.append(
            f"bbox zmax {zmax:.3f} exceeds STL height {_BUILDING_H_M}"
        )

    parts: List[str] = [_PLACEHOLDER_URDF_HEAD.format(name=name)]
    r, g, b, a = color_rgba
    for i, box in enumerate(boxes):
        cx, cy, cz = box.center
        sx, sy, sz = box.size
        parts.append(
            _PLACEHOLDER_BOX_BLOCK.format(
                idx=i,
                cx=cx, cy=cy, cz=cz,
                sx=sx, sy=sy, sz=sz,
                r=r, g=g, b=b, a=a,
            )
        )
    parts.append(_PLACEHOLDER_URDF_TAIL)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(parts), encoding="utf-8")

    return UrdfBuildResult(
        urdf_path=out,
        mesh_path=None,
        scale=(1.0, 1.0, 1.0),
        bounding_box=bbox,
        warnings=warnings,
    )


def slcf_bounds() -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """SLCF region (the *learnable* bounds) -- building must fit inside (Z may exceed by 0.2 m)."""
    lx, ly, lz = DOMAIN_SIZE_M
    return ((0.0, lx), (0.0, ly), (0.0, lz))


# ─── Self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import tempfile
    import pybullet as p

    print("=" * 60)
    print("urdf_builder.py self-test")
    print("=" * 60)

    errors: list[str] = []

    # ── 1. Default placeholder boxes summary ────────────────────────────
    print("\n[1] Default placeholder geometry")
    boxes = _default_placeholder_boxes()
    print(f"  {len(boxes)} boxes:")
    for b in boxes:
        print(
            f"    {b.label:<22}  centre={b.center}  size={b.size}"
        )
    if len(boxes) != 9:
        errors.append(f"expected 9 boxes, got {len(boxes)}")

    # ── 2. Build placeholder URDF -> temp file ──────────────────────────
    print("\n[2] build_placeholder_urdf -> temp file")
    with tempfile.TemporaryDirectory() as td:
        urdf_path = Path(td) / "placeholder.urdf"
        result = build_placeholder_urdf(urdf_path)
        print(f"  wrote {urdf_path}  ({urdf_path.stat().st_size} bytes)")
        print(f"  bbox = {result.bounding_box}")
        print(f"  scale = {result.scale}")
        print(f"  warnings = {result.warnings}")
        # Bbox must cover roughly the SLCF region.
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = result.bounding_box
        if not (xmin >= 0 and xmax <= _LX + 1e-6):
            errors.append(f"x-bbox [{xmin}, {xmax}] outside [0, {_LX}]")
        if not (ymin >= 0 and ymax <= _LY + 1e-6):
            errors.append(f"y-bbox [{ymin}, {ymax}] outside [0, {_LY}]")
        if not (zmax <= _BUILDING_H_M + 1e-6):
            errors.append(f"z-bbox top {zmax} exceeds {_BUILDING_H_M}")

        # ── 3. PyBullet loads the URDF ──────────────────────────────────
        print("\n[3] PyBullet loads the placeholder URDF (DIRECT)")
        cid = p.connect(p.DIRECT)
        try:
            building_id = p.loadURDF(
                str(urdf_path),
                basePosition=[0.0, 0.0, 0.0],
                useFixedBase=True,
                physicsClientId=cid,
            )
            print(f"  building_id = {building_id}")
            if building_id < 0:
                errors.append("loadURDF returned negative id")

            # ── 4. Body AABB roughly matches our bbox ──────────────────
            print("\n[4] PyBullet AABB roughly matches geometry bbox")
            aabb_min, aabb_max = p.getAABB(
                building_id, physicsClientId=cid
            )
            print(f"  PyBullet AABB min = {aabb_min}")
            print(f"  PyBullet AABB max = {aabb_max}")
            # AABB returned by PyBullet is only the base link's AABB; for
            # our single-link multi-collision body it should cover the
            # union of all boxes.
            for axis, lo, hi, exp_lo, exp_hi in zip(
                "xyz", aabb_min, aabb_max,
                (xmin, ymin, zmin), (xmax, ymax, zmax),
            ):
                if not (lo <= exp_lo + 0.1 and hi >= exp_hi - 0.1):
                    errors.append(
                        f"AABB {axis}: pybullet=[{lo:.3f},{hi:.3f}] "
                        f"vs geom=[{exp_lo:.3f},{exp_hi:.3f}]"
                    )
        finally:
            p.disconnect(physicsClientId=cid)

    # ── 5. Bad input rejected ─────────────────────────────────────────
    print("\n[5] Input validation")
    with tempfile.TemporaryDirectory() as td:
        bad_path = Path(td) / "bad.urdf"
        try:
            build_placeholder_urdf(bad_path, boxes=[])
        except ValueError:
            print("  PASS: empty boxes -> ValueError")
        else:
            errors.append("empty boxes did not raise")
        try:
            build_placeholder_urdf(
                bad_path,
                boxes=[BoxSpec((1.0, 1.0, 1.0), (0.0, 0.5, 0.5), "bad")],
            )
        except ValueError:
            print("  PASS: zero-size dim -> ValueError")
        else:
            errors.append("zero-size dim did not raise")

    # ── 6. build_building_urdf STL path: missing STL -> FileNotFoundError ─
    print("\n[6] build_building_urdf rejects missing STL")
    try:
        build_building_urdf(Path("dummy_missing.stl"), Path("dummy.urdf"))
    except FileNotFoundError:
        print("  PASS: missing STL -> FileNotFoundError")
    else:
        errors.append("missing STL did not raise FileNotFoundError")

    # ── 7. Real STL -> URDF (if present): mm->m auto-scale + bounds ───
    real_stl = Path("assets/science_hall_lv5.stl")
    if real_stl.exists():
        print("\n[7] build_building_urdf on real STL (mm -> m auto-scale)")
        with tempfile.TemporaryDirectory() as td:
            # Copy STL into the same dir as the target URDF (PyBullet
            # resolves <mesh filename=...> relative to URDF).
            import shutil
            stl_in_td = Path(td) / real_stl.name
            shutil.copy2(real_stl, stl_in_td)
            urdf_path = Path(td) / "building.urdf"
            res = build_building_urdf(stl_in_td, urdf_path)
            print(
                f"  scale={res.scale}  "
                f"bbox X{res.bounding_box[0]} Y{res.bounding_box[1]} "
                f"Z{res.bounding_box[2]}"
            )
            for w in res.warnings:
                print(f"  [warn] {w}")
            if res.scale != (0.001, 0.001, 0.001):
                errors.append(
                    f"expected mm-to-m scale 0.001 for real STL, got {res.scale}"
                )
            (xmin, xmax), (ymin, ymax), (zmin, zmax) = res.bounding_box
            if not (xmax <= _LX + 1e-2 and ymax <= _LY + 1e-2):
                errors.append(
                    f"scaled real-STL bbox exceeds SLCF region: "
                    f"X<={xmax}, Y<={ymax}"
                )
            # And it should actually load in PyBullet.
            cid = p.connect(p.DIRECT)
            try:
                bid = p.loadURDF(
                    str(urdf_path),
                    basePosition=[0.0, 0.0, 0.0],
                    useFixedBase=True,
                    physicsClientId=cid,
                )
                aabb_min, aabb_max = p.getAABB(bid, physicsClientId=cid)
                print(
                    f"  PyBullet AABB min={tuple(round(v, 2) for v in aabb_min)} "
                    f"max={tuple(round(v, 2) for v in aabb_max)}"
                )
                # Cell-level check: AABB should roughly match scaled bbox.
                if not all(
                    abs(aabb_max[i] - [xmax, ymax, zmax][i]) < 1.0
                    for i in range(3)
                ):
                    errors.append(
                        f"PyBullet AABB {aabb_max} disagrees with "
                        f"trimesh bbox max ({xmax}, {ymax}, {zmax})"
                    )
            finally:
                p.disconnect(physicsClientId=cid)
    else:
        print(f"\n[7] SKIP: real STL {real_stl} not present")

    # ── Verdict ───────────────────────────────────────────────────────
    if errors:
        print("\nFAIL")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print("\nPASS: placeholder URDF builder validated")
