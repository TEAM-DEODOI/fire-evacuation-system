"""
Auto-repair PyroSim-exported ``.fds`` files for our pipeline.

PyroSim defaults to a SLCF Z upper bound of 3.5 m when the STL building is
3.2 m tall. With our 0.5 m grid that produces 7 cells in Z instead of the
required 6 — fdsreader then silently drops time frames and raises a
broadcast error (L-009). Likewise, PyroSim sometimes emits
``VECTOR=.TRUE.`` on cell-centred SLCFs, which triggers an off-by-one in
fdsreader (L-001).

This script fixes both issues:

1. Lowers the SLCF Z upper bound from ``3.5`` to ``3.0`` (only when the
   ``XB`` already ends in ``3.5``; other 3.5 values are untouched).
2. Strips any ``VECTOR=.TRUE.`` attribute from SLCF lines.
3. Writes a ``.fds.bak`` backup before overwriting the original.

Usage::

    python scripts/fix_pyrosim_fds.py path/to/scenario.fds
    python scripts/fix_pyrosim_fds.py data/raw/                 # recursive
    python scripts/fix_pyrosim_fds.py --dry-run data/raw/       # preview
    python scripts/fix_pyrosim_fds.py --self-test               # built-in test
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple

# Matches a SLCF namelist whose XB has six comma-separated numbers ending in
# ``3.5`` (immediately followed by the ``/`` that closes the namelist).
# Only the Z-max ``3.5`` is replaced; everything else is preserved.
#
# ``[^/]*?`` is used instead of ``.*?`` so the match transparently spans line
# breaks — PyroSim regularly emits multi-line ``&SLCF ... /`` blocks. ``[^/]``
# guarantees we cannot stretch across a closing slash into another namelist,
# even when ``DOTALL`` is not set.
_SLCF_Z_MAX_RE = re.compile(
    r"""(?P<prefix>&SLCF\b[^/]*?XB\s*=\s*(?:[-+0-9.]+\s*,\s*){5})
        (?P<z>3\.5)
        (?P<suffix>\s*/)
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Matches ``VECTOR=.TRUE.`` with optional whitespace, anywhere in a SLCF
# line. We absorb a single adjacent comma so the resulting line keeps its
# namelist syntax (``A=1, VECTOR=.TRUE., B=2`` → ``A=1, B=2``).
_VECTOR_TRUE_RE = re.compile(
    r"""\s*,?\s*VECTOR\s*=\s*\.TRUE\.\s*""",
    re.IGNORECASE | re.VERBOSE,
)


def _split_slcf_lines(content: str) -> List[Tuple[bool, str]]:
    """Tag each line as SLCF-or-not for selective processing.

    Returns:
        List of ``(is_slcf, line)`` tuples preserving original ordering and
        newline characters.
    """
    return [
        (bool(re.match(r"\s*&SLCF\b", ln, flags=re.IGNORECASE)), ln)
        for ln in content.splitlines(keepends=True)
    ]


def fix_slcf_z_range(fds_path: Path) -> bool:
    """Rewrite ``fds_path`` so any SLCF with ``XB ... ,3.5/`` becomes ``,3.0/``.

    Args:
        fds_path: Existing ``.fds`` file.

    Returns:
        ``True`` if any line was modified, ``False`` if the file already had
        no ``XB=...,3.5/`` SLCF lines.

    Raises:
        FileNotFoundError: If ``fds_path`` does not exist.
    """
    if not fds_path.exists():
        raise FileNotFoundError(fds_path)

    original = fds_path.read_text(encoding="utf-8")
    changes: list[tuple[int, str, str]] = []

    def _repl(match: re.Match) -> str:
        return f"{match.group('prefix')}3.0{match.group('suffix')}"

    new_text, n_subs = _SLCF_Z_MAX_RE.subn(_repl, original)
    if n_subs == 0:
        return False

    # Build a human-readable change list (line numbers).
    for lineno, (old_line, new_line) in enumerate(
        zip(original.splitlines(), new_text.splitlines()), start=1
    ):
        if old_line != new_line:
            changes.append((lineno, old_line, new_line))

    fds_path.write_text(new_text, encoding="utf-8")

    print(f"  [SLCF Z] {fds_path.name}: {n_subs} line(s) changed")
    for lineno, old, new in changes:
        print(f"    line {lineno}: 3.5 → 3.0")
        print(f"      before: {old.strip()}")
        print(f"      after : {new.strip()}")
    return True


def remove_vector_true(fds_path: Path) -> bool:
    """Strip ``VECTOR=.TRUE.`` from any SLCF line in ``fds_path``.

    Only SLCF lines are touched. The function tries to leave the namelist
    syntactically valid by collapsing adjacent commas/whitespace artefacts.

    Returns:
        ``True`` if any change was made, ``False`` otherwise.
    """
    if not fds_path.exists():
        raise FileNotFoundError(fds_path)

    parts = _split_slcf_lines(fds_path.read_text(encoding="utf-8"))
    new_parts: list[str] = []
    n_changed = 0
    changes: list[tuple[int, str, str]] = []
    for lineno, (is_slcf, line) in enumerate(parts, start=1):
        if not is_slcf or not _VECTOR_TRUE_RE.search(line):
            new_parts.append(line)
            continue

        # Replace with a single comma so we don't merge two adjacent attrs;
        # then collapse leftover ", ," → "," and ", /" → " /".
        stripped = _VECTOR_TRUE_RE.sub(",", line)
        stripped = re.sub(r",\s*,", ",", stripped)
        stripped = re.sub(r",\s*/", " /", stripped)
        # Edge case: the only thing on the line was VECTOR=.TRUE.; drop the
        # leading-comma artefact if any.
        stripped = re.sub(r"&SLCF\s*,\s*", "&SLCF ", stripped, count=1, flags=re.IGNORECASE)
        if stripped != line:
            n_changed += 1
            changes.append((lineno, line, stripped))
        new_parts.append(stripped)

    if n_changed == 0:
        return False

    fds_path.write_text("".join(new_parts), encoding="utf-8")
    print(f"  [VECTOR] {fds_path.name}: {n_changed} line(s) cleaned")
    for lineno, old, new in changes:
        print(f"    line {lineno}: removed VECTOR=.TRUE.")
        print(f"      before: {old.strip()}")
        print(f"      after : {new.strip()}")
    return True


def fix_one(fds_path: Path, dry_run: bool = False) -> Tuple[bool, bool]:
    """Apply both fixes to a single ``.fds`` file.

    Args:
        fds_path: Target file.
        dry_run: If ``True``, work on a temporary copy and discard the result —
            no on-disk changes are made.

    Returns:
        ``(slcf_changed, vector_changed)``.
    """
    if dry_run:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp) / fds_path.name
            shutil.copy2(fds_path, tmp_path)
            slcf = fix_slcf_z_range(tmp_path)
            vec = remove_vector_true(tmp_path)
            print(f"  (dry-run; original {fds_path} unchanged)")
            return slcf, vec

    # Real run: back up first.
    backup = fds_path.with_suffix(fds_path.suffix + ".bak")
    shutil.copy2(fds_path, backup)
    slcf = fix_slcf_z_range(fds_path)
    vec = remove_vector_true(fds_path)
    if not (slcf or vec):
        # Nothing changed → backup is redundant. Remove to avoid clutter.
        backup.unlink()
    else:
        print(f"  backup at {backup}")
    return slcf, vec


def fix_path(target: Path, dry_run: bool = False) -> int:
    """Apply fixes to a file or every ``.fds`` under a directory tree.

    Args:
        target: File or directory.
        dry_run: Forwarded to :func:`fix_one`.

    Returns:
        Number of files modified (0 if dry-run or no changes needed).
    """
    if not target.exists():
        raise FileNotFoundError(target)

    if target.is_file():
        if target.suffix.lower() != ".fds":
            raise ValueError(f"{target} is not a .fds file")
        files = [target]
    else:
        files = sorted(target.rglob("*.fds"))

    if not files:
        print(f"No .fds files under {target}")
        return 0

    n_modified = 0
    for f in files:
        print(f"\nProcessing {f}")
        slcf, vec = fix_one(f, dry_run=dry_run)
        if (slcf or vec) and not dry_run:
            n_modified += 1
    print(f"\n{'(dry-run) ' if dry_run else ''}done — {n_modified} file(s) modified")
    return n_modified


# ─── Self-test ─────────────────────────────────────────────────────────────
_BROKEN_FDS = """\
&HEAD CHID='SELFTEST' /
&MESH IJK=100,80,8, XB=-10.0,40.0,-10.0,30.0,0.0,4.0 /
&TIME T_END=300.0 /
&DUMP DT_SLCF=10.0 /

! Bad SLCFs follow (PyroSim defaults) -- script should rewrite them.
&SLCF QUANTITY='TEMPERATURE', VECTOR=.TRUE., CELL_CENTERED=.TRUE.,
      ID='Temperature', XB=0.0,30.0,0.0,20.0,0.0,3.5/
&SLCF QUANTITY='SOOT VISIBILITY', CELL_CENTERED=.TRUE., VECTOR=.TRUE.,
      ID='Visibility', XB=0.0,30.0,0.0,20.0,0.0,3.5/
&SLCF QUANTITY='VOLUME FRACTION', SPEC_ID='CARBON MONOXIDE',
      CELL_CENTERED=.TRUE., ID='CO', XB=0.0,30.0,0.0,20.0,0.0,3.5/

! Decoy: another 3.5 outside SLCF -- must NOT be touched.
&OBST XB=10.0,11.0,3.5,4.5,0.0,1.0/

&TAIL /
"""


def _run_self_test() -> int:
    """Built-in test. Returns exit code (0 = PASS, 1 = FAIL)."""
    print("=" * 60)
    print("fix_pyrosim_fds.py self-test")
    print("=" * 60)

    errors: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "scenario_selftest.fds"
        target.write_text(_BROKEN_FDS, encoding="utf-8")

        slcf, vec = fix_one(target, dry_run=False)
        if not slcf:
            errors.append("SLCF Z-max was not modified")
        if not vec:
            errors.append("VECTOR=.TRUE. was not removed")

        text = target.read_text(encoding="utf-8")
        # SLCF XB Z-max must now be 3.0
        if re.search(r"&SLCF.*XB\s*=\s*[^/]*,\s*3\.5\s*/", text, flags=re.IGNORECASE):
            errors.append("3.5 still present in some SLCF XB after fix")
        if "VECTOR=.TRUE." in text.upper().replace(" ", ""):
            errors.append("VECTOR=.TRUE. still present after fix")

        # Decoy &OBST line: its 3.5 must survive untouched.
        if "&OBST XB=10.0,11.0,3.5,4.5,0.0,1.0/" not in text:
            errors.append("OBST decoy 3.5 was accidentally modified")

        # Backup file exists and is unchanged.
        backup = target.with_suffix(".fds.bak")
        if not backup.exists():
            errors.append("backup .fds.bak was not created")
        elif backup.read_text(encoding="utf-8") != _BROKEN_FDS:
            errors.append("backup contents differ from original")

    if errors:
        print("\nFAIL")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("\nPASS")
    return 0


# ─── Entry point ───────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "target",
        nargs="?",
        type=Path,
        help="Path to a .fds file or directory containing .fds files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without modifying files on disk.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run the built-in test on synthetic .fds content and exit.",
    )
    args = parser.parse_args()

    if args.self_test:
        sys.exit(_run_self_test())
    if args.target is None:
        parser.error("target path is required (or use --self-test)")

    fix_path(args.target, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
