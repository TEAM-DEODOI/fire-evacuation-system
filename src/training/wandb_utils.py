"""Lightweight Weights & Biases helpers for training entry points.

Both ``train_conv_lstm.py`` and ``train_fno.py`` initialise wandb the same
way — this module factors that boilerplate out so both scripts get the
same set of modes, environment-variable handling, and graceful fallback.

Supported modes (matching wandb's own conventions):

* ``online``   — log to wandb.ai (default; requires ``wandb login``)
* ``offline``  — log locally to ``wandb/`` so the run can be synced later
* ``disabled`` — no-op (still returns a stub so caller code doesn't branch)

The first ``wandb login`` invocation persists credentials in
``~/.netrc``; for headless boxes set the ``WANDB_API_KEY`` env var
instead.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def init_wandb(
    project: str,
    config: Dict[str, Any],
    run_name: Optional[str] = None,
    mode: str = "online",
    tags: Optional[List[str]] = None,
    group: Optional[str] = None,
) -> Optional[Any]:
    """Initialise a wandb run and return the run object (or ``None`` on failure).

    Args:
        project: wandb project name (e.g. ``"fire-evacuation"``).
        config: Hyperparameters / dataset stats to log at the run's start.
        run_name: Optional human-readable run name; wandb auto-generates one
            if omitted.
        mode: ``"online"``, ``"offline"``, or ``"disabled"``.
        tags: Optional list of tags shown on the run page.
        group: Optional run-group label (handy for hyperparameter sweeps).

    Returns:
        The ``wandb.Run`` instance, or ``None`` if wandb is unavailable
        or initialisation failed. Callers should check for ``None``
        before logging.
    """
    if mode not in {"online", "offline", "disabled"}:
        raise ValueError(
            f"wandb mode must be 'online' | 'offline' | 'disabled', got {mode!r}"
        )

    try:
        import wandb
    except ImportError as exc:  # pragma: no cover — wandb is in requirements
        print(f"wandb not installed: {exc}; continuing without")
        return None

    try:
        run = wandb.init(
            project=project,
            name=run_name,
            config=config,
            mode=mode,
            tags=tags or [],
            group=group,
            reinit=True,
        )
        print(
            f"wandb run started: project={project!r} mode={mode} "
            f"name={run.name!r}"
        )
        return run
    except Exception as exc:  # pragma: no cover — surfaced at runtime
        print(f"wandb init failed (continuing without): {exc}")
        return None


def finish_wandb(run: Optional[Any]) -> None:
    """Close a wandb run if one was created."""
    if run is None:
        return
    try:
        run.finish()
    except Exception:  # pragma: no cover
        pass
