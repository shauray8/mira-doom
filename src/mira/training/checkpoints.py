"""Resolving checkpoint paths (local or W&B run URL) and re-attaching W&B runs on resume."""

from __future__ import annotations

import logging
import re
from pathlib import Path

_WANDB_URL_RE = re.compile(r"https://wandb\.ai/([^/]+)/([^/]+)/runs/([^/]+)")

logger = logging.getLogger(__name__)


def wandb_run_id_from_url(checkpoint: str | Path) -> str | None:
    """Return the W&B run id if ``checkpoint`` is a W&B run URL, else None.

    Used to re-attach training to the original W&B run when resuming/continuing from it, so its
    history stays in one continuous run.
    """
    match = _WANDB_URL_RE.match(str(checkpoint))
    return match.group(3) if match else None


def resume_wandb_run_id(continue_from: str | Path | None, output_dir: str | Path) -> str | None:
    """W&B run id to re-attach to when resuming into ``output_dir``, or None to start fresh.

    So crash recovery / auto-resume continues the *same* W&B run rather than opening a new one:
    - a W&B-URL ``continue_from`` carries the id directly;
    - a local ``continue_from`` reads the ``wandb_run_id.txt`` sidecar in the source run's dir;
    - otherwise (auto-resume into the same output_dir), read the sidecar there.

    Keying off the sidecar means a *crashed fine-tune* (which wrote one on its first launch)
    re-attaches on resubmit, while a brand-new run (no sidecar yet) gets a fresh run.
    """
    if continue_from:
        run_id = wandb_run_id_from_url(continue_from)
        if run_id is not None:
            return run_id
        try:
            source_output_dir = resolve_checkpoint(continue_from).parent.parent
        except (FileNotFoundError, ValueError):
            source_output_dir = None
        if source_output_dir is not None:
            sidecar = source_output_dir / "wandb_run_id.txt"
            if sidecar.is_file():
                return sidecar.read_text(encoding="utf-8").strip()

    sidecar = Path(output_dir) / "wandb_run_id.txt"
    if sidecar.is_file():
        return sidecar.read_text(encoding="utf-8").strip()
    return None


def _find_latest_checkpoint(output_dir: Path) -> Path:
    """Find the latest ``checkpoint-{step}/checkpoint.pth`` in an output directory."""
    checkpoint_dirs = sorted(
        output_dir.glob("checkpoint-*/checkpoint.pth"),
        key=lambda p: int(p.parent.name.split("-")[1]),
    )
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoints found in {output_dir}")
    checkpoint_path = checkpoint_dirs[-1]
    logger.info(f"Resolved to latest checkpoint: {checkpoint_path}")
    return checkpoint_path


def resolve_checkpoint(checkpoint: str | Path) -> Path:
    """Resolve a checkpoint path from a local path or W&B run URL.

    Accepts:
    - A direct path to a ``checkpoint.pth`` file.
    - A checkpoint directory (``checkpoint-{step}/``) containing ``checkpoint.pth``.
    - An output directory containing ``checkpoint-{step}/`` subdirectories (picks latest).
    - A W&B run URL like ``https://wandb.ai/entity/project/runs/run_id/overview``.
    """
    if not isinstance(checkpoint, Path):
        # First check if it's a W&B URL; if so, resolve to the run's output_dir.
        match = _WANDB_URL_RE.match(checkpoint)
        if match:
            entity, project, run_id = match.groups()

            import wandb  # noqa: PLC0415 -- optional dep, used only for W&B-URL checkpoints

            api = wandb.Api()
            run = api.run(f"{entity}/{project}/{run_id}")
            output_dir = Path(run.config["run"]["output_dir"])
            logger.info(f"W&B run output_dir for {checkpoint}: {output_dir}")
            return _find_latest_checkpoint(output_dir)

        path = Path(checkpoint)
    else:
        # A Path is always local.
        path = checkpoint

    # Direct path to a .pth file.
    if path.suffix == ".pth":
        return path

    # A checkpoint-{step} dir containing checkpoint.pth.
    if (path / "checkpoint.pth").is_file():
        return path / "checkpoint.pth"

    # An output dir containing checkpoint-{step}/ subdirectories.
    return _find_latest_checkpoint(path)
