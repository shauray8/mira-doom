"""Checkpoint saving, retention, EMA-weight swapping, and resume."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import nn

from .checkpoints import resolve_checkpoint
from .ema import ModelEMA
from .tracker import periodic_event

logger = logging.getLogger(__name__)


class _StatefulComponent(Protocol):
    def state_dict(self) -> dict[str, Any]: ...
    # `object` to ignore the return type since nn.Module and optimizers/schedulers return different things.
    def load_state_dict(self, state_dict: dict[str, Any]) -> object: ...


class CheckpointManager:
    """Manages checkpoint saving, retention, EMA weight swapping, and resume.

    The model is saved with EMA weights swapped in (for downstream eval), while ``training_state.pth``
    holds the raw per-component state needed to resume training exactly via :meth:`continue_from`.
    """

    def __init__(
        self,
        model: nn.Module,
        checkpoint_dir: str | Path,
        *,
        save_every: int | str = -1,
        keep_recent: int = 1,
        keep_permanent_every: int | str = -1,
        total_steps: int | None = None,
        model_ema_decay: float = 0.0,  # 0.0 for no EMA, otherwise the decay rate for the EMA model
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = save_every
        self.keep_recent = keep_recent
        self.keep_permanent_every = keep_permanent_every
        self.total_steps = total_steps

        self.model_ema = ModelEMA(model, decay=model_ema_decay)
        self._components: dict[str, _StatefulComponent] = {}
        self.register({"model": model, "model_ema": self.model_ema})

        self.recent_checkpoints: list[tuple[int, Path]] = self._discover_existing_checkpoints()

    def register(self, components: dict[str, _StatefulComponent]) -> None:
        """Register named stateful components (each with state_dict/load_state_dict).

        These are saved into ``training_state.pth`` and restored on :meth:`continue_from`.
        """
        for name, component in components.items():
            if name in self._components:
                raise ValueError(f"Component {name!r} already registered")
            self._components[name] = component

    def _discover_existing_checkpoints(self) -> list[tuple[int, Path]]:
        """Find any checkpoint-* dirs already in checkpoint_dir, sorted by step.

        Lets retention bookkeeping survive a resume into the same directory.
        """
        found: list[tuple[int, Path]] = []
        for p in self.checkpoint_dir.glob("checkpoint-*"):
            if not p.is_dir():
                continue
            try:
                step = int(p.name.split("-", 1)[1])
            except (IndexError, ValueError):
                continue
            found.append((step, p))
        found.sort(key=lambda x: x[0])
        return found

    def _should_save(self, step: int) -> bool:
        return self.save_every != -1 and periodic_event(step, self.save_every, self.total_steps)

    def _is_permanent(self, step: int) -> bool:
        if self.keep_permanent_every == -1:
            return False
        return periodic_event(step, self.keep_permanent_every, self.total_steps)

    @property
    def latest_checkpoint(self) -> Path | None:
        if len(self.recent_checkpoints) == 0:
            return None
        return self.recent_checkpoints[-1][1] / "checkpoint.pth"

    def finetune_from(self, checkpoint: str | Path) -> dict[str, Any]:
        """Load *only* the model weights from a checkpoint, for fine-tuning.

        Unlike :meth:`continue_from`, this restores neither optimizer/scheduler state nor the step
        counter: training starts fresh at step 0 from these weights.

        Returns:
            The extra data dict saved in the checkpoint (if any), excluding the model weights.
        """
        src = resolve_checkpoint(checkpoint)
        logger.info(f"Fine-tuning: loading model weights from {src}")
        ckpt = torch.load(src, map_location="cpu", weights_only=False)
        # Defers to the model's own load_state_dict (strict for the codec). The multiplayer world
        # model overrides it to load single-player weights, remapping keys and dropping mismatched
        # shapes; that remap lives in the model, not here.
        self._components["model"].load_state_dict(ckpt["state_dict"])
        return {k: v for k, v in ckpt.items() if k != "state_dict"}

    def continue_from(self, checkpoint: str | Path) -> int:
        """Load every registered component from ``checkpoint`` (transparent crash recovery).

        ``checkpoint`` may point at a ``checkpoint.pth``, a ``checkpoint-{step}/`` dir, an output
        dir (latest is picked), or a W&B run URL. Returns the step at which the training loop should
        begin (``saved_step + 1``).
        """
        src = resolve_checkpoint(checkpoint).parent

        state_path = src / "training_state.pth"
        if not state_path.is_file():
            raise FileNotFoundError(f"No training_state.pth in {src}")
        logger.info(f"Loading training state from {state_path}")
        state = torch.load(state_path, map_location="cpu", weights_only=False)

        for name, comp in self._components.items():
            if name in state:
                comp.load_state_dict(state[name])
            else:
                logger.warning(f"Component {name!r} missing from checkpoint; using current state")
        start_step = int(state["step"]) + 1
        logger.info(f"Continuing from {src} at step {state['step']} (next step {start_step})")
        return start_step

    def maybe_save_checkpoint(
        self,
        step: int,
        extra_data: dict[str, Any] | None = None,
        final: bool = False,
    ) -> None:
        """Save a checkpoint and apply the retention policy."""
        if not self._should_save(step) and not final:
            return None

        extra_data = dict(extra_data or {})
        ckpt_dir = self.checkpoint_dir / f"checkpoint-{step}"
        # Write into a sibling temp dir and atomically rename into place, so a crash mid-save never
        # leaves a partial `checkpoint-{step}/` for auto-resume to pick up as the latest checkpoint.
        tmp_dir = self.checkpoint_dir / f".checkpoint-{step}.tmp"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True)
        logger.info(f"Saving checkpoint {ckpt_dir}")

        # Save model checkpoint (with EMA weights if available, for downstream eval).
        model = self._components["model"]
        with self.model_ema.average_parameters():
            model.save_checkpoint(tmp_dir / "checkpoint.pth", extra_data)  # type: ignore[attr-defined]

        # Save training state: per-component state_dicts, plus the step.
        training_state: dict[str, Any] = {name: comp.state_dict() for name, comp in self._components.items()}
        training_state["step"] = step
        torch.save(training_state, tmp_dir / "training_state.pth")

        if ckpt_dir.exists():  # a re-save at the same step (e.g. final == last periodic step)
            shutil.rmtree(ckpt_dir)
        os.replace(tmp_dir, ckpt_dir)

        self.recent_checkpoints.append((step, ckpt_dir))

        # Temporary (non-permanent) checkpoints are eligible for deletion.
        temp_checkpoints = [
            idx for idx, (s, _) in enumerate(self.recent_checkpoints) if not self._is_permanent(s)
        ]
        if final:
            # For the final checkpoint, clean up all remaining temporary checkpoints.
            for idx in temp_checkpoints:
                # Sanity check: don't overwrite the final checkpoint.
                if (old_checkpoint := self.recent_checkpoints[idx][1]) != ckpt_dir:
                    shutil.rmtree(old_checkpoint)
        elif len(temp_checkpoints) > self.keep_recent:
            # Otherwise retain `keep_recent` temporary checkpoints.
            old_idx = temp_checkpoints[0]
            old_checkpoint = self.recent_checkpoints[old_idx][1]
            self.recent_checkpoints.pop(old_idx)
            shutil.rmtree(old_checkpoint)
