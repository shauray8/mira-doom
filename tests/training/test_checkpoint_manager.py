"""CheckpointManager: save -> continue_from restores full state; finetune_from loads weights only."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from mira.training.checkpoint_manager import CheckpointManager


class _StubModel(nn.Module):
    """A minimal model with the ``save_checkpoint`` surface CheckpointManager expects."""

    def __init__(self, value: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.linear.weight.fill_(value)

    def save_checkpoint(self, path: str | Path, extra_data: dict[str, Any] | None = None) -> None:
        checkpoint = {"state_dict": self.state_dict(), **(extra_data or {})}
        torch.save(checkpoint, path)


def _manager(model: nn.Module, checkpoint_dir: Path) -> tuple[CheckpointManager, torch.optim.Optimizer]:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    manager = CheckpointManager(
        model, checkpoint_dir=checkpoint_dir, save_every=1, keep_recent=1, model_ema_decay=0.0
    )
    manager.register({"optimizer": optimizer})
    return manager, optimizer


def test_save_then_continue_round_trip(tmp_path: Path) -> None:
    model = _StubModel(1.0)
    manager, optimizer = _manager(model, tmp_path / "run")

    # Move the weights, take an optimizer step, then save at step 5.
    with torch.no_grad():
        model.linear.weight.fill_(2.0)
    optimizer.step()
    manager.maybe_save_checkpoint(step=5, extra_data={"iter_num": 5})
    saved_weight = model.linear.weight.detach().clone()

    # A fresh model+manager continues from the checkpoint dir and restores everything.
    model2 = _StubModel(0.0)
    manager2, _ = _manager(model2, tmp_path / "run")
    start_step = manager2.continue_from(tmp_path / "run" / "checkpoint-5")

    assert start_step == 6  # saved_step + 1
    assert torch.equal(model2.linear.weight, saved_weight)


def test_finetune_from_loads_weights_only(tmp_path: Path) -> None:
    model = _StubModel(3.0)
    manager, _ = _manager(model, tmp_path / "run")
    manager.maybe_save_checkpoint(step=7, extra_data={"iter_num": 7, "latent_mean_std": [0.5, 1.5]})
    saved_weight = model.linear.weight.detach().clone()

    model2 = _StubModel(0.0)
    manager2, _ = _manager(model2, tmp_path / "elsewhere")
    extra = manager2.finetune_from(tmp_path / "run" / "checkpoint-7" / "checkpoint.pth")

    assert torch.equal(model2.linear.weight, saved_weight)
    # Returns the extra payload (without the weights), and does not advance any step counter.
    assert extra["iter_num"] == 7
    assert extra["latent_mean_std"] == [0.5, 1.5]
    assert "state_dict" not in extra


def test_keep_recent_prunes_old_checkpoints(tmp_path: Path) -> None:
    model = _StubModel(1.0)
    manager, _ = _manager(model, tmp_path / "run")
    for step in (1, 2, 3):
        manager.maybe_save_checkpoint(step=step, extra_data={"iter_num": step})

    remaining = sorted(p.name for p in (tmp_path / "run").glob("checkpoint-*"))
    assert remaining == ["checkpoint-3"]  # keep_recent=1
