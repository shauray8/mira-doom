"""distributed.py is single-process safe: it reports (0, 1) and never touches an uninit process group."""

from __future__ import annotations

import torch

from mira.training.distributed import get_distributed_settings, set_up_distributed


def test_get_distributed_settings_single_process() -> None:
    settings = get_distributed_settings()
    assert settings.rank == 0
    assert settings.world_size == 1
    assert settings.is_main_process is True


def test_set_up_distributed_no_torchrun_is_a_noop(monkeypatch) -> None:
    # No LOCAL_RANK in the environment => no process group is initialized.
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    settings = set_up_distributed()
    assert settings.world_size == 1
    assert settings.rank == 0
    assert not torch.distributed.is_initialized()
