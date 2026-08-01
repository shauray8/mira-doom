"""Tests for the VideoActionBatch container (video + actions, no audio)."""

import pytest
import torch

from mira.data.batch import VideoActionBatch
from mira.world_model.actions_config import ActionConfig, ActionTensors

KEYS = ["W", "A", "S", "D", "Q", "E", "Space", "LShiftKey", "LControlKey"]


def _batch(b: int = 2, t: int = 6) -> VideoActionBatch:
    actions = ActionTensors(ActionConfig(valid_keys=KEYS, source_fps=20, target_fps=10), batch_size=b)
    actions.key_presses = torch.randint(0, 2, (b, t, len(KEYS)), dtype=torch.int32)
    actions.mouse_movements = torch.zeros((b, t, 2), dtype=torch.float32)
    actions.game_mouse_sensitivity = torch.full((b,), float("nan"), dtype=torch.float32)
    video = torch.randint(0, 256, (b, t, 3, 8, 8), dtype=torch.uint8)
    return VideoActionBatch(video=video, actions=actions)


def test_len_matches_video_and_actions():
    assert len(_batch(b=3)) == 3


def test_len_asserts_video_action_batch_agree():
    batch = _batch(b=2)
    batch.actions.batch_size = 3  # force a mismatch
    with pytest.raises(AssertionError):
        len(batch)


def test_slice_time_slices_video_and_actions_consistently():
    batch = _batch(b=2, t=8)
    sl = batch.slice_time(2, 5, fps=10)
    assert sl.video.shape == (2, 3, 3, 8, 8)
    assert torch.equal(sl.video, batch.video[:, 2:5])
    assert sl.actions.key_presses.shape == (2, 3, len(KEYS))
    assert torch.equal(sl.actions.key_presses, batch.actions.key_presses[:, 2:5, :])


def test_cat_time():
    a, b = _batch(b=2, t=3), _batch(b=2, t=4)
    cat = a.cat_time(b)
    assert cat.video.shape == (2, 7, 3, 8, 8)
    assert cat.actions.n_steps == 7


def test_to_propagates_to_actions():
    batch = _batch()
    moved = batch.to("cpu")
    # `to` builds a fresh container; actions must be carried through (not dropped) and equal.
    assert isinstance(moved.actions, ActionTensors)
    assert moved.actions is not batch.actions
    assert torch.equal(moved.actions.key_presses, batch.actions.key_presses)
    assert moved.video.device == batch.video.device


def test_clone_propagates_to_actions_and_is_independent():
    batch = _batch()
    cloned = batch.clone()
    assert cloned.actions is not batch.actions
    assert torch.equal(cloned.actions.key_presses, batch.actions.key_presses)
    cloned.video[:] = 0
    cloned.actions.key_presses[:] = 0
    # mutating the clone must not touch the original (deep copy of both video and actions)
    assert not torch.equal(cloned.video, batch.video)
    assert not torch.equal(cloned.actions.key_presses, batch.actions.key_presses)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="pin_memory needs CUDA")
def test_pin_memory_propagates_to_actions():
    batch = _batch()
    pinned = batch.pin_memory()
    assert pinned.video.is_pinned()
    assert pinned.actions.key_presses.is_pinned()
