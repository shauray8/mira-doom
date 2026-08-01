"""Tests for the ActionConfig / ActionTensors container the world model's encoder reads."""

import pytest
import torch

from mira.world_model.actions_config import (
    ActionConfig,
    ActionTensors,
    stack_action_tensors,
)

KEYS = ["W", "A", "S", "D", "Q", "E", "Space", "LShiftKey", "LControlKey"]


def _config() -> ActionConfig:
    return ActionConfig(valid_keys=KEYS, source_fps=20, target_fps=10)


def _filled(batch_size: int, n_steps: int, *, sensitivity: float | None = None) -> ActionTensors:
    at = ActionTensors(_config(), batch_size=batch_size)
    at.key_presses = torch.randint(0, 2, (batch_size, n_steps, len(KEYS)), dtype=torch.int32)
    at.mouse_movements = torch.zeros((batch_size, n_steps, 2), dtype=torch.float32)
    if sensitivity is not None:
        at.game_mouse_sensitivity = torch.full((batch_size,), sensitivity, dtype=torch.float32)
    return at


def test_downsampling_factor():
    assert ActionConfig(valid_keys=KEYS, source_fps=30, target_fps=10).downsampling_factor == 3
    with pytest.raises(ValueError, match="Upsampling not supported"):
        _ = ActionConfig(valid_keys=KEYS, source_fps=10, target_fps=30).downsampling_factor
    with pytest.raises(ValueError, match="integer downsampling"):
        _ = ActionConfig(valid_keys=KEYS, source_fps=25, target_fps=10).downsampling_factor


def test_empty_init_shapes_and_dtypes():
    at = ActionTensors(_config(), batch_size=3)
    assert at.key_presses.shape == (3, 0, len(KEYS)) and at.key_presses.dtype == torch.int32
    assert at.mouse_movements.shape == (3, 0, 2) and at.mouse_movements.dtype == torch.float32
    assert at.game_mouse_sensitivity.shape == (3,) and at.game_mouse_sensitivity.dtype == torch.float32
    assert torch.isnan(at.game_mouse_sensitivity).all()


def test_n_steps():
    assert _filled(2, 7).n_steps == 7


def test_slice_time():
    at = _filled(2, 8)
    sl = at.slice_time(2, 5)
    assert sl.key_presses.shape == (2, 3, len(KEYS))
    assert sl.mouse_movements.shape == (2, 3, 2)
    assert torch.equal(sl.key_presses, at.key_presses[:, 2:5, :])
    assert torch.isnan(sl.game_mouse_sensitivity).all()  # no time dim, carried through


def test_slice_batch():
    at = _filled(4, 6, sensitivity=2.0)
    sl = at.slice_batch(1, 3)
    assert sl.batch_size == 2
    assert torch.equal(sl.key_presses, at.key_presses[1:3])
    assert torch.equal(sl.game_mouse_sensitivity, at.game_mouse_sensitivity[1:3])


def test_cat_time_concatenates_steps():
    a, b = _filled(2, 3), _filled(2, 4)
    cat = a.cat_time(b)
    assert cat.n_steps == 7
    assert torch.equal(cat.key_presses[:, :3], a.key_presses)
    assert torch.equal(cat.key_presses[:, 3:], b.key_presses)


def test_cat_time_sensitivity_merge():
    nan = _filled(1, 2)  # all-NaN
    real = _filled(1, 2, sensitivity=3.0)
    # all-NaN on one side -> take the other side's sensitivity
    assert torch.equal(nan.cat_time(real).game_mouse_sensitivity, real.game_mouse_sensitivity)
    assert torch.equal(real.cat_time(nan).game_mouse_sensitivity, real.game_mouse_sensitivity)
    # both real but disagree -> error
    other = _filled(1, 2, sensitivity=5.0)
    with pytest.raises(ValueError, match="Mouse sensitivities do not match"):
        real.cat_time(other)


def test_stack_action_tensors():
    parts = [_filled(1, 4), _filled(2, 4), _filled(1, 4)]
    stacked = stack_action_tensors(parts)
    assert stacked.batch_size == 4
    assert stacked.key_presses.shape == (4, 4, len(KEYS))
    assert stacked.game_mouse_sensitivity.shape == (4,)


def test_stack_requires_matching_config():
    a = _filled(1, 4)
    b = ActionTensors(ActionConfig(valid_keys=KEYS[:3], source_fps=20, target_fps=10))
    b.key_presses = torch.zeros((1, 4, 3), dtype=torch.int32)
    b.mouse_movements = torch.zeros((1, 4, 2), dtype=torch.float32)
    with pytest.raises(AssertionError, match="same config"):
        stack_action_tensors([a, b])


def test_to_and_clone_return_independent_tensors():
    at = _filled(2, 5, sensitivity=1.5)
    moved = at.to("cpu")
    assert isinstance(moved, ActionTensors) and torch.equal(moved.key_presses, at.key_presses)
    cloned = at.clone()
    cloned.key_presses[:] = 0
    assert not torch.equal(cloned.key_presses, at.key_presses)  # clone is independent


class _TinySensitivityEmbed(torch.nn.Module):
    """A minimal stand-in for the encoder's sensitivity path: mirror its NaN -> token contract.

    The real ActionEncoder lives in ``mira.world_model.layers.action_encoder``; this
    exercises the same ``nan_to_num`` + ``where`` masking that makes all-NaN sensitivity safe to
    backprop through.
    """

    def __init__(self, dim: int = 4):
        super().__init__()
        self.mlp = torch.nn.Linear(1, dim)
        self.dropout_token = torch.nn.Parameter(0.02 * torch.randn(1, dim))

    def forward(self, actions: ActionTensors) -> torch.Tensor:
        sens = actions.game_mouse_sensitivity.view(-1, 1)
        mask = torch.isnan(sens)
        sens = torch.nan_to_num(sens, nan=1.0)
        embed = self.mlp(sens)
        return torch.where(mask, self.dropout_token, embed)


def test_finite_grads_under_all_nan_sensitivity():
    at = _filled(3, 4)  # all-NaN game_mouse_sensitivity, as the keyboard-only loader produces
    assert torch.isnan(at.game_mouse_sensitivity).all()
    model = _TinySensitivityEmbed()
    out = model(at)
    assert torch.isfinite(out).all()
    out.sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "expected gradients to flow"
    assert all(torch.isfinite(g).all() for g in grads)
