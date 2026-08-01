"""ActionEncoder tests: the prepended initial-action token and the keyboard-only (all-NaN) contract."""

from __future__ import annotations

import torch

from mira.world_model.actions_config import ActionConfig, ActionTensors
from mira.world_model.layers.action_encoder import ActionEncoder

KEYS = ["W", "A", "S", "D", "Q", "E", "Space", "LShiftKey", "LControlKey"]
DIM = 64
TEMPORAL_DOWNSAMPLING = 2


def _actions(batch_size: int, n_steps: int, *, all_nan_sensitivity: bool = True) -> ActionTensors:
    actions = ActionTensors(ActionConfig(valid_keys=KEYS, source_fps=20, target_fps=10), batch_size)
    actions.key_presses = torch.randint(0, 2, (batch_size, n_steps, len(KEYS)), dtype=torch.int32)
    actions.mouse_movements = torch.zeros(batch_size, n_steps, 2, dtype=torch.float32)
    if not all_nan_sensitivity:
        actions.game_mouse_sensitivity = torch.full((batch_size,), 1.5, dtype=torch.float32)
    return actions


def _encoder() -> ActionEncoder:
    return ActionEncoder(
        num_key_presses=len(KEYS),
        dim=DIM,
        temporal_downsampling=TEMPORAL_DOWNSAMPLING,
        dropout_prob=0.1,
        learned_temporal_pool=True,
    )


def test_output_shape_has_prepended_initial_token() -> None:
    encoder = _encoder().eval()
    batch_size, n_steps = 2, 6  # 6 actions -> 3 latent action frames + 1 initial token
    embed = encoder(_actions(batch_size, n_steps))
    n_latent = n_steps // TEMPORAL_DOWNSAMPLING
    assert embed.shape == (batch_size, n_latent + 1, DIM)


def test_initial_token_is_the_learned_parameter() -> None:
    encoder = _encoder().eval()
    embed = encoder(_actions(2, 4))
    # The first token is the broadcast initial_action_token (independent of the action inputs).
    expected = encoder.initial_action_token.expand(2, -1, -1)[:, 0]
    torch.testing.assert_close(embed[:, 0], expected)


def test_finite_embeddings_with_all_nan_sensitivity() -> None:
    encoder = _encoder().eval()
    actions = _actions(3, 8, all_nan_sensitivity=True)
    assert torch.isnan(actions.game_mouse_sensitivity).all()
    embed = encoder(actions)
    assert torch.isfinite(embed).all()


def test_finite_gradients_with_all_nan_sensitivity() -> None:
    encoder = _encoder().train()
    actions = _actions(3, 8, all_nan_sensitivity=True)
    embed = encoder(actions)
    embed.sum().backward()
    for name, param in encoder.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), f"non-finite grad in {name}"


def _pp_encoder(dropout_prob: float = 0.5) -> ActionEncoder:
    return ActionEncoder(
        num_key_presses=len(KEYS),
        dim=DIM,
        temporal_downsampling=TEMPORAL_DOWNSAMPLING,
        dropout_prob=dropout_prob,
        learned_temporal_pool=True,
        dropout_action_per_player=True,
        key_field_names=KEYS,
    )


def test_default_mode_param_topology() -> None:
    # Legacy (per-player off): whole-keyboard dropout token, no per-key key_dropout_embed.
    encoder = _encoder()
    assert encoder.key_dropout_embed is None
    assert encoder.keyboard_dropout_token is not None


def test_per_player_mode_param_topology() -> None:
    encoder = _pp_encoder()
    # Per-key dropout embedding exists, one row per key; the whole-keyboard token is omitted (it
    # would be an unused parameter that aborts DDP).
    assert encoder.key_dropout_embed is not None
    assert encoder.key_dropout_embed.shape[0] == len(KEYS)
    assert encoder.keyboard_dropout_token is None
    # subset_key_mask marks exactly the canonical subset keys (Q/E/Space/Shift/Ctrl).
    mask = encoder.subset_key_mask
    assert isinstance(mask, torch.Tensor)
    expected = {KEYS.index(k) for k in ActionEncoder.DEFAULT_SUBSET_KEYS}
    got = {i for i, on in enumerate(mask.tolist()) if on}
    assert got == expected


def test_per_player_off_by_default_matches_legacy_shape() -> None:
    encoder = _pp_encoder().eval()
    embed = encoder(_actions(2, 6))  # no drop_mask in eval -> no dropout applied
    assert embed.shape == (2, 6 // TEMPORAL_DOWNSAMPLING + 1, DIM)
    assert torch.isfinite(embed).all()


def test_per_player_eval_drop_mask_changes_output() -> None:
    encoder = _pp_encoder().eval()
    actions = _actions(2, 6)
    base = encoder(actions)
    dropped = encoder(actions, drop_mask=torch.ones(2, dtype=torch.bool))  # drop all rows' actions
    assert dropped.shape == base.shape
    assert torch.isfinite(dropped).all()
    assert not torch.allclose(base, dropped)


def test_per_player_training_finite_grads() -> None:
    encoder = _pp_encoder(dropout_prob=0.9).train()  # high prob so dropout reliably fires
    embed = encoder(_actions(4, 8))
    embed.sum().backward()
    for name, param in encoder.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), f"non-finite grad in {name}"
