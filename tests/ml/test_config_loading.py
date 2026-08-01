"""Tests for the checkpoint-config cleanup helpers (strip _target_, drop removed fields)."""

from __future__ import annotations

import pytest

from mira.ml.config_loading import drop_removed_fields, strip_hydra_targets


def test_strip_hydra_targets_removes_targets_at_every_level():
    node = {
        "_target_": "pkg.Outer",
        "keep": 1,
        "nested": {"_target_": "pkg.Inner", "value": 2},
        "items": [{"_target_": "pkg.Item", "x": 3}, {"y": 4}],
    }
    assert strip_hydra_targets(node) == {
        "keep": 1,
        "nested": {"value": 2},
        "items": [{"x": 3}, {"y": 4}],
    }


def test_strip_hydra_targets_leaves_scalars_and_target_free_trees():
    assert strip_hydra_targets(7) == 7
    tree = {"a": [1, 2], "b": {"c": 3}}
    assert strip_hydra_targets(tree) == tree


def test_drop_removed_fields_drops_at_noop_value():
    node = {"latent_dim": 32, "is_audio_model": False, "nested": {"is_audio_model": False, "k": 1}}
    assert drop_removed_fields(node, {"is_audio_model": False}) == {"latent_dim": 32, "nested": {"k": 1}}


def test_drop_removed_fields_raises_on_non_noop_value():
    with pytest.raises(ValueError, match="is_audio_model"):
        drop_removed_fields({"is_audio_model": True}, {"is_audio_model": False})


def test_drop_removed_fields_recurses_into_lists():
    node = {"models": [{"is_audio_model": False, "dim": 8}, {"dim": 16}]}
    assert drop_removed_fields(node, {"is_audio_model": False}) == {"models": [{"dim": 8}, {"dim": 16}]}


def test_drop_removed_fields_noop_when_nothing_removed():
    tree = {"a": 1, "b": {"c": [2, 3]}}
    assert drop_removed_fields(tree, {"is_audio_model": False}) == tree
