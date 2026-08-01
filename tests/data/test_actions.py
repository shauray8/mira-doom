import json

import pytest
import torch

from mira.data.actions import KeyVocab, tensorize_actions


def _lines(frames):
    return [json.dumps({"keys": ks}) for ks in frames]


def test_multihot_and_or_downsampling():
    vocab = KeyVocab.default_rl()  # W A S D Q E Space LShiftKey LControlKey
    # source 20 -> target 10 => factor 2: OR over each pair of frames
    lines = _lines([["W"], ["A"], ["Space"], []])
    out = tensorize_actions(lines, vocab, source_fps=20, target_fps=10)
    assert out.shape == (2, 9)
    assert out.dtype == torch.int32
    # step 0 = OR({W},{A}); step 1 = OR({Space},{})
    assert out[0, vocab.keys.index("W")] == 1 and out[0, vocab.keys.index("A")] == 1
    assert out[0, vocab.keys.index("Space")] == 0
    assert out[1, vocab.keys.index("Space")] == 1
    assert set(out.unique().tolist()) <= {0, 1}


def test_partial_trailing_window_dropped():
    vocab = KeyVocab.default_rl()
    out = tensorize_actions(_lines([["W"], ["A"], ["S"]]), vocab, source_fps=20, target_fps=10)
    assert out.shape == (1, 9)  # 3 frames / factor 2 -> 1 full step, last dropped


def test_keep_last_partial_emits_short_final_window():
    vocab = KeyVocab.default_rl()
    # 3 frames / factor 2 -> 1 full step + 1 short window; keep_last_partial OR-s the leftover frame
    out = tensorize_actions(
        _lines([["W"], ["A"], ["S"]]), vocab, source_fps=20, target_fps=10, keep_last_partial=True
    )
    assert out.shape == (2, 9)
    assert out[0, vocab.keys.index("W")] == 1 and out[0, vocab.keys.index("A")] == 1
    assert out[1, vocab.keys.index("S")] == 1  # the final step carries the real key, not zeros
    assert torch.count_nonzero(out[1]) == 1


def test_keep_last_partial_noop_when_window_is_full():
    vocab = KeyVocab.default_rl()  # 4 frames / factor 2 -> 2 full steps, nothing trailing
    lines = _lines([["W"], ["A"], ["Space"], []])
    full = tensorize_actions(lines, vocab, source_fps=20, target_fps=10)
    kept = tensorize_actions(lines, vocab, source_fps=20, target_fps=10, keep_last_partial=True)
    assert torch.equal(full, kept)  # no partial window -> identical to the default


def test_no_downsampling():
    vocab = KeyVocab.default_rl()
    out = tensorize_actions(_lines([["W"], []]), vocab, source_fps=20, target_fps=20)
    assert out.shape == (2, 9)


def test_empty_input_is_well_shaped():
    out = tensorize_actions([], KeyVocab.default_rl(), source_fps=20, target_fps=10)
    assert out.shape == (0, 9)
    assert out.dtype == torch.int32


def test_unknown_key_policies():
    vocab_err = KeyVocab.default_rl(on_unknown="error")
    with pytest.raises(ValueError):
        tensorize_actions(_lines([["W", "ZZZ"], []]), vocab_err, 20, 10)

    vocab_ignore = KeyVocab.default_rl(on_unknown="ignore")
    out = tensorize_actions(_lines([["W", "ZZZ"], []]), vocab_ignore, 20, 10)
    assert out[0, vocab_ignore.keys.index("W")] == 1


def test_warn_policy_warns_once_per_distinct_key():
    vocab = KeyVocab.default_rl()  # default on_unknown="warn"
    with pytest.warns(UserWarning) as record:
        tensorize_actions(_lines([["ZZZ"], ["ZZZ"], ["ZZZ"], ["ZZZ"]]), vocab, 20, 10)
    assert sum(issubclass(w.category, UserWarning) for w in record) == 1


def test_non_integer_downsampling_raises():
    with pytest.raises(ValueError):
        tensorize_actions(_lines([[], []]), KeyVocab.default_rl(), source_fps=20, target_fps=3)


def test_invalid_on_unknown_rejected():
    with pytest.raises(ValueError):
        KeyVocab.default_rl(on_unknown="Error")  # pyright: ignore[reportArgumentType]
