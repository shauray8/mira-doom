"""ModelEMA: the parameter average tracks correctly and swap/restore is exact."""

from __future__ import annotations

import torch
from torch import nn

from mira.training.ema import ModelEMA


def _model(value: float) -> nn.Linear:
    m = nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        m.weight.fill_(value)
    return m


def test_average_parameters_swaps_then_restores_exactly() -> None:
    model = _model(1.0)
    ema = ModelEMA(model, decay=0.9, unbias=False)

    # Move the live weights away from the EMA state, then step the EMA once.
    with torch.no_grad():
        model.weight.fill_(2.0)
    ema.step()  # state := 0.1 * 2.0 + 0.9 * 1.0 = 1.1
    expected_ema = torch.full_like(model.weight, 1.1)
    assert torch.allclose(ema.state["weight"], expected_ema)

    raw_before = model.weight.detach().clone()  # 2.0 everywhere
    with ema.average_parameters():
        # Inside the context the live weights are the EMA values.
        assert torch.allclose(model.weight, expected_ema)
    # After the context the raw weights are restored bit-for-bit.
    assert torch.equal(model.weight, raw_before)


def test_decay_zero_is_a_noop() -> None:
    model = _model(3.0)
    ema = ModelEMA(model, decay=0.0)
    raw = model.weight.detach().clone()

    ema.step()  # no update
    with ema.average_parameters():  # no swap
        assert torch.equal(model.weight, raw)
    assert torch.equal(model.weight, raw)


def test_state_dict_round_trip() -> None:
    model = _model(1.0)
    ema = ModelEMA(model, decay=0.5, unbias=True)
    with torch.no_grad():
        model.weight.fill_(5.0)
    ema.step()
    ema.step()

    restored = ModelEMA(_model(0.0), decay=0.5, unbias=True)
    restored.load_state_dict(ema.state_dict())
    assert restored.count == ema.count
    assert torch.allclose(restored.state["weight"], ema.state["weight"])
