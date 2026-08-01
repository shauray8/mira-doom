"""VideoCodec tests: preprocessing, normalisation, and the encode->decode round-trip.

The round-trip and preprocessing-on-the-model tests build the full codec and so skip when the
DINOv3 backbone is unavailable (see conftest.build_codec_or_skip)."""

from __future__ import annotations

import torch

from mira.codec import VideoCodecOutputs
from mira.world_model.actions_config import ActionConfig, ActionTensors
from tests.codec.conftest import VIDEO, build_codec_or_skip, random_video

KEYS = ["W", "A", "S", "D", "Q", "E", "Space", "LShiftKey", "LControlKey"]


def _batch_with_video(video: torch.Tensor):
    from mira.data.batch import VideoActionBatch

    batch_size, n_frames = video.shape[0], video.shape[1]
    actions = ActionTensors(ActionConfig(valid_keys=KEYS, source_fps=20, target_fps=20), batch_size)
    actions.key_presses = torch.zeros((batch_size, n_frames, len(KEYS)), dtype=torch.int32)
    actions.mouse_movements = torch.zeros((batch_size, n_frames, 2), dtype=torch.float32)
    return VideoActionBatch(video=video, actions=actions)


def test_preprocess_resizes_to_target_and_normalises() -> None:
    codec = build_codec_or_skip()
    batch = _batch_with_video(random_video(batch=2, frames=40, height=200, width=360))
    codec.preprocess_batch(batch)
    assert batch.video.shape == (2, 40, 3, VIDEO.height, VIDEO.width)
    assert batch.video.dtype == torch.float32
    assert 0.0 <= float(batch.video.min()) and float(batch.video.max()) <= 1.0


def test_preprocess_pads_right_then_bottom() -> None:
    codec = build_codec_or_skip()
    # Taller-than-target aspect -> right pad; wider-than-target aspect -> bottom pad. Both resize
    # to the target H/W, so we only assert the final shape (the pad branch is exercised either way).
    for height, width in [(200, 320), (200, 400)]:
        batch = _batch_with_video(random_video(batch=1, frames=40, height=height, width=width))
        codec.preprocess_batch(batch)
        assert batch.video.shape[-2:] == (VIDEO.height, VIDEO.width)


def test_preprocess_passthrough_when_already_target() -> None:
    codec = build_codec_or_skip()
    video = random_video(batch=1, frames=40, height=VIDEO.height, width=VIDEO.width)
    batch = _batch_with_video(video)
    codec.preprocess_batch(batch)
    # Exact-aspect, exact-size input is only divided by 255 (no pad / no resize).
    assert torch.allclose(batch.video, video.float() / 255.0)


def test_normalize_video_trims_and_rescales() -> None:
    codec = build_codec_or_skip()
    video = torch.rand(1, 50, 3, VIDEO.height, VIDEO.width)  # more frames than encoder timesteps
    out = codec.normalize_video(video, trim_video=True)
    assert out.shape[1] == VIDEO.timesteps  # trimmed to 40
    assert torch.allclose(out, (video[:, : VIDEO.timesteps] - 0.5) / 0.5)


def test_encode_decode_round_trip_and_td_factor() -> None:
    codec = build_codec_or_skip()
    assert (codec.temporal_downsampling, codec.spatial_downsampling) == (2, 32)

    video = (torch.rand(1, 40, 3, VIDEO.height, VIDEO.width) * 2) - 1  # [-1, 1]
    with torch.no_grad():
        input_video, encoder_output = codec.encode(video, trim_video=True)
        decoded = codec.decode(encoder_output.z)

    # td=2: 40 frames -> 20 latents; spatial /32: 288x512 -> 9x16.
    assert encoder_output.z.shape == (1, 20, 32, 9, 16)
    assert decoded.shape == (1, 40, 3, VIDEO.height, VIDEO.width)
    assert input_video.shape == (1, 40, 3, VIDEO.height, VIDEO.width)


def test_forward_from_uint8_batch() -> None:
    codec = build_codec_or_skip()
    batch = _batch_with_video(random_video(batch=1, frames=40, height=200, width=360))
    with torch.no_grad():
        out = codec.forward(batch)
    assert isinstance(out, VideoCodecOutputs)
    assert out.output_video.shape == (1, 40, 3, VIDEO.height, VIDEO.width)
    assert out.z.shape == (1, 20, 32, 9, 16)
    assert out.dino_features is not None and len(out.dino_features) == 7
