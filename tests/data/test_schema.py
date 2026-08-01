"""Tests for the index schema: parsing, chunk-index mapping, and tolerance of extra fields."""

from __future__ import annotations

import json

from mira.data import Index, MatchEntry


def _entry(**overrides) -> dict:
    entry = {
        "match_id": "m0",
        "shard": "dataset_000000.tar",
        "n_players": 2,
        "chunk_frames": [80, 80, 40],
        "perspectives": [
            {"player_id": 0, "team": 0, "frames": 200, "duration": 10.0},
            {"player_id": 1, "team": 1, "frames": 200, "duration": 10.0},
        ],
    }
    entry.update(overrides)
    return entry


def test_index_parses_and_defaults():
    idx = Index.model_validate({"total_samples": 1, "entries": [_entry()]})
    assert len(idx.entries) == 1
    e = idx.entries[0]
    assert e.match_id == "m0"
    assert e.chunk_indices is None
    assert e.arena is None
    # recording_offset_sec / anchors default when omitted.
    assert e.perspectives[0].recording_offset_sec == 0.0
    assert e.perspectives[0].anchors == []


def test_chunk_id_contiguous_vs_noncontiguous():
    contiguous = MatchEntry.model_validate(_entry())
    assert [contiguous.chunk_id(p) for p in range(3)] == [0, 1, 2]

    sparse = MatchEntry.model_validate(_entry(chunk_indices=[0, 1, 3]))
    assert [sparse.chunk_id(p) for p in range(3)] == [0, 1, 3]


def test_extra_fields_allowed():
    e = MatchEntry.model_validate(_entry(content_id="abc123"))
    assert e.content_id == "abc123"  # pyright: ignore[reportAttributeAccessIssue]  # extra field via extra="allow"


def test_index_load_from_file(tmp_path):
    path = tmp_path / "index.json"
    path.write_text(json.dumps({"total_samples": 1, "entries": [_entry()]}))
    idx = Index.load(path)
    assert idx.total_samples == 1
    assert idx.entries[0].perspectives[1].team == 1
