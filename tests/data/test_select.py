import random

import pytest

from mira.data.dataset import DoomWebDataset, chunk_key, parse_chunk_key

sel = DoomWebDataset._select_perspectives


def test_chunk_key_roundtrip_and_guard():
    mid = "2026-05-10T01-25-13Z-fdcbf9"
    assert chunk_key(mid, 7) == f"{mid}_c00007"
    assert parse_chunk_key(chunk_key(mid, 7)) == (mid, 7)
    assert parse_chunk_key("not-a-chunk-key") is None
    with pytest.raises(ValueError):  # a '.' would break WebDataset member parsing
        chunk_key("bad.match.id", 0)


def test_all_and_single():
    assert sel("all", 4, random.Random(0)) == [0, 1, 2, 3]
    assert sel("player1", 4, random.Random(0)) == [0]
    assert sel("player4", 4, random.Random(0)) == [3]
    assert sel(2, 4, random.Random(0)) == [2]


def test_random_is_seeded_and_in_range():
    pick = sel("random", 4, random.Random(0))
    assert pick == [random.Random(0).randrange(4)]
    assert 0 <= pick[0] < 4


def test_out_of_range_and_invalid():
    with pytest.raises(ValueError):
        sel(4, 4, random.Random(0))
    with pytest.raises(ValueError):
        sel("player5", 4, random.Random(0))
    with pytest.raises(ValueError):
        sel("bogus", 4, random.Random(0))
