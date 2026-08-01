"""Shared builders for synthetic physics clips, exposed as factory fixtures.

These let physics and viz tests construct a `MatchClip` with scripted per-frame state (moving vs
frozen ball, anchor windows, demolitions) without reading any real data.
"""

from __future__ import annotations

import pytest
import torch

from mira.data.dataset import MatchClip
from mira.data.events import Event
from mira.data.state import CarState, FrameState


def _car(pid: int, team: int, *, local: bool = False, loc=(0.0, 0.0, 17.0), vel=(0.0, 0.0, 0.0),
         boost: float = 0.5, **flags) -> CarState:  # fmt: skip
    car: CarState = {
        "player_id": pid,
        "team": team,
        "attacker_player_id": -1,
        "is_local": local,
        "location": {"x": loc[0], "y": loc[1], "z": loc[2]},
        "velocity": {"x": vel[0], "y": vel[1], "z": vel[2]},
        "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        "angular_velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
        "boost_amount": boost,
        "is_on_ground": True,
        "is_supersonic": False,
    }
    car.update(flags)  # type: ignore[typeddict-item]  # tests may set attacker_player_id
    return car


def _frame(local_pid: int, *, ball=(0.0, 0.0, 93.0), tr: float = 120.0) -> FrameState:
    return {
        "game": {"time_remaining": tr, "score_blue": 1, "score_orange": 0, "is_overtime": False},
        "ball": {
            "location": {"x": ball[0], "y": ball[1], "z": ball[2]},
            "velocity": {"x": 10.0, "y": 5.0, "z": 0.0},
            "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            "angular_velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
        },
        "cars": [
            _car(10, 0, local=(local_pid == 10)),
            _car(11, 0, local=(local_pid == 11)),
            _car(20, 1, local=(local_pid == 20)),
            _car(21, 1, local=(local_pid == 21)),
        ],
    }


def _clip(p: int = 4, t: int = 6, *, tr: float = 120.0, moving: bool = True, ball_ys=None) -> MatchClip:
    player_ids = [10, 11, 20, 21][:p]
    # ball_ys overrides per-step ball-y (to script moving/frozen segments); else use `moving`.
    ys = ball_ys if ball_ys is not None else [(ti * 200.0 if moving else 0.0) for ti in range(t)]
    phys = [[_frame(player_ids[pi], ball=(0.0, ys[ti], 93.0), tr=tr) for ti in range(t)] for pi in range(p)]
    return MatchClip(
        match_id="m0",
        chunk_idx=0,
        clip_id=0,
        perspectives=list(range(p)),
        frame_indices=list(range(t)),
        stride=2,
        target_fps=10,
        src_fps=20.0,  # stride 2 * target_fps 10
        player_ids=player_ids,
        teams=[0, 0, 1, 1][:p],
        recording_offsets=[0.0] * p,
        metadata=[{"arena": "cs_p"} for _ in range(p)],
        actions=torch.zeros((p, t, 9), dtype=torch.int32),
        events=[Event(1, "GoalScored", master_sec=0.4)],
        frames=torch.zeros((p, t, 3, 16, 16), dtype=torch.uint8),
        physics=phys,
    )


def _anchor_events() -> list[Event]:
    # one goal lifecycle: kickoff 0-4, live 4-40, goal->replay pause 40-44, replay 44-54, then live.
    return [
        Event(4, "KickoffStarted", 0.0),
        Event(0, "KickoffEnded", 4.0),
        Event(1, "GoalScored", 40.0),
        Event(2, "GoalReplayStarted", 44.0),
        Event(3, "GoalReplayEnded", 54.0),
        Event(4, "KickoffStarted", 54.0),
        Event(0, "KickoffEnded", 58.0),
    ]


def _clip_with_anchors(ball_ys, g_idx, t) -> MatchClip:
    clip = _clip(t=t, ball_ys=ball_ys)
    clip.global_frame_indices = g_idx
    clip.metadata = [{"anchors": _anchor_events()} for _ in range(4)]
    return clip


@pytest.fixture
def make_frame():
    return _frame


@pytest.fixture
def make_clip():
    return _clip


@pytest.fixture
def anchor_events():
    return _anchor_events()


@pytest.fixture
def make_clip_with_anchors():
    return _clip_with_anchors
