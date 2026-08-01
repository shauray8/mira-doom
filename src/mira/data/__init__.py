"""mira.data: loader for MIRA's WebDataset layout (Doom 2-player).

Each sample bundles the perspectives of one chunk; a clip is taken from within one chunk.

Public API:
    DoomWebDataset, MatchClip — load time-aligned per-player clips (random access / streaming)
    Index, MatchEntry, Perspective, Anchor — typed schema for the dataset index (`index.json`)
    Vec3, Quat, GameInfo, BallState, CarState, FrameState — typed per-frame game state
    KeyVocab, DEFAULT_KEYS, tensorize_actions — multi-hot keyboard action parsing
    Event, replay_spans — discrete game events with frame-index mapping

"""

from .actions import DEFAULT_KEYS, KeyVocab, tensorize_actions
from .dataset import MatchClip, DoomWebDataset
from .events import Event, replay_spans
from .schema import Anchor, Index, MatchEntry, Perspective
from .state import BallState, CarState, FrameState, GameInfo, Quat, Vec3

__all__ = [
    "DoomWebDataset",
    "MatchClip",
    "Index",
    "MatchEntry",
    "Perspective",
    "Anchor",
    "Vec3",
    "Quat",
    "GameInfo",
    "BallState",
    "CarState",
    "FrameState",
    "KeyVocab",
    "DEFAULT_KEYS",
    "tensorize_actions",
    "Event",
    "replay_spans",
]
