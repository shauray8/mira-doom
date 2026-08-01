from mira.data.events import (
    Event,
    events_in_frame_window,
    overlaps_any,
    parse_anchors,
    replay_spans,
)


def test_event_to_frame_offset_corrected():
    e = Event(1, "GoalScored", master_sec=10.0)
    # offset shifts the event earlier on this perspective's timeline
    assert e.frame_index(fps=20, recording_offset_sec=0.0) == 200
    assert e.frame_index(fps=20, recording_offset_sec=0.5) == 190


def test_parse_anchors_dict_and_obj():
    d = [{"event_type": 1, "event_name": "GoalScored", "master_sec": 1.0}]
    assert parse_anchors(d)[0].event_name == "GoalScored"

    class A:
        event_type, event_name, master_sec = 2, "Demolition", 2.0

    assert parse_anchors([A()])[0].event_name == "Demolition"


def test_replay_spans_pairing_and_clamping():
    anchors = parse_anchors(
        [
            {"event_type": 3, "event_name": "GoalReplayStarted", "master_sec": 5.0},
            {"event_type": 4, "event_name": "GoalReplayEnded", "master_sec": 10.0},
            {"event_type": 3, "event_name": "GoalReplayStarted", "master_sec": 95.0},  # dangling
        ]
    )
    spans = replay_spans(anchors, fps=20, recording_offset_sec=0.0, n_frames=2000)
    assert spans[0] == (100, 200)
    assert spans[1] == (1900, 2000)  # dangling start extends to end


def test_overlaps_and_window():
    spans = [(100, 200)]
    assert overlaps_any(150, 260, spans)
    assert not overlaps_any(0, 100, spans)
    events = parse_anchors([{"event_type": 1, "event_name": "GoalScored", "master_sec": 7.5}])
    win = events_in_frame_window(events, 100, 200, fps=20, recording_offset_sec=0.0)  # frame 150
    assert len(win) == 1
