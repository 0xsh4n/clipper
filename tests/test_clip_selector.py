"""Timestamp validation, duration fitting and boundary refinement."""

from __future__ import annotations

import pytest

from app.candidate_generator import Candidate
from app.clip_selector import ClipSelector, fit_duration, validate_timestamps
from app.config import load_config
from app.errors import SelectionError


def test_validate_timestamps_accepts_strings():
    assert validate_timestamps("12.5", "40", 130.0) == (12.5, 40.0)


def test_validate_timestamps_clamps_to_source_duration():
    start, end = validate_timestamps(10.0, 999.0, 120.0)
    assert end == 120.0
    assert start == 10.0


@pytest.mark.parametrize(
    "start,end,duration",
    [
        (50.0, 40.0, 120.0),     # end before start
        (10.0, 10.0, 120.0),     # zero length
        ("abc", 30.0, 120.0),    # not numeric
        (float("nan"), 30.0, 120.0),
        (0.0, 30.0, 0.0),        # unknown source duration
    ],
)
def test_validate_timestamps_rejects_nonsense(start, end, duration):
    with pytest.raises(SelectionError):
        validate_timestamps(start, end, duration)


def test_fit_duration_trims_overlong_clips(config):
    start, end = fit_duration(100.0, 160.0, config.clip, 600.0)
    assert end - start == config.clip.max_duration


def test_fit_duration_extends_short_clips(config):
    start, end = fit_duration(100.0, 110.0, config.clip, 600.0)
    assert end - start == pytest.approx(config.clip.min_duration)


def test_fit_duration_shifts_start_when_the_video_ends(config):
    start, end = fit_duration(580.0, 595.0, config.clip, 600.0)
    assert end == 600.0
    assert end - start == pytest.approx(config.clip.min_duration)


def test_refine_boundaries_snaps_to_sentences(transcript, config):
    selector = ClipSelector(config)
    # Deliberately start and end mid-sentence.
    start, end, notes = selector.refine_boundaries(transcript, 11.4, 39.7, transcript.duration)
    assert config.clip.min_duration <= end - start <= config.clip.max_duration
    assert notes  # it reports what it moved
    boundaries = {round(s.start, 2) for s in transcript.segments}
    # The lead-in padding is applied after snapping, so compare without it.
    assert round(start + config.clip.lead_in, 2) in boundaries or start == 0.0


def test_build_clip_keeps_duration_in_range(transcript, config):
    selector = ClipSelector(config)
    candidate = Candidate(id=0, start=9.0, end=48.0, text="x", overall=7.5)
    clip = selector.build_clip(
        candidate,
        transcript,
        selection={"title": "T", "hook": "H", "reason": "R", "score": 9.1},
        video_duration=transcript.duration,
        index=1,
    )
    assert config.clip.min_duration <= clip.duration <= config.clip.max_duration
    assert clip.title == "T" and clip.score == 9.1
    assert clip.text


def test_build_clip_ignores_out_of_range_llm_boundaries(transcript, config):
    selector = ClipSelector(config)
    candidate = Candidate(id=0, start=20.0, end=50.0, text="x")
    clip = selector.build_clip(
        candidate,
        transcript,
        selection={"llm_start": 0.0, "llm_end": 5.0},  # nonsense: outside the window
        video_duration=transcript.duration,
        index=1,
    )
    assert clip.start > 5.0
    assert config.clip.min_duration <= clip.duration <= config.clip.max_duration


def test_build_clips_drops_overlapping_selections(transcript, config):
    selector = ClipSelector(config)
    candidates = [
        Candidate(id=0, start=10.0, end=40.0, text="a", overall=9.0),
        Candidate(id=1, start=12.0, end=42.0, text="b", overall=8.0),  # overlaps id 0
    ]
    clips = selector.build_clips(
        [{"id": 0, "score": 9.0}, {"id": 1, "score": 8.0}],
        candidates,
        transcript,
        video_duration=transcript.duration,
    )
    assert len(clips) == 1


def test_build_clips_raises_when_nothing_is_valid(transcript, config):
    selector = ClipSelector(config)
    with pytest.raises(SelectionError):
        selector.build_clips([{"id": 99}], [], transcript, video_duration=transcript.duration)


def test_clip_to_dict_is_json_shaped(transcript, config):
    selector = ClipSelector(config)
    candidate = Candidate(id=0, start=20.0, end=50.0, text="x", scores={"hook": 9.0})
    clip = selector.build_clip(
        candidate, transcript, selection={"score": 9.1}, video_duration=transcript.duration
    )
    payload = clip.to_dict()
    for key in ("selected", "start", "end", "duration", "title", "hook", "reason", "score"):
        assert key in payload
    assert payload["selected"] is True
