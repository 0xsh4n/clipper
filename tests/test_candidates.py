"""Candidate generation and heuristic scoring."""

from __future__ import annotations

from app.candidate_generator import (
    Candidate,
    CandidateGenerator,
    compute_overall,
    heuristic_score,
    normalise_scores,
)
from app.config import ScoringWeights, load_config


def make_candidate(**kwargs) -> Candidate:
    defaults = dict(
        id=0,
        start=100.0,
        end=130.0,
        text="Most people think this is impossible. It turns out ninety percent of them are wrong.",
        word_count=78,
        speech_density=2.6,
        silence_ratio=0.08,
        starts_sentence=True,
        ends_sentence=True,
    )
    defaults.update(kwargs)
    return Candidate(**defaults)


def test_strong_candidate_outscores_sponsor_read():
    strong, _, highlights = heuristic_score(make_candidate())
    weak, penalties, _ = heuristic_score(
        make_candidate(
            text="Hey guys, welcome back. This video is brought to you by our sponsor, use code SAVE20.",
            word_count=18,
            speech_density=0.6,
            silence_ratio=0.55,
            starts_sentence=False,
            ends_sentence=False,
        )
    )
    assert strong > weak
    assert strong >= 7.0
    assert weak <= 2.0
    assert highlights
    assert any("sponsor" in p for p in penalties)


def test_mid_sentence_boundaries_are_penalised():
    aligned, _, _ = heuristic_score(make_candidate())
    unaligned, penalties, _ = heuristic_score(
        make_candidate(starts_sentence=False, ends_sentence=False)
    )
    assert aligned > unaligned
    assert "starts mid-sentence" in penalties
    assert "ends abruptly" in penalties


def test_scores_are_clamped_to_range():
    score, _, _ = heuristic_score(
        make_candidate(text="um uh like you know i mean " * 10, word_count=4, speech_density=0.1, silence_ratio=0.9)
    )
    assert 0.0 <= score <= 10.0


def test_duration_drift_is_penalised():
    near, _, _ = heuristic_score(make_candidate(start=0.0, end=30.0), target_duration=30.0)
    far, _, _ = heuristic_score(make_candidate(start=0.0, end=45.0, word_count=117), target_duration=30.0)
    assert near > far


def test_compute_overall_uses_configured_weights():
    scores = {
        "hook": 10.0,
        "interestingness": 10.0,
        "information_value": 0.0,
        "emotional_impact": 0.0,
        "shareability": 0.0,
        "clarity": 0.0,
        "standalone_context": 0.0,
    }
    # hook + interestingness carry 0.40 of the weight.
    assert compute_overall(scores, ScoringWeights()) == 4.0
    assert compute_overall({}, ScoringWeights()) == 0.0
    # A partially answered score set is renormalised, not silently deflated.
    assert compute_overall({"hook": 8.0}, ScoringWeights()) == 8.0


def test_normalise_scores_filters_and_clamps():
    cleaned = normalise_scores({"hook": "11", "clarity": -3, "bogus": 5, "clipbait": "x"})
    assert cleaned == {"hook": 10.0, "clarity": 0.0}


def test_generator_produces_in_range_windows(transcript):
    config = load_config(overrides={"clip.max_candidates": 10}, environ={})
    candidates = CandidateGenerator(config).generate(transcript)

    assert candidates
    assert len(candidates) <= 10
    for candidate in candidates:
        assert config.clip.min_duration - 0.01 <= candidate.duration <= config.clip.max_duration + 0.01
        assert candidate.text
        assert candidate.word_count > 0
        assert 0.0 <= candidate.silence_ratio <= 1.0
    assert [c.id for c in candidates] == list(range(len(candidates)))


def test_generator_prefers_the_substantive_middle(transcript):
    config = load_config(overrides={"clip.max_candidates": 3}, environ={})
    candidates = CandidateGenerator(config).generate(transcript)
    best = max(candidates, key=lambda c: c.heuristic_score)
    # The sponsor read and the sign-off must not win.
    assert "SAVE20" not in best.text
    assert "like and subscribe" not in best.text.lower()


def test_generator_respects_skip_intro(transcript):
    config = load_config(overrides={"clip.skip_intro_seconds": 20.0}, environ={})
    candidates = CandidateGenerator(config).generate(transcript)
    assert all(c.start >= 20.0 for c in candidates)


def test_candidate_round_trip():
    candidate = make_candidate(scores={"hook": 9.0}, overall=8.4, title="T")
    restored = Candidate.from_dict(candidate.to_dict())
    assert restored.start == candidate.start
    assert restored.scores == {"hook": 9.0}
    assert restored.title == "T"
