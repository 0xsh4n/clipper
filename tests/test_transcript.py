"""Transcript parsing, querying and round-tripping."""

from __future__ import annotations

from app.transcript import Transcript, build_transcript, shift_words


def test_build_transcript_sorts_reindexes_and_drops_empties():
    transcript = build_transcript(
        [
            {"start": 10.0, "end": 12.0, "text": "second"},
            {"start": 1.0, "end": 3.0, "text": "first"},
            {"start": 5.0, "end": 6.0, "text": "   "},  # dropped
        ],
        language="en",
    )
    assert [s.text for s in transcript.segments] == ["first", "second"]
    assert [s.id for s in transcript.segments] == [0, 1]
    assert transcript.duration == 12.0


def test_build_transcript_fixes_inverted_timestamps():
    transcript = build_transcript([{"start": 5.0, "end": 2.0, "text": "oops"}])
    assert transcript.segments[0].end == 5.0


def test_round_trip_through_json(tmp_path, transcript):
    path = transcript.save(tmp_path / "transcript.json")
    reloaded = Transcript.load(path)
    assert len(reloaded) == len(transcript)
    assert reloaded.language == "en"
    assert reloaded.word_count == transcript.word_count
    assert reloaded.segments[3].words[0].text == transcript.segments[3].words[0].text


def test_text_and_segments_in_range(transcript):
    segment = transcript.segments[4]
    text = transcript.text_in_range(segment.start, segment.end)
    assert "ninety percent" in text
    assert len(transcript.segments_in_range(segment.start, segment.end)) == 1


def test_silence_ratio_is_bounded(transcript):
    ratio = transcript.silence_ratio(0.0, transcript.duration)
    assert 0.0 <= ratio < 0.3  # the fixture speaks almost continuously
    assert transcript.silence_ratio(transcript.duration + 10, transcript.duration + 40) == 1.0


def test_sentence_boundary_detection(transcript):
    assert transcript.segments[0].ends_sentence() is True
    assert transcript.starts_sentence(transcript.segments[0]) is True
    assert transcript.starts_sentence(transcript.segments[5]) is True


def test_words_in_range_and_shift(transcript):
    words = transcript.words_in_range(10.0, 14.0)
    assert words
    shifted = shift_words(words, 10.0)
    assert shifted[0].start >= 0.0
    assert shifted[0].start < words[0].start


def test_gap_helpers(transcript):
    first, second = transcript.segments[0], transcript.segments[1]
    assert transcript.gap_before(first) == float("inf")
    assert transcript.gap_after(transcript.segments[-1]) == float("inf")
    assert 0.3 < transcript.gap_before(second) < 0.6
