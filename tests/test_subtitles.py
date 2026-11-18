"""Caption construction, wrapping and file writing."""

from __future__ import annotations

import re

import pytest

from app.config import load_config
from app.subtitle_generator import (
    SubtitleGenerator,
    escape_ass_text,
    hex_to_ass_color,
    wrap_lines,
)


def test_wrap_lines_respects_limits():
    lines = wrap_lines("the quick brown fox jumps over the lazy dog", 20, 2)
    assert 1 <= len(lines) <= 2
    assert all(line.strip() for line in lines)


def test_wrap_lines_never_exceeds_max_lines():
    text = "word " * 60
    assert len(wrap_lines(text, 16, 2)) == 2
    assert len(wrap_lines(text, 16, 1)) == 1


def test_wrap_lines_handles_empty_input():
    assert wrap_lines("   ", 20, 2) == []


def test_hex_to_ass_color_is_bgr():
    assert hex_to_ass_color("#FFCC00") == "&H0000CCFF"
    assert hex_to_ass_color("000000") == "&H00000000"
    assert hex_to_ass_color("nonsense") == "&H00FFFFFF"
    assert hex_to_ass_color("#FFFFFF", alpha=128) == "&H80FFFFFF"


def test_escape_ass_text():
    assert escape_ass_text("a\nb") == "a\\Nb"
    assert escape_ass_text("{bold}") == "\\{bold\\}"


def test_build_cues_from_word_timestamps(transcript, config):
    generator = SubtitleGenerator(config)
    cues = generator.build_cues(transcript, 10.0, 40.0)

    assert cues
    assert cues[0].start >= 0.0                       # rebased to the clip
    assert all(cue.end > cue.start for cue in cues)
    assert all(len(cue.lines) <= config.subtitles.max_lines for cue in cues)
    assert all(cue.end <= 30.01 for cue in cues)      # never past the clip length
    for earlier, later in zip(cues, cues[1:]):
        assert earlier.end <= later.start + 0.001     # no overlapping cues


def test_cues_fall_back_to_segments_without_word_timings(config):
    from app.transcript import build_transcript

    transcript = build_transcript(
        [{"start": 0.0, "end": 8.0, "text": "A sentence with no word level timings at all."}],
        duration=8.0,
    )
    cues = SubtitleGenerator(config).build_cues(transcript, 0.0, 8.0)
    assert cues
    assert cues[0].text


def test_uppercase_styling(transcript):
    config = load_config(overrides={"subtitles.uppercase": True}, environ={})
    cues = SubtitleGenerator(config).build_cues(transcript, 10.0, 40.0)
    assert cues[0].text == cues[0].text.upper()


def test_generate_writes_all_files(tmp_path, transcript, config):
    generator = SubtitleGenerator(config)
    files = generator.generate(transcript, 10.0, 40.0, tmp_path, formats=("16:9", "9:16"))

    assert set(files) == {"srt", "vtt", "ass_landscape", "ass_vertical"}
    srt = files["srt"].read_text(encoding="utf-8")
    assert "-->" in srt and srt.strip().startswith("1")
    assert files["vtt"].read_text(encoding="utf-8").startswith("WEBVTT")

    vertical = files["ass_vertical"].read_text(encoding="utf-8")
    assert "PlayResX: 1080" in vertical and "PlayResY: 1920" in vertical
    assert "Dialogue:" in vertical
    landscape = files["ass_landscape"].read_text(encoding="utf-8")
    assert "PlayResX: 1920" in landscape


def test_ass_position_changes_alignment(tmp_path, transcript):
    for position, alignment in (("bottom", ",2,"), ("center", ",5,"), ("top", ",8,")):
        config = load_config(overrides={"subtitles.position": position}, environ={})
        generator = SubtitleGenerator(config)
        cues = generator.build_cues(transcript, 10.0, 40.0)
        path = generator.write_ass(cues, tmp_path / f"{position}.ass", resolution=(1080, 1920))
        style_line = [
            line for line in path.read_text(encoding="utf-8").splitlines()
            if line.startswith("Style: Default,")
        ][0]
        assert alignment in style_line


def style_field(path, index: int) -> str:
    line = [
        l for l in path.read_text(encoding="utf-8").splitlines()
        if l.startswith("Style: Default,")
    ][0]
    return line.split("Style: ", 1)[1].split(",")[index]


def test_font_size_scales_with_output_height(tmp_path, transcript, config):
    generator = SubtitleGenerator(config)
    cues = generator.build_cues(transcript, 10.0, 40.0)
    half = generator.write_ass(cues, tmp_path / "half.ass", resolution=(960, 540))
    landscape = generator.write_ass(cues, tmp_path / "landscape.ass", resolution=(1920, 1080))
    vertical = generator.write_ass(cues, tmp_path / "vertical.ass", resolution=(1080, 1920))

    # Authored for a 1080px tall frame, so 1080p landscape is the baseline...
    assert int(style_field(landscape, 2)) == config.subtitles.font_size
    # ...a half-height frame gets half the size...
    assert int(style_field(half, 2)) == config.subtitles.font_size // 2
    # ...and the vertical render scales up to stay equally readable.
    assert int(style_field(vertical, 2)) > config.subtitles.font_size


def test_caption_size_and_margin_are_a_stable_share_of_height(tmp_path, transcript, config):
    """Captions must occupy the same relative space in both orientations."""

    generator = SubtitleGenerator(config)
    cues = generator.build_cues(transcript, 10.0, 40.0)
    for (width, height) in ((1920, 1080), (1080, 1920)):
        path = generator.write_ass(cues, tmp_path / f"{height}.ass", resolution=(width, height))
        size_ratio = int(style_field(path, 2)) / height
        margin_ratio = int(style_field(path, 21)) / height
        assert 0.035 < size_ratio < 0.06
        assert margin_ratio == pytest.approx(config.subtitles.margin_ratio, abs=0.01)


def test_captions_cannot_overflow_the_frame(tmp_path, transcript):
    """WrapStyle 2 disables libass wrapping, so we must guarantee the fit."""

    config = load_config(overrides={"subtitles.max_chars_per_line": 40}, environ={})
    generator = SubtitleGenerator(config)
    cues = generator.build_cues(transcript, 10.0, 40.0)
    path = generator.write_ass(cues, tmp_path / "wide.ass", resolution=(1080, 1920))

    font_size = int(style_field(path, 2))
    margin = int(style_field(path, 19))
    usable = 1080 - 2 * margin
    dialogue = [l for l in path.read_text(encoding="utf-8").splitlines() if l.startswith("Dialogue:")]
    assert dialogue
    for line in dialogue:
        # Dialogue has 9 comma-separated fields before the text itself.
        text = line.split("Dialogue: ", 1)[1].split(",", 9)[9]
        size = font_size
        override = re.match(r"\{\\fs(\d+)\}", text)
        if override:                       # the cue carries its own smaller size
            size = int(override.group(1))
            text = text[override.end():]
            assert size >= font_size * 0.4
        for text_line in text.split("\\N"):
            assert len(text_line) * 0.55 * size <= usable * 1.02
