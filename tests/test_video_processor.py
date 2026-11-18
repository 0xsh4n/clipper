"""Reframing filter graphs and FFmpeg argument safety (no FFmpeg required)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import load_config
from app.errors import DependencyError, RenderError
from app.media import FFmpeg, MediaInfo
from app.video_processor import VideoProcessor, _filter_escape


def info(width: int, height: int) -> MediaInfo:
    return MediaInfo(
        path=Path("clip.mp4"), duration=30.0, width=width, height=height, fps=30.0, has_audio=True
    )


def test_vertical_crop_is_a_crop_not_a_squash(config):
    filters, center, strategy = VideoProcessor(config)._build_filters(
        info(1920, 1080), 1080, 1920, "9:16", 0.5, "center"
    )
    assert filters[0].startswith("crop=")
    crop_w, crop_h, crop_x, _ = filters[0].split("=", 1)[1].split(":")
    # 1080 * 9/16 = 607.5 -> 608 (even), centred.
    assert int(crop_w) == 608 and int(crop_h) == 1080
    assert int(crop_x) == (1920 - 608) // 2
    assert "scale=1080:1920" in filters[1]
    assert center == 0.5


def test_vertical_crop_follows_the_subject(config):
    processor = VideoProcessor(config)
    left = processor._build_filters(info(1920, 1080), 1080, 1920, "9:16", 0.2, "face")[0][0]
    right = processor._build_filters(info(1920, 1080), 1080, 1920, "9:16", 0.8, "face")[0][0]
    left_x = int(left.split(":")[2])
    right_x = int(right.split(":")[2])
    assert left_x < right_x
    assert left_x >= 0
    assert right_x + 608 <= 1920          # never crops past the frame
    assert left_x % 2 == 0 and right_x % 2 == 0


def test_vertical_source_is_padded_not_cropped(config):
    filters, _, strategy = VideoProcessor(config)._build_filters(
        info(1080, 1920), 1080, 1920, "9:16", 0.5, ""
    )
    assert any(f.startswith("pad=") for f in filters)
    assert not any(f.startswith("crop=") for f in filters)
    assert "already vertical" in strategy


def test_landscape_matching_aspect_is_scaled(config):
    filters, _, strategy = VideoProcessor(config)._build_filters(
        info(1920, 1080), 1920, 1080, "16:9", 0.5, ""
    )
    assert filters[0].startswith("scale=1920:1080")
    assert strategy == "scaled"


def test_landscape_from_vertical_source_uses_blurred_background(config):
    filters, _, strategy = VideoProcessor(config)._build_filters(
        info(1080, 1920), 1920, 1080, "16:9", 0.5, ""
    )
    assert "boxblur" in filters[0] and "overlay=" in filters[0]
    assert strategy == "blurred pillarbox"


def test_smart_crop_disabled_returns_centre(tmp_path):
    config = load_config(overrides={"video.smart_crop": False}, environ={})
    center, strategy = VideoProcessor(config).detect_subject_center(Path("missing.mp4"), tmp_path)
    assert center == 0.5
    assert "disabled" in strategy


def test_cut_clip_rejects_a_non_positive_duration(config, tmp_path):
    with pytest.raises(RenderError):
        VideoProcessor(config).cut_clip(Path("in.mp4"), 30.0, 30.0, tmp_path / "out.mp4")


def test_filter_escape_protects_separators():
    escaped = _filter_escape("C:\\subs\\my [clip], v2.ass")
    for char in (":", "[", "]", ","):
        assert f"\\{char}" in escaped


def test_probe_rejects_a_missing_file():
    with pytest.raises(RenderError):
        FFmpeg().probe(Path("definitely-not-here.mp4"))


def test_verify_reports_a_missing_binary():
    with pytest.raises(DependencyError) as excinfo:
        FFmpeg(ffmpeg_path="ffmpeg-does-not-exist", ffprobe_path="ffprobe-does-not-exist").verify()
    assert "ffmpeg" in str(excinfo.value).lower()
