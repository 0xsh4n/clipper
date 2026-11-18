"""URL validation, filename sanitisation and time formatting."""

from __future__ import annotations

import pytest

from app.errors import ClipperError
from app.utils import (
    ass_timestamp,
    extract_video_id,
    format_timestamp,
    parse_bool,
    sanitize_filename,
    srt_timestamp,
    validate_youtube_url,
    vtt_timestamp,
)


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/live/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://m.youtube.com/watch?v=dQw4w9WgXcQ&t=42s", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ],
)
def test_extract_video_id_supported_forms(url, expected):
    assert extract_video_id(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        "https://vimeo.com/123456",
        "https://www.youtube.com/watch?v=tooshort",
        "https://www.youtube.com/",
        "file:///etc/passwd",
        "",
        "not a url",
    ],
)
def test_validate_youtube_url_rejects_bad_input(url):
    with pytest.raises(ClipperError):
        validate_youtube_url(url)


def test_validate_youtube_url_adds_scheme():
    assert validate_youtube_url("youtu.be/dQw4w9WgXcQ").startswith("https://")


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("../../etc/passwd", "etc_passwd"),
        ("/absolute/path", "absolute_path"),
        ("C:\\Windows\\System32", "C_Windows_System32"),
        ("my: video | title?", "my_video_title"),
        ("...", "clip"),
        ("", "clip"),
        ("CON", "CON_file"),
        ("a" * 200, "a" * 80),
    ],
)
def test_sanitize_filename(raw, expected):
    assert sanitize_filename(raw) == expected


def test_sanitize_filename_never_escapes_a_directory():
    for raw in ["../../../x", "..\\..\\x", "a/b/c", "a\x00b"]:
        result = sanitize_filename(raw)
        assert "/" not in result and "\\" not in result and ".." not in result


def test_timestamp_formats():
    assert format_timestamp(3738.4) == "01:02:18"
    assert srt_timestamp(1.5) == "00:00:01,500"
    assert vtt_timestamp(1.5) == "00:00:01.500"
    assert ass_timestamp(1.5) == "0:00:01.50"
    assert srt_timestamp(-5) == "00:00:00,000"


def test_parse_bool():
    assert parse_bool("yes") is True
    assert parse_bool("FALSE") is False
    assert parse_bool(None, default=True) is True
    with pytest.raises(ClipperError):
        parse_bool("maybe")
