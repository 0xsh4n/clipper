"""Malformed LLM output: extraction, repair and refusal."""

from __future__ import annotations

import pytest

from app.errors import LLMError
from app.llm_analyzer import coerce_number, extract_json, repair_json


def test_plain_json():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_markdown_fenced_json():
    raw = 'Here you go:\n```json\n{"scores": [{"id": 0, "hook": 8}]}\n```\nHope that helps!'
    assert extract_json(raw) == {"scores": [{"id": 0, "hook": 8}]}


def test_trailing_commas_are_repaired():
    assert extract_json('{"a": [1, 2,], "b": 3,}') == {"a": [1, 2], "b": 3}


def test_truncated_response_is_closed():
    assert extract_json('{"selected": true, "start": 12.5, "title": "x"') == {
        "selected": True,
        "start": 12.5,
        "title": "x",
    }


def test_prose_around_json_is_stripped():
    raw = 'Sure thing. {"id": 3, "score": 9.1} Let me know if you want another.'
    assert extract_json(raw) == {"id": 3, "score": 9.1}


def test_python_style_dict_is_repaired():
    assert extract_json("{'scores': [{'id': 0, 'hook': 9}]}") == {"scores": [{"id": 0, "hook": 9}]}


def test_smart_quotes_and_comments_are_repaired():
    raw = '{“title”: “the truth”, // chosen\n "score": 9}'
    assert extract_json(raw) == {"title": "the truth", "score": 9}


def test_python_literals_are_repaired():
    assert extract_json('{"selected": True, "note": None, "ok": False}') == {
        "selected": True,
        "note": None,
        "ok": False,
    }


def test_bare_array():
    assert extract_json('[{"id": 1}, {"id": 2}]') == [{"id": 1}, {"id": 2}]


@pytest.mark.parametrize("raw", ["", "   ", "I cannot help with that.", None])
def test_unrecoverable_output_raises(raw):
    with pytest.raises(LLMError):
        extract_json(raw)


def test_repair_json_handles_unterminated_string():
    assert repair_json('{"title": "unfinished') is not None


@pytest.mark.parametrize(
    "raw,expected",
    [
        (9.1, 9.1),
        ("9.1", 9.1),
        ("9,1", 9.1),
        ("13:32", 812.0),
        ("1:02:18", 3738.0),
        ("score: 8.5/10", 8.5),
        (True, None),
        ("", None),
        (None, None),
        ("not a number", None),
    ],
)
def test_coerce_number(raw, expected):
    assert coerce_number(raw, None) == expected
