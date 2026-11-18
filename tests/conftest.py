"""Shared fixtures: a synthetic transcript so tests need no network, GPU or FFmpeg."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import AppConfig, load_config  # noqa: E402
from app.transcript import Transcript, build_transcript  # noqa: E402

SENTENCES = [
    "Welcome back to the channel, it is great to have you here again.",
    "Before we start, this video is brought to you by our sponsor, use code SAVE20.",
    "Okay so let me show you the setup we are going to use today.",
    "Most people think that training a model needs a huge cluster of GPUs.",
    "It turns out you can get ninety percent of the quality on a single laptop.",
    "The reason is simple: the bottleneck is data quality, not compute.",
    "We measured this across forty experiments and the result held every time.",
    "That completely changes who can build these systems.",
    "Um, so, yeah, anyway, where was I, let me check my notes here.",
    "The second thing nobody tells you is that evaluation is harder than training.",
    "You can ship a model that looks perfect on benchmarks and fails on day one.",
    "We lost three weeks to exactly that mistake and it cost us a customer.",
    "So the lesson is to build the evaluation set before you build the model.",
    "Alright, that is it for today, do not forget to like and subscribe.",
]


@pytest.fixture()
def config() -> AppConfig:
    return load_config()


@pytest.fixture()
def transcript() -> Transcript:
    """A 14 sentence, ~70 second transcript with word timestamps."""

    raw: List[dict] = []
    clock = 0.0
    for sentence in SENTENCES:
        words = sentence.split()
        per_word = 0.34
        word_entries = []
        cursor = clock
        for word in words:
            word_entries.append(
                {"start": round(cursor, 3), "end": round(cursor + per_word * 0.9, 3), "word": word}
            )
            cursor += per_word
        raw.append(
            {
                "start": round(clock, 3),
                "end": round(cursor, 3),
                "text": sentence,
                "words": word_entries,
            }
        )
        clock = cursor + 0.45  # natural pause between sentences
    return build_transcript(raw, language="en", duration=clock, model="small")
