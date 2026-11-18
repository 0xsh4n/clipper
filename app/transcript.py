"""Transcript data model and text utilities.

The transcript is the single source of truth for every later stage: candidate
windows, LLM prompts, clip boundary refinement and subtitles all read from it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .utils import read_json, write_json

SENTENCE_END_RE = re.compile(r"[.!?…。！？](?:[\"'”’\)\]]+)?$")
_WORD_RE = re.compile(r"\w+", re.UNICODE)


@dataclass
class Word:
    start: float
    end: float
    text: str
    probability: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": round(float(self.start), 3),
            "end": round(float(self.end), 3),
            "text": self.text,
            "probability": round(float(self.probability), 4),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Word":
        return cls(
            start=float(data.get("start", 0.0)),
            end=float(data.get("end", data.get("start", 0.0))),
            text=str(data.get("text") or data.get("word") or ""),
            probability=float(data.get("probability", 1.0) or 1.0),
        )


@dataclass
class Segment:
    """One transcript segment with its own timestamps."""

    id: int
    start: float
    end: float
    text: str
    words: List[Word] = field(default_factory=list)
    no_speech_prob: float = 0.0
    avg_logprob: float = 0.0

    @property
    def duration(self) -> float:
        return max(0.0, float(self.end) - float(self.start))

    @property
    def word_count(self) -> int:
        return len(_WORD_RE.findall(self.text))

    def ends_sentence(self) -> bool:
        return bool(SENTENCE_END_RE.search(self.text.strip()))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": int(self.id),
            "start": round(float(self.start), 3),
            "end": round(float(self.end), 3),
            "text": self.text,
            "no_speech_prob": round(float(self.no_speech_prob), 4),
            "avg_logprob": round(float(self.avg_logprob), 4),
            "words": [w.to_dict() for w in self.words],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Segment":
        return cls(
            id=int(data.get("id", 0)),
            start=float(data.get("start", 0.0)),
            end=float(data.get("end", 0.0)),
            text=str(data.get("text") or "").strip(),
            words=[Word.from_dict(w) for w in (data.get("words") or [])],
            no_speech_prob=float(data.get("no_speech_prob", 0.0) or 0.0),
            avg_logprob=float(data.get("avg_logprob", 0.0) or 0.0),
        )


@dataclass
class Transcript:
    """Timestamped transcript of the whole source video."""

    segments: List[Segment] = field(default_factory=list)
    language: str = ""
    language_probability: float = 0.0
    duration: float = 0.0
    model: str = ""
    source_path: str = ""

    # -- basics ----------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self.segments)

    @property
    def text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments if s.text.strip())

    @property
    def word_count(self) -> int:
        return sum(s.word_count for s in self.segments)

    @property
    def speech_duration(self) -> float:
        return sum(s.duration for s in self.segments)

    def words(self) -> List[Word]:
        out: List[Word] = []
        for segment in self.segments:
            out.extend(segment.words)
        return out

    # -- queries ---------------------------------------------------------- #
    def segments_in_range(self, start: float, end: float, *, overlap_ratio: float = 0.5) -> List[Segment]:
        """Segments that overlap ``[start, end]`` by at least ``overlap_ratio``."""

        out: List[Segment] = []
        for segment in self.segments:
            overlap = min(segment.end, end) - max(segment.start, start)
            if overlap <= 0:
                continue
            if segment.duration <= 0 or overlap / segment.duration >= overlap_ratio:
                out.append(segment)
        return out

    def text_in_range(self, start: float, end: float) -> str:
        return " ".join(
            s.text.strip() for s in self.segments_in_range(start, end) if s.text.strip()
        ).strip()

    def words_in_range(self, start: float, end: float) -> List[Word]:
        return [w for w in self.words() if w.end > start and w.start < end]

    def silence_ratio(self, start: float, end: float) -> float:
        """Fraction of ``[start, end]`` with no speech (0.0 - 1.0)."""

        span = max(1e-6, float(end) - float(start))
        spoken = 0.0
        for segment in self.segments:
            overlap = min(segment.end, end) - max(segment.start, start)
            if overlap > 0:
                spoken += overlap
        return max(0.0, min(1.0, 1.0 - spoken / span))

    def gap_before(self, segment: Segment) -> float:
        """Silence immediately before ``segment`` (``inf`` for the first one)."""

        index = self._index_of(segment)
        if index <= 0:
            return float("inf")
        return max(0.0, float(segment.start) - float(self.segments[index - 1].end))

    def gap_after(self, segment: Segment) -> float:
        index = self._index_of(segment)
        if index < 0 or index >= len(self.segments) - 1:
            return float("inf")
        return max(0.0, float(self.segments[index + 1].start) - float(segment.end))

    def starts_sentence(self, segment: Segment) -> bool:
        """True when ``segment`` plausibly begins a new sentence."""

        index = self._index_of(segment)
        if index <= 0:
            return True
        previous = self.segments[index - 1]
        if previous.ends_sentence():
            return True
        if float(segment.start) - float(previous.end) >= 0.6:
            return True
        stripped = segment.text.strip()
        return bool(stripped[:1].isupper() and previous.text.strip().endswith(","))

    def _index_of(self, segment: Segment) -> int:
        for index, candidate in enumerate(self.segments):
            if candidate.id == segment.id and candidate.start == segment.start:
                return index
        return -1

    # -- persistence ------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        return {
            "language": self.language,
            "language_probability": round(float(self.language_probability), 4),
            "duration": round(float(self.duration), 3),
            "model": self.model,
            "source_path": self.source_path,
            "word_count": self.word_count,
            "segment_count": len(self.segments),
            "segments": [s.to_dict() for s in self.segments],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Transcript":
        segments = [Segment.from_dict(s) for s in (data.get("segments") or [])]
        duration = float(data.get("duration") or 0.0)
        if duration <= 0 and segments:
            duration = max(s.end for s in segments)
        return cls(
            segments=segments,
            language=str(data.get("language") or ""),
            language_probability=float(data.get("language_probability") or 0.0),
            duration=duration,
            model=str(data.get("model") or ""),
            source_path=str(data.get("source_path") or ""),
        )

    def save(self, path: Path) -> Path:
        return write_json(Path(path), self.to_dict())

    @classmethod
    def load(cls, path: Path) -> "Transcript":
        return cls.from_dict(read_json(Path(path)))


# --------------------------------------------------------------------------- #
# free functions
# --------------------------------------------------------------------------- #
def build_transcript(
    raw_segments: Iterable[Dict[str, Any]],
    *,
    language: str = "",
    language_probability: float = 0.0,
    duration: float = 0.0,
    model: str = "",
    source_path: str = "",
) -> Transcript:
    """Normalise raw segment dicts (any source) into a :class:`Transcript`."""

    segments: List[Segment] = []
    for index, raw in enumerate(raw_segments):
        text = str(raw.get("text") or "").strip()
        start = float(raw.get("start") or 0.0)
        end = float(raw.get("end") or start)
        if end < start:
            end = start
        if not text:
            continue
        words = [Word.from_dict(w) for w in (raw.get("words") or [])]
        words = [w for w in words if w.text.strip()]
        segments.append(
            Segment(
                id=index,
                start=start,
                end=end,
                text=text,
                words=words,
                no_speech_prob=float(raw.get("no_speech_prob") or 0.0),
                avg_logprob=float(raw.get("avg_logprob") or 0.0),
            )
        )
    segments.sort(key=lambda s: (s.start, s.end))
    for index, segment in enumerate(segments):
        segment.id = index
    if duration <= 0 and segments:
        duration = max(s.end for s in segments)
    return Transcript(
        segments=segments,
        language=language,
        language_probability=language_probability,
        duration=duration,
        model=model,
        source_path=source_path,
    )


def shift_words(words: Sequence[Word], offset: float) -> List[Word]:
    """Rebase word timings so a clip starts at 0.0."""

    out: List[Word] = []
    for word in words:
        out.append(
            Word(
                start=max(0.0, float(word.start) - offset),
                end=max(0.0, float(word.end) - offset),
                text=word.text,
                probability=word.probability,
            )
        )
    return out
