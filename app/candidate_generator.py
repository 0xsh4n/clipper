"""Stage 1: candidate window generation.

The LLM never sees the whole transcript at once. Instead we build overlapping
windows that are already aligned to transcript segment boundaries, compute
cheap objective metadata for each (word count, speech density, silence ratio,
sentence alignment), drop the obvious junk (intros, sponsor reads, dead air)
and hand a diverse shortlist to Qwen for semantic scoring.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config import AppConfig, ScoringWeights
from .logger import get_logger
from .transcript import Transcript, Segment
from .utils import clamp, format_clock

log = get_logger("candidates")

SCORE_KEYS = (
    "hook",
    "interestingness",
    "information_value",
    "emotional_impact",
    "shareability",
    "clarity",
    "standalone_context",
)

_WORD_RE = re.compile(r"\w+", re.UNICODE)

# Phrases that almost always make a bad short.
AD_PATTERNS = (
    "sponsor", "sponsored by", "this video is brought to you",
    "promo code", "discount code", "coupon", "use code",
    "link in the description", "link in the bio", "check the description",
    "sign up at", "free trial", "limited time offer", "patreon",
    "merch", "affiliate",
)
CTA_PATTERNS = (
    "like and subscribe", "subscribe to the channel", "hit the bell",
    "smash that like", "ring the bell", "don't forget to subscribe",
    "leave a comment below", "follow me on",
)
GREETING_PATTERNS = (
    "hey guys", "hey everyone", "hi everyone", "hello everyone",
    "what's up guys", "whats up guys", "welcome back", "welcome to the channel",
    "welcome to another", "in today's video", "in this video we're going to",
    "before we get started", "my name is", "thanks for watching",
    "see you in the next", "that's it for today",
)
# Phrases that tend to mark a strong, self-contained moment.
HOOK_PATTERNS = (
    "the problem is", "here's the thing", "heres the thing", "what most people",
    "most people think", "nobody tells you", "the truth is", "the reality is",
    "i was wrong", "turns out", "it turns out", "surprisingly", "shockingly",
    "the mistake", "the biggest", "the secret", "never", "always",
    "the reason", "because of this", "which means", "for example",
    "let me explain", "think about it", "imagine", "actually",
    "and that's why", "the craziest", "insane", "unbelievable",
    "i can't believe", "we just", "breaking", "announcement", "announcing",
)
FILLER_WORDS = ("um", "uh", "erm", "mm", "hmm", "like", "you know", "i mean", "sort of", "kind of")


@dataclass
class Candidate:
    """A potential clip window plus everything we know about it."""

    id: int
    start: float
    end: float
    text: str
    segment_ids: List[int] = field(default_factory=list)
    word_count: int = 0
    speech_density: float = 0.0      # words per second
    silence_ratio: float = 0.0       # 0.0 - 1.0
    starts_sentence: bool = False
    ends_sentence: bool = False
    heuristic_score: float = 0.0
    penalties: List[str] = field(default_factory=list)
    highlights: List[str] = field(default_factory=list)
    # Filled in by the LLM stage.
    scores: Dict[str, float] = field(default_factory=dict)
    overall: float = 0.0
    title: str = ""
    hook: str = ""
    reason: str = ""
    scored_by_llm: bool = False

    @property
    def duration(self) -> float:
        return max(0.0, float(self.end) - float(self.start))

    @property
    def label(self) -> str:
        return f"{format_clock(self.start)}-{format_clock(self.end)}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "start": round(float(self.start), 3),
            "end": round(float(self.end), 3),
            "duration": round(self.duration, 3),
            "label": self.label,
            "word_count": self.word_count,
            "speech_density": round(float(self.speech_density), 3),
            "silence_ratio": round(float(self.silence_ratio), 3),
            "starts_sentence": self.starts_sentence,
            "ends_sentence": self.ends_sentence,
            "heuristic_score": round(float(self.heuristic_score), 3),
            "penalties": list(self.penalties),
            "highlights": list(self.highlights),
            "scores": {k: round(float(v), 2) for k, v in self.scores.items()},
            "overall": round(float(self.overall), 3),
            "title": self.title,
            "hook": self.hook,
            "reason": self.reason,
            "scored_by_llm": self.scored_by_llm,
            "text": self.text,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Candidate":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        payload = {k: v for k, v in data.items() if k in known}
        payload.setdefault("id", 0)
        payload.setdefault("start", 0.0)
        payload.setdefault("end", 0.0)
        payload.setdefault("text", "")
        return cls(**payload)


# --------------------------------------------------------------------------- #
# scoring helpers (pure functions - unit tested)
# --------------------------------------------------------------------------- #
def _count_patterns(text: str, patterns: Sequence[str]) -> List[str]:
    low = text.lower()
    return [p for p in patterns if p in low]


def heuristic_score(candidate: Candidate, target_duration: float = 30.0) -> Tuple[float, List[str], List[str]]:
    """Cheap, deterministic pre-score in the 0-10 range.

    Returns ``(score, penalties, highlights)``. This only decides which windows
    are worth spending LLM tokens on; the semantic judgement is Qwen's.
    """

    penalties: List[str] = []
    highlights: List[str] = []
    score = 5.0

    # Speech density: ~2.2-3.2 words/second is comfortable speech.
    density = candidate.speech_density
    if density < 1.2:
        score -= 2.5
        penalties.append("very low speech density")
    elif density < 1.8:
        score -= 1.0
        penalties.append("low speech density")
    elif 2.0 <= density <= 3.6:
        score += 1.0
    elif density > 4.6:
        score -= 0.5
        penalties.append("rushed speech")

    # Dead air.
    if candidate.silence_ratio > 0.45:
        score -= 2.5
        penalties.append("long pauses / dead air")
    elif candidate.silence_ratio > 0.3:
        score -= 1.0
        penalties.append("noticeable pauses")
    elif candidate.silence_ratio < 0.12:
        score += 0.5

    # Sentence alignment.
    if candidate.starts_sentence:
        score += 0.9
    else:
        score -= 1.2
        penalties.append("starts mid-sentence")
    if candidate.ends_sentence:
        score += 0.9
    else:
        score -= 1.0
        penalties.append("ends abruptly")

    # Ads, calls to action and greetings are disqualifying.
    ads = _count_patterns(candidate.text, AD_PATTERNS)
    if ads:
        score -= 4.0
        penalties.append(f"advertisement/sponsor language ({ads[0]})")
    ctas = _count_patterns(candidate.text, CTA_PATTERNS)
    if ctas:
        score -= 2.5
        penalties.append(f"channel call-to-action ({ctas[0]})")
    greetings = _count_patterns(candidate.text, GREETING_PATTERNS)
    if greetings:
        score -= 2.0
        penalties.append(f"intro/outro boilerplate ({greetings[0]})")

    hooks = _count_patterns(candidate.text, HOOK_PATTERNS)
    if hooks:
        score += min(1.8, 0.6 * len(hooks))
        highlights.extend(hooks[:3])

    # Numbers and named entities hint at concrete information.
    if re.search(r"\b\d{2,}\b|\b\d+(?:\.\d+)?\s?(?:%|percent|million|billion|thousand|dollars|x)\b", candidate.text, re.I):
        score += 0.6
        highlights.append("contains concrete figures")

    # Questions often make strong hooks.
    if "?" in candidate.text[: max(60, len(candidate.text) // 3)]:
        score += 0.5
        highlights.append("opens with a question")

    # Filler-heavy text reads badly as a caption.
    words = _WORD_RE.findall(candidate.text.lower())
    if words:
        filler = sum(1 for w in words if w in FILLER_WORDS)
        if filler / len(words) > 0.12:
            score -= 1.0
            penalties.append("filler-heavy delivery")

    # Too few words cannot carry a 30 second clip.
    expected_words = max(12.0, 1.7 * candidate.duration)
    if candidate.word_count < expected_words * 0.5:
        score -= 2.0
        penalties.append("too little speech for the duration")

    # Prefer durations near the target.
    drift = abs(candidate.duration - float(target_duration))
    score -= min(1.5, drift * 0.08)

    # Repetition inside the window.
    if words and len(set(words)) / len(words) < 0.45:
        score -= 1.0
        penalties.append("repetitive wording")

    return clamp(score, 0.0, 10.0), penalties, highlights


def compute_overall(scores: Dict[str, float], weights: ScoringWeights) -> float:
    """Weighted blend of the seven LLM dimensions (0-10)."""

    weight_map = weights.as_dict()
    total = 0.0
    total_weight = 0.0
    for key, weight in weight_map.items():
        if key not in scores:
            continue
        total += clamp(float(scores[key]), 0.0, 10.0) * float(weight)
        total_weight += float(weight)
    if total_weight <= 0:
        return 0.0
    # Normalising by the weights actually present keeps the result on the 0-10
    # scale even when the LLM omitted a dimension.
    return round(total / total_weight, 4)


def normalise_scores(raw: Dict[str, Any]) -> Dict[str, float]:
    """Keep only the known dimensions, coerced into 0-10 floats."""

    out: Dict[str, float] = {}
    for key in SCORE_KEYS:
        if key not in raw:
            continue
        try:
            out[key] = clamp(float(raw[key]), 0.0, 10.0)
        except (TypeError, ValueError):
            continue
    return out


# --------------------------------------------------------------------------- #
# generator
# --------------------------------------------------------------------------- #
class CandidateGenerator:
    """Builds the shortlist of candidate windows for a transcript."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def generate(self, transcript: Transcript) -> List[Candidate]:
        clip_cfg = self.config.clip
        windows = self._build_windows(transcript)
        if not windows:
            windows = self._build_fallback_windows(transcript)
        log.info("Generated %d raw candidate windows", len(windows))

        for candidate in windows:
            score, penalties, highlights = heuristic_score(candidate, clip_cfg.target_duration)
            candidate.heuristic_score = score
            candidate.penalties = penalties
            candidate.highlights = highlights

        shortlist = self._select_diverse(windows, clip_cfg.max_candidates, clip_cfg.target_duration)
        for index, candidate in enumerate(shortlist):
            candidate.id = index
        log.info(
            "Shortlisted %d candidates spanning %s - %s",
            len(shortlist),
            format_clock(min((c.start for c in shortlist), default=0.0)),
            format_clock(max((c.end for c in shortlist), default=0.0)),
        )
        return shortlist

    # -- window construction ---------------------------------------------- #
    def _build_windows(self, transcript: Transcript) -> List[Candidate]:
        clip_cfg = self.config.clip
        segments = transcript.segments
        if not segments:
            return []

        lower = float(clip_cfg.min_duration)
        upper = float(clip_cfg.max_duration)
        total = transcript.duration or (segments[-1].end if segments else 0.0)
        earliest = float(clip_cfg.skip_intro_seconds)
        latest = total - float(clip_cfg.skip_outro_seconds) if clip_cfg.skip_outro_seconds else total

        windows: List[Candidate] = []
        seen: set[Tuple[int, int]] = set()

        for i, first in enumerate(segments):
            if first.start < earliest:
                continue
            if latest and first.start > latest:
                break
            accumulated: List[Segment] = []
            for j in range(i, len(segments)):
                accumulated.append(segments[j])
                end = segments[j].end
                duration = end - first.start
                if latest and end > latest + 1.0:
                    break
                if duration < lower:
                    continue
                if duration > upper:
                    # One more window ending exactly on this segment is still
                    # useful if nothing shorter qualified.
                    if not any(key[0] == i for key in seen):
                        windows.append(self._make_candidate(transcript, accumulated, first.start, end))
                        seen.add((i, j))
                    break
                key = (i, j)
                if key in seen:
                    continue
                seen.add(key)
                windows.append(self._make_candidate(transcript, accumulated, first.start, end))

        # A single very long segment (common on live streams) is split by words.
        for segment in segments:
            if segment.duration <= upper or not segment.words:
                continue
            windows.extend(self._split_long_segment(transcript, segment))

        return self._dedupe(windows)

    def _split_long_segment(self, transcript: Transcript, segment: Segment) -> List[Candidate]:
        target = float(self.config.clip.target_duration)
        step = max(5.0, target / 2.0)
        out: List[Candidate] = []
        cursor = segment.start
        while cursor + float(self.config.clip.min_duration) <= segment.end:
            end = min(segment.end, cursor + target)
            words = [w for w in segment.words if w.end > cursor and w.start < end]
            if words:
                text = " ".join(w.text.strip() for w in words).strip()
                out.append(
                    self._make_candidate_from_text(
                        transcript, text, cursor, end, [segment.id],
                        starts_sentence=True, ends_sentence=text.endswith((".", "!", "?")),
                    )
                )
            cursor += step
        return out

    def _build_fallback_windows(self, transcript: Transcript) -> List[Candidate]:
        """Last resort: fixed stride windows (only when segment walking fails)."""

        target = float(self.config.clip.target_duration)
        total = transcript.duration
        out: List[Candidate] = []
        cursor = 0.0
        while cursor + float(self.config.clip.min_duration) <= total:
            end = min(total, cursor + target)
            text = transcript.text_in_range(cursor, end)
            if text:
                out.append(
                    self._make_candidate_from_text(
                        transcript, text, cursor, end, [], starts_sentence=False, ends_sentence=False
                    )
                )
            cursor += max(5.0, target / 2.0)
        return out

    def _make_candidate(
        self,
        transcript: Transcript,
        segments: Sequence[Segment],
        start: float,
        end: float,
    ) -> Candidate:
        text = " ".join(s.text.strip() for s in segments if s.text.strip()).strip()
        return self._make_candidate_from_text(
            transcript,
            text,
            start,
            end,
            [s.id for s in segments],
            starts_sentence=transcript.starts_sentence(segments[0]),
            ends_sentence=segments[-1].ends_sentence() or transcript.gap_after(segments[-1]) >= 0.8,
        )

    def _make_candidate_from_text(
        self,
        transcript: Transcript,
        text: str,
        start: float,
        end: float,
        segment_ids: Sequence[int],
        *,
        starts_sentence: bool,
        ends_sentence: bool,
    ) -> Candidate:
        duration = max(1e-6, float(end) - float(start))
        word_count = len(_WORD_RE.findall(text))
        return Candidate(
            id=0,
            start=float(start),
            end=float(end),
            text=text,
            segment_ids=list(segment_ids),
            word_count=word_count,
            speech_density=word_count / duration,
            silence_ratio=transcript.silence_ratio(start, end),
            starts_sentence=bool(starts_sentence),
            ends_sentence=bool(ends_sentence),
        )

    # -- shortlisting ----------------------------------------------------- #
    @staticmethod
    def _dedupe(windows: Sequence[Candidate]) -> List[Candidate]:
        out: List[Candidate] = []
        seen: set[Tuple[int, int]] = set()
        for candidate in windows:
            key = (int(round(candidate.start * 2)), int(round(candidate.end * 2)))
            if key in seen:
                continue
            seen.add(key)
            out.append(candidate)
        out.sort(key=lambda c: c.start)
        return out

    @staticmethod
    def _select_diverse(
        windows: Sequence[Candidate],
        limit: int,
        target_duration: float,
    ) -> List[Candidate]:
        """Pick the best non-overlapping windows, spread across the video."""

        limit = max(1, int(limit))
        ranked = sorted(windows, key=lambda c: (-c.heuristic_score, abs(c.duration - target_duration)))
        min_separation = max(4.0, target_duration * 0.6)
        chosen: List[Candidate] = []

        for candidate in ranked:
            if len(chosen) >= limit:
                break
            if any(abs(candidate.start - other.start) < min_separation for other in chosen):
                continue
            chosen.append(candidate)

        # If strict separation starved the list, relax it rather than under-fill.
        if len(chosen) < limit:
            for candidate in ranked:
                if len(chosen) >= limit:
                    break
                if candidate in chosen:
                    continue
                if any(abs(candidate.start - other.start) < 2.0 for other in chosen):
                    continue
                chosen.append(candidate)

        chosen.sort(key=lambda c: c.start)
        return chosen
