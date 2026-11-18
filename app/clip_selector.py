"""Clip boundary refinement.

Qwen chooses *which* moment to clip; this module decides the exact in/out
points. It snaps to sentence boundaries from the transcript so a clip never
starts mid-word or ends on a cliff, while keeping the duration inside the
configured window (25-35 s by default, 30 s target).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .candidate_generator import Candidate
from .config import AppConfig, ClipConfig
from .errors import SelectionError
from .logger import get_logger
from .transcript import Transcript
from .utils import clamp, format_clock

log = get_logger("selector")

#: How far (seconds) we may move a boundary while looking for a clean cut.
SEARCH_WINDOW = 6.0


@dataclass
class Clip:
    """A validated, renderable clip."""

    index: int
    start: float
    end: float
    title: str = ""
    hook: str = ""
    reason: str = ""
    score: float = 0.0
    candidate_id: int = -1
    scores: Dict[str, float] = field(default_factory=dict)
    text: str = ""
    boundary_notes: List[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.0, float(self.end) - float(self.start))

    @property
    def label(self) -> str:
        return f"{format_clock(self.start)} - {format_clock(self.end)}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected": True,
            "index": self.index,
            "start": round(float(self.start), 2),
            "end": round(float(self.end), 2),
            "duration": round(self.duration, 2),
            "timecode": self.label,
            "title": self.title,
            "hook": self.hook,
            "reason": self.reason,
            "score": round(float(self.score), 2),
            "scores": {k: round(float(v), 2) for k, v in self.scores.items()},
            "candidate_id": self.candidate_id,
            "boundary_notes": list(self.boundary_notes),
            "transcript": self.text,
        }


# --------------------------------------------------------------------------- #
# pure helpers (unit tested)
# --------------------------------------------------------------------------- #
def validate_timestamps(start: Any, end: Any, video_duration: float) -> Tuple[float, float]:
    """Coerce and sanity check a timestamp pair against the source duration."""

    try:
        start_f = float(start)
        end_f = float(end)
    except (TypeError, ValueError) as exc:
        raise SelectionError(f"Non-numeric clip timestamps: {start!r} -> {end!r}") from exc

    if start_f != start_f or end_f != end_f:  # NaN
        raise SelectionError("Clip timestamps are not finite numbers.")
    if video_duration <= 0:
        raise SelectionError("Source duration is unknown, cannot validate timestamps.")

    start_f = clamp(start_f, 0.0, max(0.0, video_duration - 0.5))
    end_f = clamp(end_f, 0.0, video_duration)
    if end_f <= start_f:
        raise SelectionError(
            f"Clip end ({end_f:.2f}s) is not after its start ({start_f:.2f}s)."
        )
    return start_f, end_f


def fit_duration(
    start: float,
    end: float,
    clip_cfg: ClipConfig,
    video_duration: float,
) -> Tuple[float, float]:
    """Force ``[start, end]`` into the configured duration window.

    Over-long clips are trimmed from the end (the hook lives at the start);
    short clips are extended forwards first, then backwards.
    """

    start = max(0.0, float(start))
    end = min(float(video_duration), float(end))
    duration = end - start
    minimum = float(clip_cfg.min_duration)
    maximum = float(clip_cfg.max_duration)

    if duration > maximum:
        end = start + maximum
    elif duration < minimum:
        end = min(video_duration, start + minimum)
        if end - start < minimum:
            start = max(0.0, end - minimum)
    return start, min(end, float(video_duration))


def duration_penalty(duration: float, clip_cfg: ClipConfig) -> float:
    """Penalty (lower is better) for drifting away from the target duration."""

    drift = abs(float(duration) - float(clip_cfg.target_duration))
    penalty = drift * 0.09
    if duration < clip_cfg.min_duration or duration > clip_cfg.max_duration:
        penalty += 5.0
    return penalty


# --------------------------------------------------------------------------- #
# selector
# --------------------------------------------------------------------------- #
class ClipSelector:
    """Turns scored candidates + the model's choice into renderable clips."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def build_clips(
        self,
        selections: Sequence[Dict[str, Any]],
        candidates: Sequence[Candidate],
        transcript: Transcript,
        *,
        video_duration: float,
        wanted: Optional[int] = None,
    ) -> List[Clip]:
        by_id = {candidate.id: candidate for candidate in candidates}
        clips: List[Clip] = []

        ordered_selections = list(selections)
        selected_ids = {int(s.get("id", -1)) for s in ordered_selections}
        if wanted is not None:
            for candidate in sorted(candidates, key=lambda c: -c.overall):
                if candidate.id in selected_ids:
                    continue
                ordered_selections.append({
                    "id": candidate.id, "title": candidate.title, "hook": candidate.hook,
                    "reason": candidate.reason, "score": candidate.overall, "fallback": True,
                })
                selected_ids.add(candidate.id)
        for selection in ordered_selections:
            candidate = by_id.get(int(selection.get("id", -1)))
            if candidate is None:
                log.warning("Skipping selection for unknown candidate id %r", selection.get("id"))
                continue
            clip = self.build_clip(
                candidate,
                transcript,
                selection=selection,
                video_duration=video_duration,
                index=len(clips) + 1,
            )
            if self._overlaps_existing(clip, clips):
                log.info("Skipping %s - too close to an already selected clip", clip.label)
                continue
            clips.append(clip)
            if wanted is not None and len(clips) >= wanted:
                break

        if not clips:
            raise SelectionError(
                "No valid clip could be built from the model's selection.",
                "Try a larger Whisper model or a video with more spoken content.",
            )
        return clips

    def build_clip(
        self,
        candidate: Candidate,
        transcript: Transcript,
        *,
        selection: Optional[Dict[str, Any]] = None,
        video_duration: float,
        index: int = 1,
    ) -> Clip:
        selection = dict(selection or {})
        start, end = validate_timestamps(candidate.start, candidate.end, video_duration)

        # The model may suggest a tweak; accept it only if it stays inside the
        # candidate window (plus a small margin) and keeps a sane duration.
        llm_start = selection.get("llm_start")
        llm_end = selection.get("llm_end")
        if llm_start is not None and llm_end is not None:
            try:
                cand_start, cand_end = validate_timestamps(llm_start, llm_end, video_duration)
            except SelectionError:
                cand_start = cand_end = None  # type: ignore[assignment]
            if cand_start is not None:
                inside = (
                    cand_start >= start - SEARCH_WINDOW
                    and cand_end <= end + SEARCH_WINDOW
                    and (cand_end - cand_start) >= self.config.clip.min_duration * 0.6
                )
                if inside:
                    start, end = cand_start, cand_end
                else:
                    log.debug(
                        "Ignoring out-of-range LLM boundaries %.1f-%.1f for candidate %s",
                        cand_start,
                        cand_end,
                        candidate.label,
                    )

        start, end, notes = self.refine_boundaries(transcript, start, end, video_duration)
        text = transcript.text_in_range(start, end) or candidate.text

        score = float(selection.get("score") or 0.0) or float(candidate.overall)
        clip = Clip(
            index=index,
            start=start,
            end=end,
            title=str(selection.get("title") or candidate.title or "").strip(),
            hook=str(selection.get("hook") or candidate.hook or "").strip(),
            reason=str(selection.get("reason") or candidate.reason or "").strip(),
            score=clamp(score, 0.0, 10.0),
            candidate_id=candidate.id,
            scores=dict(candidate.scores),
            text=text,
            boundary_notes=notes,
        )
        return clip

    # -- boundary logic --------------------------------------------------- #
    def refine_boundaries(
        self,
        transcript: Transcript,
        start: float,
        end: float,
        video_duration: float,
    ) -> Tuple[float, float, List[str]]:
        """Snap ``[start, end]`` to natural speech boundaries."""

        clip_cfg = self.config.clip
        notes: List[str] = []
        starts = self._start_options(transcript, start)
        ends = self._end_options(transcript, end)

        best: Optional[Tuple[float, float, float]] = None  # (score, start, end)
        for s_time, s_is_sentence in starts:
            for e_time, e_is_sentence in ends:
                duration = e_time - s_time
                if duration <= 0:
                    continue
                if duration < clip_cfg.min_duration or duration > clip_cfg.max_duration:
                    continue
                score = 0.0
                score += 2.2 if s_is_sentence else 0.0
                score += 1.8 if e_is_sentence else 0.0
                score -= duration_penalty(duration, clip_cfg)
                # Stay close to what the analysis actually judged.
                score -= 0.05 * (abs(s_time - start) + abs(e_time - end))
                if best is None or score > best[0]:
                    best = (score, s_time, e_time)

        if best is None:
            fitted_start, fitted_end = fit_duration(start, end, clip_cfg, video_duration)
            notes.append("no clean sentence boundary found within the duration window")
            return self._apply_padding(transcript, fitted_start, fitted_end, video_duration, notes)

        _, new_start, new_end = best
        if abs(new_start - start) > 0.05:
            notes.append(f"start moved {new_start - start:+.2f}s to a sentence boundary")
        if abs(new_end - end) > 0.05:
            notes.append(f"end moved {new_end - end:+.2f}s to a sentence boundary")
        return self._apply_padding(transcript, new_start, new_end, video_duration, notes)

    def _apply_padding(
        self,
        transcript: Transcript,
        start: float,
        end: float,
        video_duration: float,
        notes: List[str],
    ) -> Tuple[float, float, List[str]]:
        clip_cfg = self.config.clip
        padded_start = max(0.0, start - float(clip_cfg.lead_in))
        padded_end = min(float(video_duration), end + float(clip_cfg.tail_out))

        # Never let padding break the hard duration limits.
        if padded_end - padded_start > clip_cfg.max_duration:
            padded_end = padded_start + clip_cfg.max_duration
        if padded_end - padded_start < clip_cfg.min_duration:
            padded_end = min(float(video_duration), padded_start + clip_cfg.min_duration)
            if padded_end - padded_start < clip_cfg.min_duration:
                padded_start = max(0.0, padded_end - clip_cfg.min_duration)

        duration = padded_end - padded_start
        if duration < clip_cfg.min_duration - 0.5:
            notes.append(
                f"clip is {duration:.1f}s, shorter than the configured minimum "
                f"({clip_cfg.min_duration:.0f}s) - the source does not allow more"
            )
        return round(padded_start, 3), round(padded_end, 3), notes

    def _start_options(self, transcript: Transcript, start: float) -> List[Tuple[float, bool]]:
        options: List[Tuple[float, bool]] = [(round(float(start), 3), False)]
        for segment in transcript.segments:
            if segment.start < start - SEARCH_WINDOW:
                continue
            if segment.start > start + SEARCH_WINDOW:
                break
            options.append((round(float(segment.start), 3), transcript.starts_sentence(segment)))
        # Mark the original start as a sentence start when it really is one.
        for segment in transcript.segments:
            if abs(segment.start - start) < 0.05 and transcript.starts_sentence(segment):
                options[0] = (options[0][0], True)
                break
        return _dedupe_options(options)

    def _end_options(self, transcript: Transcript, end: float) -> List[Tuple[float, bool]]:
        options: List[Tuple[float, bool]] = [(round(float(end), 3), False)]
        for segment in transcript.segments:
            if segment.end < end - SEARCH_WINDOW:
                continue
            if segment.end > end + SEARCH_WINDOW:
                break
            options.append((round(float(segment.end), 3), segment.ends_sentence()))
            # Word ends give finer control when a segment overshoots the max.
            for word in segment.words:
                if end - SEARCH_WINDOW <= word.end <= end + SEARCH_WINDOW:
                    is_sentence = word.text.strip().endswith((".", "!", "?", "…"))
                    options.append((round(float(word.end), 3), is_sentence))
        for segment in transcript.segments:
            if abs(segment.end - end) < 0.05 and segment.ends_sentence():
                options[0] = (options[0][0], True)
                break
        return _dedupe_options(options)

    def _overlaps_existing(self, clip: Clip, existing: Sequence[Clip]) -> bool:
        gap = float(self.config.clip.min_gap_between_clips)
        for other in existing:
            if clip.start < other.end + gap and other.start < clip.end + gap:
                return True
        return False


def _dedupe_options(options: Sequence[Tuple[float, bool]]) -> List[Tuple[float, bool]]:
    """Collapse duplicate times, keeping the sentence-boundary flag if any."""

    merged: Dict[float, bool] = {}
    for time_value, is_sentence in options:
        key = round(float(time_value), 2)
        merged[key] = merged.get(key, False) or bool(is_sentence)
    return sorted(merged.items())
