"""Subtitle generation: SRT, WebVTT and burn-in ASS.

Captions are built from Whisper's *word* timestamps where available, which is
what makes short-form captions feel tight: cues are short, break on natural
pauses and sentence ends, and never exceed the configured line budget.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .config import AppConfig, SubtitleConfig, parse_resolution
from .logger import get_logger
from .transcript import SENTENCE_END_RE, Transcript, Word, shift_words
from .utils import ass_timestamp, srt_timestamp, vtt_timestamp

log = get_logger("subtitles")

_ALIGNMENT = {"bottom": 2, "center": 5, "top": 8}

#: Mean glyph advance as a fraction of font size for a bold sans face. Used
#: only to guarantee captions fit inside the frame, so an estimate is enough.
_AVG_GLYPH_WIDTH = 0.55


@dataclass
class Cue:
    """One caption, already wrapped into display lines."""

    index: int
    start: float
    end: float
    lines: List[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


# --------------------------------------------------------------------------- #
# text helpers
# --------------------------------------------------------------------------- #
def wrap_lines(text: str, max_chars: int, max_lines: int) -> List[str]:
    """Greedy wrap that balances the last two lines and never exceeds ``max_lines``."""

    words = [w for w in re.split(r"\s+", text.strip()) if w]
    if not words:
        return []
    max_chars = max(8, int(max_chars))
    max_lines = max(1, int(max_lines))

    lines: List[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= max_chars or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)

    if len(lines) > max_lines:
        # Re-flow: distribute words evenly across the allowed lines.
        per_line = max(1, -(-len(words) // max_lines))
        lines = [" ".join(words[i : i + per_line]) for i in range(0, len(words), per_line)]
        if len(lines) > max_lines:
            head = lines[: max_lines - 1]
            tail = " ".join(lines[max_lines - 1 :])
            lines = head + [tail]
    return [line.strip() for line in lines if line.strip()]


def hex_to_ass_color(value: str, alpha: int = 0) -> str:
    """``#RRGGBB`` -> ``&H{AA}{BB}{GG}{RR}`` (ASS stores colours reversed)."""

    text = str(value).strip().lstrip("#")
    if len(text) == 3:
        text = "".join(ch * 2 for ch in text)
    if len(text) == 8 and re.fullmatch(r"[0-9A-Fa-f]{8}", text):  # #RRGGBBAA
        alpha = 255 - int(text[6:8], 16)
        text = text[:6]
    if not re.fullmatch(r"[0-9A-Fa-f]{6}", text):
        text = "FFFFFF"
    red, green, blue = text[0:2], text[2:4], text[4:6]
    return f"&H{max(0, min(255, int(alpha))):02X}{blue}{green}{red}".upper()


def escape_ass_text(text: str) -> str:
    """Escape user text for an ASS dialogue line."""

    return (
        text.replace("\\", "\\\\")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("\r", "")
        .replace("\n", "\\N")
    )


# --------------------------------------------------------------------------- #
# generator
# --------------------------------------------------------------------------- #
class SubtitleGenerator:
    """Builds cue lists and writes subtitle files for a clip."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.settings: SubtitleConfig = config.subtitles

    # -- cue construction -------------------------------------------------- #
    def build_cues(self, transcript: Transcript, start: float, end: float) -> List[Cue]:
        """Cues for ``[start, end]``, rebased so the clip begins at 0.0."""

        words = transcript.words_in_range(start, end)
        if words:
            words = shift_words(
                [
                    Word(
                        start=max(start, w.start),
                        end=min(end, w.end),
                        text=w.text,
                        probability=w.probability,
                    )
                    for w in words
                    if w.text.strip()
                ],
                start,
            )
            cues = self._cues_from_words(words, duration=end - start)
        else:
            cues = self._cues_from_segments(transcript, start, end)

        if self.settings.uppercase:
            for cue in cues:
                cue.lines = [line.upper() for line in cue.lines]
        return cues

    def _cues_from_words(self, words: Sequence[Word], duration: float) -> List[Cue]:
        budget = max(10, int(self.settings.max_chars_per_line) * max(1, int(self.settings.max_lines)))
        max_cue = float(self.settings.max_cue_duration)

        cues: List[Cue] = []
        buffer: List[Word] = []

        def flush() -> None:
            if not buffer:
                return
            text = " ".join(w.text.strip() for w in buffer).strip()
            lines = wrap_lines(text, self.settings.max_chars_per_line, self.settings.max_lines)
            if not lines:
                buffer.clear()
                return
            start_time = max(0.0, buffer[0].start)
            end_time = max(start_time + 0.4, buffer[-1].end)
            cues.append(Cue(index=len(cues) + 1, start=start_time, end=end_time, lines=lines))
            buffer.clear()

        for index, word in enumerate(words):
            buffer.append(word)
            text_so_far = " ".join(w.text.strip() for w in buffer).strip()
            cue_duration = buffer[-1].end - buffer[0].start
            ends_sentence = bool(SENTENCE_END_RE.search(word.text.strip()))
            gap_ahead = (
                words[index + 1].start - word.end if index + 1 < len(words) else float("inf")
            )

            if (
                len(text_so_far) >= budget
                or cue_duration >= max_cue
                or ends_sentence
                or gap_ahead >= 0.55
            ):
                flush()
        flush()

        # Keep cues on screen until the next one starts (no flicker), and clamp
        # the final cue to the clip length.
        for index, cue in enumerate(cues):
            next_start = cues[index + 1].start if index + 1 < len(cues) else duration
            cue.end = min(max(cue.end, cue.start + 0.5), max(next_start, cue.start + 0.5), duration)
        return [cue for cue in cues if cue.duration > 0.05]

    def _cues_from_segments(self, transcript: Transcript, start: float, end: float) -> List[Cue]:
        cues: List[Cue] = []
        for segment in transcript.segments_in_range(start, end, overlap_ratio=0.2):
            seg_start = max(0.0, max(segment.start, start) - start)
            seg_end = max(seg_start + 0.5, min(segment.end, end) - start)
            lines = wrap_lines(segment.text, self.settings.max_chars_per_line, self.settings.max_lines)
            if not lines:
                continue
            # Split long segments into readable chunks of equal length.
            chunk_count = max(1, int(round((seg_end - seg_start) / float(self.settings.max_cue_duration))))
            if chunk_count == 1 or len(lines) <= self.settings.max_lines:
                cues.append(Cue(index=len(cues) + 1, start=seg_start, end=seg_end, lines=lines))
                continue
            per_chunk = max(1, -(-len(lines) // chunk_count))
            span = (seg_end - seg_start) / chunk_count
            for chunk_index, offset in enumerate(range(0, len(lines), per_chunk)):
                chunk_start = seg_start + chunk_index * span
                cues.append(
                    Cue(
                        index=len(cues) + 1,
                        start=chunk_start,
                        end=min(seg_end, chunk_start + span),
                        lines=lines[offset : offset + per_chunk],
                    )
                )
        for index, cue in enumerate(cues):
            cue.index = index + 1
        return cues

    # -- writers ----------------------------------------------------------- #
    def write_srt(self, cues: Sequence[Cue], path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        blocks = []
        for index, cue in enumerate(cues, start=1):
            blocks.append(
                f"{index}\n{srt_timestamp(cue.start)} --> {srt_timestamp(cue.end)}\n{cue.text}\n"
            )
        path.write_text("\n".join(blocks), encoding="utf-8")
        return path

    def write_vtt(self, cues: Sequence[Cue], path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        blocks = ["WEBVTT", ""]
        for index, cue in enumerate(cues, start=1):
            blocks.append(str(index))
            blocks.append(f"{vtt_timestamp(cue.start)} --> {vtt_timestamp(cue.end)}")
            blocks.append(cue.text)
            blocks.append("")
        path.write_text("\n".join(blocks), encoding="utf-8")
        return path

    def write_ass(self, cues: Sequence[Cue], path: Path, *, resolution: Tuple[int, int]) -> Path:
        """Styled ASS for burn-in at a specific output resolution."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        width, height = resolution
        settings = self.settings

        # The configured font size is authored for a 1080px tall frame and
        # scales with output height, so captions keep the same relative size
        # (and the same readability) in both 1920x1080 and 1080x1920.
        scale = max(0.4, height / 1080.0)
        font_size = int(round(float(settings.font_size) * scale))
        margin_v = int(round(float(settings.margin_ratio) * height))
        margin_h = int(round(0.06 * width))
        alignment = _ALIGNMENT.get(settings.position, 2)
        if settings.position == "center":
            margin_v = int(round(0.05 * height))

        # WrapStyle 2 means libass does no wrapping of its own: we own the line
        # breaks, so a line that is too wide would run off the frame. Re-wrap
        # anything that cannot fit between the horizontal margins at this size.
        fitted_chars = max(8, int((width - 2 * margin_h) / (_AVG_GLYPH_WIDTH * font_size)))
        if fitted_chars < settings.max_chars_per_line:
            log.debug(
                "Re-wrapping captions to %d chars/line for %dx%d", fitted_chars, width, height
            )
        cues = [
            Cue(
                index=cue.index,
                start=cue.start,
                end=cue.end,
                lines=wrap_lines(
                    " ".join(cue.lines),
                    min(settings.max_chars_per_line, fitted_chars),
                    settings.max_lines,
                ),
            )
            for cue in cues
        ]

        header = "\n".join(
            [
                "[Script Info]",
                "; Generated by ai-video-clipper",
                "ScriptType: v4.00+",
                "WrapStyle: 2",
                "YCbCr Matrix: TV.709",
                "ScaledBorderAndShadow: yes",
                f"PlayResX: {width}",
                f"PlayResY: {height}",
                "",
                "[V4+ Styles]",
                "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
                "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
                "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
                "Style: " + ",".join(
                    [
                        "Default",
                        str(settings.font_name),
                        str(font_size),
                        hex_to_ass_color(settings.primary_color),
                        hex_to_ass_color(settings.primary_color),
                        hex_to_ass_color(settings.outline_color),
                        hex_to_ass_color(settings.back_color, alpha=120),
                        "-1" if settings.bold else "0",
                        "0", "0", "0",
                        "100", "100", "0", "0",
                        "1",
                        f"{float(settings.outline) * scale:.1f}",
                        f"{float(settings.shadow) * scale:.1f}",
                        str(alignment),
                        str(margin_h),
                        str(margin_h),
                        str(margin_v),
                        "1",
                    ]
                ),
                "",
                "[Events]",
                "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
            ]
        )
        usable_width = max(1, width - 2 * margin_h)
        events: List[str] = []
        for cue in cues:
            # A cue whose longest line still does not fit (a very long word, or
            # more text than max_lines can hold) gets its own smaller size
            # rather than being cut off at the frame edge.
            longest = max((len(line) for line in cue.lines), default=0)
            override = ""
            if longest and longest * _AVG_GLYPH_WIDTH * font_size > usable_width:
                shrunk = int(usable_width / (_AVG_GLYPH_WIDTH * longest))
                shrunk = max(int(font_size * 0.4), min(font_size, shrunk))
                override = "{\\fs%d}" % shrunk
            events.append(
                "Dialogue: 0,{start},{end},Default,,0,0,0,,{override}{text}".format(
                    start=ass_timestamp(cue.start),
                    end=ass_timestamp(cue.end),
                    override=override,
                    text=escape_ass_text(cue.text),
                )
            )
        path.write_text(header + "\n" + "\n".join(events) + "\n", encoding="utf-8")
        return path

    # -- orchestration ----------------------------------------------------- #
    def generate(
        self,
        transcript: Transcript,
        start: float,
        end: float,
        destination: Path,
        *,
        stem: str = "subtitles",
        formats: Sequence[str] = ("16:9", "9:16"),
    ) -> Dict[str, Path]:
        """Write every subtitle artefact for one clip.

        Returns a mapping of logical name -> path, e.g.
        ``{"srt": ..., "vtt": ..., "ass_16:9": ..., "ass_9:16": ...}``.
        """

        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        cues = self.build_cues(transcript, start, end)
        if not cues:
            log.warning("No captions could be generated for %.2f-%.2f", start, end)
            return {}

        log.info("Generating subtitles (%d cues)", len(cues))
        out: Dict[str, Path] = {
            "srt": self.write_srt(cues, destination / f"{stem}.srt"),
            "vtt": self.write_vtt(cues, destination / f"{stem}.vtt"),
        }
        for fmt in formats:
            resolution = self.config.video.resolution_for(fmt)
            key = "ass_vertical" if fmt == "9:16" else "ass_landscape"
            suffix = "9x16" if fmt == "9:16" else "16x9"
            out[key] = self.write_ass(
                cues, destination / f"{stem}_{suffix}.ass", resolution=resolution
            )
        return out
