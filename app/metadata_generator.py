"""Output metadata: ``metadata.json`` and ``clip_analysis.json``.

``metadata.json`` is what a publishing step would consume (title, description,
hashtags, timings). ``clip_analysis.json`` is the audit trail: every candidate
the model saw, its seven scores and why the winner won.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .candidate_generator import Candidate
from .clip_selector import Clip
from .config import AppConfig
from .downloader import SourceVideo
from .llm_analyzer import LLMAnalyzer
from .logger import get_logger
from .transcript import Transcript
from .utils import format_timestamp, write_json
from .video_processor import RenderResult

log = get_logger("metadata")


class MetadataGenerator:
    """Builds and writes the JSON side-cars for a finished clip."""

    def __init__(self, config: AppConfig, analyzer: Optional[LLMAnalyzer] = None) -> None:
        self.config = config
        self.analyzer = analyzer

    # -- publishing metadata ---------------------------------------------- #
    def build_metadata(
        self,
        *,
        source: SourceVideo,
        clip: Clip,
        renders: Sequence[RenderResult],
        subtitle_files: Dict[str, Path],
        output_dir: Path,
        transcript_language: str = "",
    ) -> Dict[str, Any]:
        publishing = {"title": clip.title, "caption": "", "description": "", "hashtags": [], "alt_titles": []}
        if self.analyzer is not None:
            log.info("Generating title, description and hashtags")
            publishing = self.analyzer.generate_publishing_metadata(
                clip.text,
                {
                    "title": source.title,
                    "uploader": source.uploader,
                    "duration": clip.duration,
                    "hook": clip.hook,
                    "reason": clip.reason,
                },
            )

        title = publishing.get("title") or clip.title or source.title
        caption = publishing.get("caption") or publishing.get("description") or clip.hook or clip.text[:300]
        description = publishing.get("description") or caption
        hashtags: List[str] = list(publishing.get("hashtags") or [])

        metadata: Dict[str, Any] = {
            "source_url": source.url,
            "source_title": source.title,
            "source_id": source.video_id,
            "source_channel": source.uploader,
            "source_duration": round(float(source.duration), 2),
            "source_duration_label": format_timestamp(source.duration),
            "source_was_live": bool(source.was_live),
            "clip_index": clip.index,
            "clip_start": round(float(clip.start), 2),
            "clip_end": round(float(clip.end), 2),
            "clip_duration": round(float(clip.duration), 2),
            "clip_timecode": clip.label,
            "title": title,
            "caption": caption,
            "description": description,
            "hashtags": hashtags,
            "alt_titles": list(publishing.get("alt_titles") or []),
            "hook": clip.hook,
            "reason": clip.reason,
            "score": round(float(clip.score), 2),
            "scores": {k: round(float(v), 2) for k, v in clip.scores.items()},
            "language": transcript_language,
            "transcript": clip.text,
            "boundary_notes": list(clip.boundary_notes),
            "outputs": [
                {
                    "format": render.format,
                    "file": render.path.name,
                    "width": render.width,
                    "height": render.height,
                    "duration": round(float(render.duration), 2),
                    "burned_subtitles": render.burned_subtitles,
                    "crop_strategy": render.crop_strategy,
                    "crop_center": (
                        round(float(render.crop_center), 3) if render.crop_center is not None else None
                    ),
                }
                for render in renders
            ],
            "subtitles": {key: path.name for key, path in subtitle_files.items()},
            "models": {
                "llm": self.config.ollama.model,
                "llm_host": self.config.ollama.base_url,
                "whisper": self.config.whisper.model,
            },
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "generator": "ai-video-clipper",
        }
        write_json(Path(output_dir) / "metadata.json", metadata)
        return metadata

    # -- analysis report --------------------------------------------------- #
    def build_analysis(
        self,
        *,
        clip: Clip,
        candidates: Sequence[Candidate],
        output_dir: Path,
        source: SourceVideo,
        selection_payload: Optional[Dict[str, Any]] = None,
        top_n: int = 10,
    ) -> Dict[str, Any]:
        ranked = sorted(candidates, key=lambda c: -c.overall)
        analysis: Dict[str, Any] = {
            # The headline block mirrors the final-selection contract.
            "selected": True,
            "start": round(float(clip.start), 2),
            "end": round(float(clip.end), 2),
            "duration": round(float(clip.duration), 2),
            "timecode": clip.label,
            "title": clip.title,
            "hook": clip.hook,
            "reason": clip.reason,
            "score": round(float(clip.score), 2),
            "scores": {k: round(float(v), 2) for k, v in clip.scores.items()},
            "scoring_weights": self.config.scoring.as_dict(),
            "selection_method": (
                "heuristic-fallback" if (selection_payload or {}).get("fallback") else "qwen-final-selection"
            ),
            "boundary_notes": list(clip.boundary_notes),
            "candidate_id": clip.candidate_id,
            "candidates_considered": len(candidates),
            "source": {
                "url": source.url,
                "title": source.title,
                "duration": round(float(source.duration), 2),
            },
            "duration_policy": {
                "target": self.config.clip.target_duration,
                "min": self.config.clip.min_duration,
                "max": self.config.clip.max_duration,
            },
            "top_candidates": [candidate.to_dict() for candidate in ranked[: max(1, top_n)]],
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        write_json(Path(output_dir) / "clip_analysis.json", analysis)
        return analysis

    # -- transcripts ------------------------------------------------------- #
    def write_transcripts(
        self,
        *,
        transcript: Transcript,
        clip: Clip,
        output_dir: Path,
    ) -> Dict[str, Path]:
        output_dir = Path(output_dir)
        full_path = write_json(output_dir / "transcript.json", transcript.to_dict())

        clip_segments = [
            {
                "start": round(max(0.0, segment.start - clip.start), 3),
                "end": round(max(0.0, segment.end - clip.start), 3),
                "source_start": round(segment.start, 3),
                "source_end": round(segment.end, 3),
                "text": segment.text,
            }
            for segment in transcript.segments_in_range(clip.start, clip.end, overlap_ratio=0.2)
        ]
        clip_path = write_json(
            output_dir / "clip_transcript.json",
            {
                "clip_start": round(float(clip.start), 3),
                "clip_end": round(float(clip.end), 3),
                "duration": round(float(clip.duration), 3),
                "language": transcript.language,
                "text": clip.text,
                "segments": clip_segments,
            },
        )
        return {"transcript": full_path, "clip_transcript": clip_path}
