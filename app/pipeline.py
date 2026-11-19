"""Pipeline orchestration.

    URL -> download -> audio -> transcript -> candidates -> Qwen scoring
        -> final selection -> cut -> subtitles -> 16:9 / 9:16 render -> output/

Every expensive stage is cached inside the job workspace, so a crash during
rendering (or a second run with different formats) never re-downloads or
re-transcribes anything.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .candidate_generator import Candidate, CandidateGenerator
from .clip_selector import Clip, ClipSelector
from .config import AppConfig
from .downloader import SourceVideo, YouTubeDownloader
from .errors import ClipperError, SelectionError
from .llm_analyzer import LLMAnalyzer, OllamaClient
from .logger import get_logger, setup_logging
from .media import FFmpeg
from .metadata_generator import MetadataGenerator
from .preflight import PreflightReport, run_preflight
from .subtitle_generator import SubtitleGenerator
from .transcript import Transcript
from .transcription import Transcriber
from .utils import (
    ensure_dir,
    format_clock,
    format_timestamp,
    read_json,
    sanitize_filename,
    unique_dir,
    validate_youtube_url,
    write_json,
)
from .video_processor import RenderResult, VideoProcessor

log = get_logger("pipeline")

STAGE_DIRS = ("source", "audio", "transcript", "analysis", "clips", "subtitles", "final")


def automatic_clip_count(duration_seconds: float) -> int:
    """Return roughly five clips per source hour, with a hard cap of 30."""

    return min(30, max(1, int(float(duration_seconds) / 3600.0 * 5 + 0.5)))


@dataclass
class ClipArtifacts:
    """Everything produced for a single clip."""

    clip: Clip
    output_dir: Path
    renders: List[RenderResult] = field(default_factory=list)
    subtitle_files: Dict[str, Path] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    analysis: Dict[str, Any] = field(default_factory=dict)
    source_clip: Optional[Path] = None


@dataclass
class JobResult:
    """The result of one pipeline run."""

    job_id: str
    url: str
    workspace: Path
    output_root: Path
    source: SourceVideo
    transcript: Transcript
    candidates: List[Candidate]
    clips: List[ClipArtifacts] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    preflight: Optional[PreflightReport] = None

    @property
    def primary(self) -> Optional[ClipArtifacts]:
        return self.clips[0] if self.clips else None


class Pipeline:
    """End-to-end clipping pipeline."""

    def __init__(self, config: AppConfig, *, skip_llm_check: bool = False) -> None:
        self.config = config
        self.skip_llm_check = skip_llm_check
        self.ffmpeg = FFmpeg(config.runtime.ffmpeg_path, config.runtime.ffprobe_path)
        self.downloader = YouTubeDownloader(config)
        self.transcriber = Transcriber(config)
        self.candidate_generator = CandidateGenerator(config)
        self.ollama = OllamaClient(config.ollama)
        self.analyzer = LLMAnalyzer(config, client=self.ollama)
        self.selector = ClipSelector(config)
        self.subtitles = SubtitleGenerator(config)
        self.video = VideoProcessor(config)
        self.metadata = MetadataGenerator(config, analyzer=self.analyzer)

    # -- entry point ------------------------------------------------------ #
    def run(self, url: str, *, clips: Optional[int] = None) -> JobResult:
        started = time.time()
        url = validate_youtube_url(url)
        requested_clips = max(1, int(clips)) if clips is not None else None
        log.info("Starting video processing")

        report = run_preflight(self.config, require_llm=not self.skip_llm_check)

        job_id = self._job_id(url)
        workspace = self._prepare_workspace(job_id)
        setup_logging(self.config.runtime.log_level, workspace / "clipper.log")
        log.info("Job %s (workspace: %s)", job_id, workspace)

        use_cache = bool(self.config.runtime.use_cache)

        # 1-2. download + audio
        source = self.downloader.download(url, workspace, use_cache=use_cache)
        wanted_clips = requested_clips if requested_clips is not None else (
            automatic_clip_count(source.duration) if self.config.clip.auto_count
            else max(1, int(self.config.clip.clips))
        )
        log.info("Targeting %d clips", wanted_clips)
        audio_path = self.transcriber.extract_audio(source.path, workspace, use_cache=use_cache)

        # 3. transcription
        try:
            transcript = self.transcriber.transcribe(
                audio_path,
                workspace,
                use_cache=use_cache,
                expected_duration=source.duration,
            )
        finally:
            # Whisper and Ollama share the laptop's 4 GB GPU. Free Whisper's
            # weights before scoring and clip rendering begin.
            self.transcriber.unload_model()

        # 4-5. candidates + LLM scoring
        candidates = self._analyze_candidates(transcript, source, workspace, use_cache=use_cache)

        # 6. final selection
        selections = self._select_clips(
            candidates, transcript, source, workspace, wanted=wanted_clips, use_cache=use_cache
        )
        built = self.selector.build_clips(
            selections, candidates, transcript, video_duration=source.duration, wanted=wanted_clips
        )
        for clip in built:
            log.info(
                "Selected clip: %s (%.1fs, score %.1f) %s",
                clip.label,
                clip.duration,
                clip.score,
                f"- {clip.title}" if clip.title else "",
            )

        # 7-10. render each clip
        output_root = self._output_root(source)
        artifacts: List[ClipArtifacts] = []
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="clip-render") as executor:
            futures = [executor.submit(
                self._produce_clip,
                    clip=clip,
                    source=source,
                    transcript=transcript,
                    candidates=candidates,
                    workspace=workspace,
                    output_root=output_root,
                    multiple=True,
                    selection=next((s for s in selections if s.get("id") == clip.candidate_id), None),
                ) for clip in built]
            artifacts = [future.result() for future in futures]

        self._cleanup(workspace)
        elapsed = time.time() - started
        log.info("Finished in %s", format_timestamp(elapsed))

        return JobResult(
            job_id=job_id,
            url=url,
            workspace=workspace,
            output_root=output_root,
            source=source,
            transcript=transcript,
            candidates=candidates,
            clips=artifacts,
            elapsed_seconds=elapsed,
            preflight=report,
        )

    # -- stages ----------------------------------------------------------- #
    def _analyze_candidates(
        self,
        transcript: Transcript,
        source: SourceVideo,
        workspace: Path,
        *,
        use_cache: bool,
    ) -> List[Candidate]:
        analysis_dir = ensure_dir(workspace / "analysis")
        cache_path = analysis_dir / "candidates.json"
        fingerprint = self._fingerprint(
            stage="candidates",
            extra={
                "prompt_version": 2,
                "whisper": self.config.whisper.model,
                "beam_size": self.config.whisper.beam_size,
                "language": self.config.whisper.language,
                "segments": len(transcript),
                "min": self.config.clip.min_duration,
                "max": self.config.clip.max_duration,
                "target": self.config.clip.target_duration,
                "max_candidates": self.config.clip.max_candidates,
                "llm": self.config.ollama.model,
                "weights": self.config.scoring.as_dict(),
            },
        )

        if use_cache and cache_path.is_file():
            cached = self._load_candidate_cache(cache_path, fingerprint)
            if cached:
                log.info("Reusing cached candidate analysis (%d candidates)", len(cached))
                return cached

        log.info("Generating candidate clips")
        candidates = self.candidate_generator.generate(transcript)
        if not candidates:
            raise SelectionError(
                "No candidate windows could be generated from the transcript.",
                "The video may be too short or contain almost no speech.",
            )

        context = {
            "title": source.title,
            "duration_label": format_timestamp(source.duration),
            "language": transcript.language,
        }
        candidates = self.analyzer.score_candidates(candidates, context)
        write_json(
            cache_path,
            {
                "fingerprint": fingerprint,
                "candidates": [candidate.to_dict() for candidate in candidates],
            },
        )
        return candidates

    def _select_clips(
        self,
        candidates: Sequence[Candidate],
        transcript: Transcript,
        source: SourceVideo,
        workspace: Path,
        *,
        wanted: int,
        use_cache: bool,
    ) -> List[Dict[str, Any]]:
        analysis_dir = ensure_dir(workspace / "analysis")
        cache_path = analysis_dir / "selection.json"
        fingerprint = self._fingerprint(
            stage="selection",
            extra={
                "prompt_version": 2,
                "wanted": wanted,
                "llm": self.config.ollama.model,
                "candidates": [c.id for c in candidates],
                "overall": [round(c.overall, 3) for c in candidates],
            },
        )

        if use_cache and cache_path.is_file():
            try:
                payload = read_json(cache_path)
                if payload.get("fingerprint") == fingerprint and payload.get("selections"):
                    log.info("Reusing cached final selection")
                    return list(payload["selections"])
            except Exception as exc:
                log.debug("Ignoring unreadable selection cache: %s", exc)

        log.info("Selecting the final clip with %s", self.config.ollama.model)
        selections = self.analyzer.select_final(
            candidates,
            {
                "title": source.title,
                "duration_label": format_timestamp(source.duration),
            },
            wanted=wanted,
        )
        if not selections:
            raise SelectionError("The model did not return a usable clip selection.")
        write_json(cache_path, {"fingerprint": fingerprint, "selections": selections})
        return selections

    def _produce_clip(
        self,
        *,
        clip: Clip,
        source: SourceVideo,
        transcript: Transcript,
        candidates: Sequence[Candidate],
        workspace: Path,
        output_root: Path,
        multiple: bool,
        selection: Optional[Dict[str, Any]],
    ) -> ClipArtifacts:
        output_dir = ensure_dir(output_root / f"clip_{clip.index:02d}" if multiple else output_root)
        clips_dir = ensure_dir(workspace / "clips")
        subtitle_dir = ensure_dir(workspace / "subtitles" / f"clip_{clip.index:02d}")

        # 7. cut the segment
        raw_clip = self.video.cut_clip(
            source.path, clip.start, clip.end, clips_dir / f"clip_{clip.index:02d}.mp4"
        )
        clip_info = self.ffmpeg.probe(raw_clip)

        # 8. subtitles
        subtitle_files: Dict[str, Path] = {}
        if self.config.subtitles.enabled:
            subtitle_files = self.subtitles.generate(
                transcript,
                clip.start,
                clip.end,
                subtitle_dir,
                stem="subtitles",
                formats=self.config.video.formats,
            )
        else:
            log.info("Subtitles disabled - skipping caption generation")

        # 9. smart crop detection (only needed for vertical output)
        subject_center, crop_strategy = 0.5, "center"
        if "9:16" in self.config.video.formats:
            subject_center, crop_strategy = self.video.detect_subject_center(
                raw_clip, workspace / "frames" / f"clip_{clip.index:02d}", info=clip_info
            )

        # 10. render every requested format
        renders: List[RenderResult] = []
        for fmt in self.config.video.formats:
            filename = "shorts_9x16.mp4" if fmt == "9:16" else "youtube_16x9.mp4"
            ass_key = "ass_vertical" if fmt == "9:16" else "ass_landscape"
            renders.append(
                self.video.render(
                    raw_clip,
                    output_dir / filename,
                    fmt,
                    subtitle_path=subtitle_files.get(ass_key),
                    subject_center=subject_center,
                    crop_strategy=crop_strategy,
                    info=clip_info,
                )
            )

        # Keep the untouched cut next to the renders.
        original_copy = output_dir / "original_clip.mp4"
        shutil.copy2(raw_clip, original_copy)

        for key in ("srt", "vtt"):
            if key in subtitle_files:
                shutil.copy2(subtitle_files[key], output_dir / f"subtitles.{key}")

        # 11-12. reports
        self.metadata.write_transcripts(transcript=transcript, clip=clip, output_dir=output_dir)
        analysis = self.metadata.build_analysis(
            clip=clip,
            candidates=candidates,
            output_dir=output_dir,
            source=source,
            selection_payload=selection,
        )
        metadata = self.metadata.build_metadata(
            source=source,
            clip=clip,
            renders=renders,
            subtitle_files={
                key: path for key, path in subtitle_files.items() if key in {"srt", "vtt"}
            },
            output_dir=output_dir,
            transcript_language=transcript.language,
        )

        log.info("Saved clip %d to %s", clip.index, output_dir)
        return ClipArtifacts(
            clip=clip,
            output_dir=output_dir,
            renders=renders,
            subtitle_files=subtitle_files,
            metadata=metadata,
            analysis=analysis,
            source_clip=original_copy,
        )

    # -- helpers ---------------------------------------------------------- #
    def _job_id(self, url: str) -> str:
        from .utils import extract_video_id

        video_id = extract_video_id(url) or hashlib.sha256(url.encode("utf-8")).hexdigest()[:11]
        return sanitize_filename(video_id, max_length=32, fallback="job")

    def _prepare_workspace(self, job_id: str) -> Path:
        workspace = ensure_dir(Path(self.config.paths.workspace_dir) / job_id)
        for name in STAGE_DIRS:
            ensure_dir(workspace / name)
        return workspace

    def _output_root(self, source: SourceVideo) -> Path:
        base = Path(self.config.paths.output_dir)
        ensure_dir(base)
        name = sanitize_filename(f"{source.title}-{source.video_id}", max_length=90, fallback=source.video_id)
        target = base / name
        if target.exists() and any(target.iterdir()):
            # Never overwrite a previous export.
            target = unique_dir(base, f"{name}-{time.strftime('%Y%m%d-%H%M%S')}")
        return ensure_dir(target)

    def _fingerprint(self, *, stage: str, extra: Dict[str, Any]) -> str:
        payload = json.dumps({"stage": stage, **extra}, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def _load_candidate_cache(self, path: Path, fingerprint: str) -> List[Candidate]:
        try:
            payload = read_json(path)
        except Exception as exc:
            log.debug("Ignoring unreadable candidate cache: %s", exc)
            return []
        if payload.get("fingerprint") != fingerprint:
            return []
        out: List[Candidate] = []
        for entry in payload.get("candidates") or []:
            try:
                out.append(Candidate.from_dict(entry))
            except Exception:
                return []
        return out

    def _cleanup(self, workspace: Path) -> None:
        if self.config.runtime.keep_temp_files:
            log.info("Keeping temporary files in %s", workspace)
            return
        for name in ("audio", "frames", "clips"):
            target = Path(workspace) / name
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
        log.info("Temporary files removed (source + transcript kept for re-runs)")
