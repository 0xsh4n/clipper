"""Speech-to-text with faster-whisper.

Produces word-level timestamps (needed for natural clip boundaries and
accurate captions) and caches the result as ``transcript/transcript.json`` so a
crashed render never costs a re-transcription.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import AppConfig
from .errors import TranscriptionError
from .logger import get_logger
from .media import FFmpeg
from .transcript import Transcript, build_transcript
from .utils import format_timestamp

log = get_logger("transcription")


def detect_device(preference: str = "auto") -> Tuple[str, str]:
    """Resolve ``(device, compute_type)``, preferring CUDA when it works."""

    preference = (preference or "auto").lower()
    if preference == "cpu":
        return "cpu", "int8"
    if preference == "cuda":
        return "cuda", "float16"

    # auto: probe for a usable CUDA runtime, fall back silently.
    try:
        import ctranslate2

        count = int(ctranslate2.get_cuda_device_count())
        if count > 0:
            log.info("CUDA device detected (%d) - using GPU acceleration", count)
            return "cuda", "float16"
    except Exception as exc:  # pragma: no cover - depends on the host
        log.debug("CUDA probe failed (%s)", exc)

    try:
        import torch  # optional

        if torch.cuda.is_available():  # pragma: no cover - depends on the host
            log.info("CUDA available via torch - using GPU acceleration")
            return "cuda", "float16"
    except Exception:
        pass

    log.info("No CUDA device found - running Whisper on CPU")
    return "cpu", "int8"


class Transcriber:
    """Thin, cached wrapper around :class:`faster_whisper.WhisperModel`."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.ffmpeg = FFmpeg(config.runtime.ffmpeg_path, config.runtime.ffprobe_path)
        self._model: Any = None
        self._device: str = ""
        self._compute_type: str = ""

    # -- public API ------------------------------------------------------- #
    def extract_audio(self, source_path: Path, workspace: Path, *, use_cache: bool = True) -> Path:
        audio_dir = Path(workspace) / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        audio_path = audio_dir / "audio.wav"
        if use_cache and audio_path.is_file() and audio_path.stat().st_size > 1024:
            log.info("Reusing cached audio: %s", audio_path.name)
            return audio_path
        log.info("Extracting audio")
        return self.ffmpeg.extract_audio(source_path, audio_path)

    def transcribe(
        self,
        audio_path: Path,
        workspace: Path,
        *,
        use_cache: bool = True,
        expected_duration: float = 0.0,
    ) -> Transcript:
        transcript_dir = Path(workspace) / "transcript"
        transcript_dir.mkdir(parents=True, exist_ok=True)
        transcript_path = transcript_dir / "transcript.json"

        if use_cache and transcript_path.is_file():
            cached = self._load_cached(transcript_path)
            if cached is not None:
                log.info(
                    "Reusing cached transcript (%d segments, %d words)",
                    len(cached),
                    cached.word_count,
                )
                return cached

        log.info("Transcribing (model: %s)", self.config.whisper.model)
        transcript = self._run_whisper(audio_path, expected_duration=expected_duration)
        if not transcript.segments:
            raise TranscriptionError(
                "The transcript is empty - no speech was detected.",
                "This pipeline needs a video with spoken content. Check the audio track, "
                "or try a larger Whisper model (--whisper medium).",
            )
        transcript.save(transcript_path)
        log.info(
            "Transcript generated: %d segments, %d words, language=%s",
            len(transcript),
            transcript.word_count,
            transcript.language or "unknown",
        )
        return transcript

    # -- internals -------------------------------------------------------- #
    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise TranscriptionError(
                "faster-whisper is not installed.",
                "pip install -r requirements.txt",
            ) from exc

        device, compute_type = detect_device(self.config.whisper.device)
        if self.config.whisper.compute_type not in {"auto", "", None}:
            compute_type = str(self.config.whisper.compute_type)

        kwargs: Dict[str, Any] = {
            "device": device,
            "compute_type": compute_type,
            "cpu_threads": max(1, (os.cpu_count() or 4) - 1) if device == "cpu" else 0,
        }
        if self.config.whisper.download_root:
            kwargs["download_root"] = str(self.config.whisper.download_root)

        try:
            model = WhisperModel(self.config.whisper.model, **kwargs)
        except Exception as exc:
            if device == "cuda":
                log.warning("CUDA init failed (%s) - falling back to CPU", exc)
                try:
                    model = WhisperModel(
                        self.config.whisper.model,
                        device="cpu",
                        compute_type="int8",
                        cpu_threads=max(1, (os.cpu_count() or 4) - 1),
                    )
                    device, compute_type = "cpu", "int8"
                except Exception as cpu_exc:
                    raise TranscriptionError(
                        f"Could not load Whisper model {self.config.whisper.model!r}: {cpu_exc}",
                        "Check the model name and that the model can be downloaded.",
                    ) from cpu_exc
            else:
                raise TranscriptionError(
                    f"Could not load Whisper model {self.config.whisper.model!r}: {exc}",
                    "Check the model name (tiny/base/small/medium/large-v3) and disk space "
                    "in the Hugging Face cache.",
                ) from exc

        self._model = model
        self._device = device
        self._compute_type = compute_type
        log.info("Whisper loaded on %s (%s)", device, compute_type)
        return model

    def _run_whisper(self, audio_path: Path, *, expected_duration: float = 0.0) -> Transcript:
        model = self._load_model()
        language = self.config.whisper.language or None

        try:
            segments_iter, info = model.transcribe(
                str(audio_path),
                language=language,
                beam_size=int(self.config.whisper.beam_size),
                vad_filter=bool(self.config.whisper.vad_filter),
                vad_parameters={"min_silence_duration_ms": 400},
                word_timestamps=True,
                condition_on_previous_text=False,
            )
        except Exception as exc:
            raise TranscriptionError(f"Transcription failed: {exc}") from exc

        total = float(getattr(info, "duration", 0.0) or expected_duration or 0.0)
        raw_segments: List[Dict[str, Any]] = []
        next_log = 0.0
        try:
            for segment in segments_iter:
                raw_segments.append(
                    {
                        "start": float(segment.start or 0.0),
                        "end": float(segment.end or 0.0),
                        "text": str(segment.text or "").strip(),
                        "no_speech_prob": float(getattr(segment, "no_speech_prob", 0.0) or 0.0),
                        "avg_logprob": float(getattr(segment, "avg_logprob", 0.0) or 0.0),
                        "words": [
                            {
                                "start": float(getattr(w, "start", 0.0) or 0.0),
                                "end": float(getattr(w, "end", 0.0) or 0.0),
                                "word": str(getattr(w, "word", "") or ""),
                                "probability": float(getattr(w, "probability", 1.0) or 1.0),
                            }
                            for w in (getattr(segment, "words", None) or [])
                        ],
                    }
                )
                if total > 0 and float(segment.end or 0.0) >= next_log:
                    percent = min(100.0, 100.0 * float(segment.end or 0.0) / total)
                    log.info(
                        "Transcribing: %s / %s (%.0f%%)",
                        format_timestamp(float(segment.end or 0.0)),
                        format_timestamp(total),
                        percent,
                    )
                    next_log = float(segment.end or 0.0) + max(60.0, total / 20.0)
        except Exception as exc:
            raise TranscriptionError(f"Transcription failed mid-stream: {exc}") from exc

        return build_transcript(
            raw_segments,
            language=str(getattr(info, "language", "") or language or ""),
            language_probability=float(getattr(info, "language_probability", 0.0) or 0.0),
            duration=total,
            model=self.config.whisper.model,
            source_path=str(audio_path),
        )

    def _load_cached(self, path: Path) -> Optional[Transcript]:
        try:
            transcript = Transcript.load(path)
        except Exception as exc:
            log.debug("Ignoring unreadable transcript cache: %s", exc)
            return None
        if not transcript.segments:
            return None
        if transcript.model and transcript.model != self.config.whisper.model:
            log.info(
                "Cached transcript was made with %r, now using %r - re-transcribing",
                transcript.model,
                self.config.whisper.model,
            )
            return None
        return transcript
