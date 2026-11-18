"""Startup dependency checks.

The pipeline refuses to download a two hour video only to discover that
Ollama is not running, so everything is verified up front.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .config import AppConfig
from .errors import DependencyError
from .llm_analyzer import OllamaClient
from .logger import get_logger
from .media import FFmpeg

log = get_logger("preflight")


@dataclass
class PreflightReport:
    ffmpeg_version: str = ""
    ollama_version: str = ""
    ollama_model: str = ""
    yt_dlp_version: str = ""
    whisper_available: bool = False
    details: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


def check_yt_dlp() -> str:
    try:
        import yt_dlp  # noqa: WPS433 (runtime import is deliberate)
    except ImportError as exc:
        raise DependencyError(
            "yt-dlp is not installed.",
            "pip install -r requirements.txt   (or: pip install -U yt-dlp)",
        ) from exc
    return str(getattr(yt_dlp.version, "__version__", "unknown"))


def check_faster_whisper() -> bool:
    try:
        import faster_whisper  # noqa: F401
    except ImportError as exc:
        raise DependencyError(
            "faster-whisper is not installed.",
            "pip install -r requirements.txt   (or: pip install faster-whisper)\n"
            "     On Python 3.13+ wheels may be missing; Python 3.10-3.12 is recommended.",
        ) from exc
    return True


def run_preflight(config: AppConfig, *, require_llm: bool = True, require_whisper: bool = True) -> PreflightReport:
    """Verify every external dependency, raising :class:`DependencyError`."""

    report = PreflightReport()

    ffmpeg = FFmpeg(config.runtime.ffmpeg_path, config.runtime.ffprobe_path)
    info = ffmpeg.verify()
    report.ffmpeg_version = info["version"]
    report.details["ffmpeg_path"] = info["ffmpeg"]
    report.details["ffprobe_path"] = info["ffprobe"]
    log.info("FFmpeg ready: %s", report.ffmpeg_version)

    report.yt_dlp_version = check_yt_dlp()
    log.info("yt-dlp ready: %s", report.yt_dlp_version)

    if require_whisper:
        report.whisper_available = check_faster_whisper()
        log.info("faster-whisper ready (model: %s)", config.whisper.model)

    if require_llm:
        client = OllamaClient(config.ollama)
        health = client.verify()
        report.ollama_version = health.get("version", "")
        report.ollama_model = health.get("model", config.ollama.model)
        log.info(
            "Ollama ready at %s (version %s, model %s)",
            config.ollama.base_url,
            report.ollama_version or "unknown",
            report.ollama_model,
        )
    return report
