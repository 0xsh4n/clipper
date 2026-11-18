"""Safe FFmpeg / FFprobe wrappers.

All commands are executed as argument *lists* via :mod:`subprocess` with
``shell=False`` -- no string interpolation into a shell, ever. Nothing in this
module accepts data produced by the LLM except already-validated numbers.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .errors import DependencyError, RenderError
from .logger import get_logger

log = get_logger("media")


@dataclass
class MediaInfo:
    """Subset of ``ffprobe`` output the pipeline cares about."""

    path: Path
    duration: float
    width: int
    height: int
    fps: float
    has_audio: bool
    video_codec: str = ""
    audio_codec: str = ""

    @property
    def aspect(self) -> float:
        return (self.width / self.height) if self.height else 0.0


def which(binary: str) -> Optional[str]:
    """``shutil.which`` that also accepts an explicit path."""

    candidate = Path(binary)
    if candidate.is_file():
        return str(candidate)
    return shutil.which(binary)


def require_binary(binary: str, name: str, install_hint: str) -> str:
    resolved = which(binary)
    if not resolved:
        raise DependencyError(f"{name} was not found (looked for {binary!r}).", install_hint)
    return resolved


def run_command(
    args: Sequence[str],
    *,
    cwd: Optional[Path] = None,
    timeout: Optional[int] = None,
    description: str = "command",
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Run a command, capturing output, and raise a readable error on failure."""

    args = [str(a) for a in args]
    log.debug("exec: %s", " ".join(args))
    try:
        proc = subprocess.run(
            args,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
            shell=False,
            text=True,
            errors="replace",
        )
    except FileNotFoundError as exc:
        raise DependencyError(f"{args[0]} is not installed or not on PATH.", str(exc)) from exc
    except subprocess.TimeoutExpired as exc:
        raise RenderError(f"{description} timed out after {timeout}s.") from exc

    if check and proc.returncode != 0:
        tail = "\n".join((proc.stderr or "").strip().splitlines()[-12:])
        raise RenderError(
            f"{description} failed (exit code {proc.returncode}).",
            tail or "No error output captured.",
        )
    return proc


class FFmpeg:
    """Thin facade over the ffmpeg/ffprobe binaries."""

    def __init__(self, ffmpeg_path: str = "ffmpeg", ffprobe_path: str = "ffprobe") -> None:
        self.ffmpeg_path = ffmpeg_path
        self.ffprobe_path = ffprobe_path

    # -- availability ----------------------------------------------------- #
    def verify(self) -> Dict[str, str]:
        hint = (
            "Install FFmpeg and make sure ffmpeg/ffprobe are on PATH.\n"
            "     Windows : winget install Gyan.FFmpeg   (or download from https://www.gyan.dev/ffmpeg/builds/)\n"
            "     Debian  : sudo apt install ffmpeg\n"
            "     macOS   : brew install ffmpeg"
        )
        ffmpeg = require_binary(self.ffmpeg_path, "FFmpeg", hint)
        ffprobe = require_binary(self.ffprobe_path, "FFprobe", hint)
        version = run_command([ffmpeg, "-version"], description="ffmpeg -version").stdout
        first = version.splitlines()[0] if version else "unknown"
        return {"ffmpeg": ffmpeg, "ffprobe": ffprobe, "version": first}

    # -- inspection ------------------------------------------------------- #
    def probe(self, path: Path) -> MediaInfo:
        path = Path(path)
        if not path.is_file():
            raise RenderError(f"Media file not found: {path}")
        args = [
            self.ffprobe_path, "-v", "error",
            "-print_format", "json",
            "-show_format", "-show_streams",
            str(path),
        ]
        proc = run_command(args, description=f"ffprobe {path.name}")
        try:
            data: Dict[str, Any] = json.loads(proc.stdout or "{}")
        except json.JSONDecodeError as exc:
            raise RenderError(f"Could not parse ffprobe output for {path.name}.") from exc

        streams = data.get("streams") or []
        video = next((s for s in streams if s.get("codec_type") == "video"), None)
        audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

        duration = 0.0
        for source in (data.get("format") or {}, video or {}, audio or {}):
            try:
                duration = float(source.get("duration") or 0.0)
            except (TypeError, ValueError):
                duration = 0.0
            if duration > 0:
                break

        width = int(video.get("width") or 0) if video else 0
        height = int(video.get("height") or 0) if video else 0
        fps = 0.0
        if video:
            raw = video.get("avg_frame_rate") or video.get("r_frame_rate") or "0/0"
            try:
                num, _, den = str(raw).partition("/")
                fps = float(num) / float(den) if den and float(den) else float(num or 0)
            except (TypeError, ValueError, ZeroDivisionError):
                fps = 0.0

        return MediaInfo(
            path=path,
            duration=duration,
            width=width,
            height=height,
            fps=fps or 30.0,
            has_audio=audio is not None,
            video_codec=str((video or {}).get("codec_name") or ""),
            audio_codec=str((audio or {}).get("codec_name") or ""),
        )

    # -- execution -------------------------------------------------------- #
    def run(
        self,
        args: Sequence[str],
        *,
        description: str,
        cwd: Optional[Path] = None,
        timeout: Optional[int] = None,
    ) -> subprocess.CompletedProcess:
        full = [self.ffmpeg_path, "-hide_banner", "-nostdin", "-loglevel", "error", "-y", *args]
        return run_command(full, description=description, cwd=cwd, timeout=timeout)

    def extract_audio(
        self,
        source: Path,
        destination: Path,
        *,
        sample_rate: int = 16000,
        channels: int = 1,
    ) -> Path:
        """Extract mono PCM WAV audio (what faster-whisper likes best)."""

        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.run(
            [
                "-i", str(source),
                "-vn", "-sn", "-dn",
                "-ac", str(channels),
                "-ar", str(sample_rate),
                "-acodec", "pcm_s16le",
                str(destination),
            ],
            description=f"audio extraction from {Path(source).name}",
        )
        if not destination.is_file() or destination.stat().st_size == 0:
            raise RenderError(f"Audio extraction produced no data: {destination}")
        return destination

    def extract_frames(
        self,
        source: Path,
        destination_dir: Path,
        *,
        start: float,
        duration: float,
        count: int,
        width: int = 640,
    ) -> List[Path]:
        """Sample ``count`` evenly spaced JPEG frames (used for smart cropping)."""

        destination_dir = Path(destination_dir)
        destination_dir.mkdir(parents=True, exist_ok=True)
        count = max(1, int(count))
        fps = count / max(0.5, float(duration))
        pattern = "frame_%04d.jpg"
        self.run(
            [
                "-ss", f"{max(0.0, float(start)):.3f}",
                "-t", f"{max(0.1, float(duration)):.3f}",
                "-i", str(source),
                "-vf", f"fps={fps:.6f},scale={int(width)}:-2",
                "-frames:v", str(count),
                "-q:v", "3",
                pattern,
            ],
            description="frame sampling",
            cwd=destination_dir,
        )
        return sorted(destination_dir.glob("frame_*.jpg"))
