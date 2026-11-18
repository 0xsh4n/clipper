"""YouTube download stage (yt-dlp).

Handles normal videos, live *recordings* (VODs) and long uploads. The video is
downloaded exactly once per workspace: a cached ``source/source.json`` marker
lets re-runs skip straight to transcription.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional

from .config import AppConfig
from .errors import DownloadError
from .logger import get_logger
from .media import FFmpeg
from .utils import extract_video_id, format_timestamp, read_json, sanitize_filename, write_json

log = get_logger("downloader")

_MEDIA_SUFFIXES = {".mp4", ".mkv", ".webm", ".mov", ".m4v", ".flv", ".ts"}


@dataclass
class SourceVideo:
    """A downloaded source video plus the metadata we keep about it."""

    path: Path
    video_id: str
    title: str
    duration: float
    url: str
    uploader: str = ""
    upload_date: str = ""
    channel_url: str = ""
    was_live: bool = False
    is_live: bool = False
    width: int = 0
    height: int = 0
    fps: float = 0.0
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["path"] = str(self.path)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SourceVideo":
        payload = dict(data)
        payload["path"] = Path(payload["path"])
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in payload.items() if k in known})


class YouTubeDownloader:
    """Downloads a YouTube URL into ``<workspace>/source``."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.ffmpeg = FFmpeg(config.runtime.ffmpeg_path, config.runtime.ffprobe_path)

    # -- public API ------------------------------------------------------- #
    def probe_metadata(self, url: str) -> Dict[str, Any]:
        """Fetch metadata without downloading (used for job naming + checks)."""

        import yt_dlp

        options = self._base_options()
        options.update({"skip_download": True})
        try:
            with yt_dlp.YoutubeDL(options) as ydl:
                info = ydl.extract_info(url, download=False)
        except Exception as exc:  # yt-dlp raises many subclasses
            raise DownloadError(
                f"Could not read video information: {exc}",
                "Check the URL, your network, and that yt-dlp is up to date "
                "(pip install -U yt-dlp).",
            ) from exc
        if info is None:
            raise DownloadError("yt-dlp returned no information for this URL.")
        if info.get("_type") == "playlist":
            entries = [e for e in (info.get("entries") or []) if e]
            if not entries:
                raise DownloadError("This URL is an empty playlist.")
            log.warning("URL points to a playlist; using the first entry.")
            info = entries[0]
        return info

    def download(self, url: str, workspace: Path, *, use_cache: bool = True) -> SourceVideo:
        source_dir = Path(workspace) / "source"
        source_dir.mkdir(parents=True, exist_ok=True)
        marker = source_dir / "source.json"

        if use_cache and marker.is_file():
            cached = self._load_cached(marker)
            if cached is not None:
                log.info("Reusing cached download: %s", cached.path.name)
                log.info("Video duration: %s", format_timestamp(cached.duration))
                return cached

        log.info("Downloading video")
        info = self.probe_metadata(url)
        self._guard_live_and_length(info)

        video_id = info.get("id") or extract_video_id(url) or "video"
        title = str(info.get("title") or video_id)
        stem = sanitize_filename(f"{sanitize_filename(title, max_length=60)}-{video_id}")

        import yt_dlp

        options = self._base_options()
        options.update(
            {
                "format": self.config.download.format,
                "outtmpl": {"default": str(source_dir / f"{stem}.%(ext)s")},
                "merge_output_format": "mp4",
                "noprogress": False,
                "progress_hooks": [self._progress_hook],
                "retries": int(self.config.download.retries),
                "fragment_retries": int(self.config.download.retries),
                "concurrent_fragment_downloads": 4,
                "overwrites": False,
                "continuedl": True,
            }
        )

        try:
            with yt_dlp.YoutubeDL(options) as ydl:
                result = ydl.extract_info(url, download=True)
        except Exception as exc:
            raise DownloadError(
                f"Download failed: {exc}",
                "Common causes: age/region restriction (try "
                "YTDLP_COOKIES_FROM_BROWSER=chrome), an outdated yt-dlp, or no network.",
            ) from exc

        if result and result.get("_type") == "playlist":
            entries = [e for e in (result.get("entries") or []) if e]
            result = entries[0] if entries else result
        info = result or info

        path = self._resolve_downloaded_path(info, source_dir, stem)
        probed = self.ffmpeg.probe(path)
        duration = float(info.get("duration") or 0.0) or probed.duration
        if duration <= 0:
            raise DownloadError(
                "Could not determine the video duration.",
                "The file may be an incomplete live stream capture.",
            )

        source = SourceVideo(
            path=path,
            video_id=str(video_id),
            title=title,
            duration=float(duration),
            url=url,
            uploader=str(info.get("uploader") or info.get("channel") or ""),
            upload_date=str(info.get("upload_date") or ""),
            channel_url=str(info.get("channel_url") or info.get("uploader_url") or ""),
            was_live=bool(info.get("was_live")),
            is_live=bool(info.get("is_live")),
            width=int(probed.width or info.get("width") or 0),
            height=int(probed.height or info.get("height") or 0),
            fps=float(probed.fps or info.get("fps") or 0.0),
            description=str(info.get("description") or "")[:4000],
        )
        write_json(marker, source.to_dict())
        log.info("Downloaded: %s", path.name)
        log.info("Video duration: %s", format_timestamp(source.duration))
        return source

    # -- internals -------------------------------------------------------- #
    def _base_options(self) -> Dict[str, Any]:
        options: Dict[str, Any] = {
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "ignoreerrors": False,
            "nocheckcertificate": False,
            "geo_bypass": True,
            "extract_flat": False,
            "logger": _YtdlpLogBridge(),
        }
        if self.config.download.proxy:
            options["proxy"] = self.config.download.proxy
        browser = self.config.download.cookies_from_browser
        if browser:
            options["cookiesfrombrowser"] = (str(browser),)
        return options

    def _guard_live_and_length(self, info: Dict[str, Any]) -> None:
        if info.get("is_live"):
            raise DownloadError(
                "This URL is a currently *live* stream.",
                "Wait for the stream to finish (or use a /live/ VOD URL) - clipping needs "
                "a complete recording.",
            )
        duration = float(info.get("duration") or 0.0)
        limit = float(self.config.download.max_duration)
        if duration and limit and duration > limit:
            raise DownloadError(
                f"Video is {format_timestamp(duration)} long, which exceeds the configured "
                f"limit of {format_timestamp(limit)}.",
                "Raise download.max_duration in config.yaml if this is intentional.",
            )

    def _resolve_downloaded_path(self, info: Dict[str, Any], source_dir: Path, stem: str) -> Path:
        candidates = []
        requested = info.get("requested_downloads") or []
        for entry in requested:
            for key in ("filepath", "_filename", "filename"):
                if entry.get(key):
                    candidates.append(Path(entry[key]))
        for key in ("filepath", "_filename"):
            if info.get(key):
                candidates.append(Path(info[key]))

        for candidate in candidates:
            if candidate.is_file():
                return candidate
            sibling = source_dir / candidate.name
            if sibling.is_file():
                return sibling

        matches = [
            p for p in source_dir.glob(f"{stem}.*")
            if p.is_file() and p.suffix.lower() in _MEDIA_SUFFIXES
        ]
        if not matches:
            matches = [
                p for p in source_dir.iterdir()
                if p.is_file() and p.suffix.lower() in _MEDIA_SUFFIXES
            ]
        if not matches:
            raise DownloadError(
                "The download finished but no media file was found.",
                f"Looked in {source_dir}",
            )
        return max(matches, key=lambda p: p.stat().st_size)

    def _load_cached(self, marker: Path) -> Optional[SourceVideo]:
        try:
            cached = SourceVideo.from_dict(read_json(marker))
        except Exception as exc:
            log.debug("Ignoring unreadable download cache: %s", exc)
            return None
        if not cached.path.is_file():
            # Workspace may have moved; try the same filename next to the marker.
            relocated = marker.parent / cached.path.name
            if not relocated.is_file():
                return None
            cached.path = relocated
        if cached.duration <= 0:
            return None
        return cached

    def _progress_hook(self, status: Dict[str, Any]) -> None:
        if status.get("status") != "finished":
            return
        name = Path(str(status.get("filename") or "")).name
        log.info("Download finished: %s", name or "stream")


class _YtdlpLogBridge:
    """Routes yt-dlp chatter into our logger at DEBUG level."""

    def debug(self, msg: str) -> None:
        log.debug("yt-dlp: %s", str(msg).strip())

    def info(self, msg: str) -> None:
        log.debug("yt-dlp: %s", str(msg).strip())

    def warning(self, msg: str) -> None:
        log.debug("yt-dlp warning: %s", str(msg).strip())

    def error(self, msg: str) -> None:
        log.error("yt-dlp: %s", str(msg).strip())
