"""Small shared helpers: validation, sanitisation, time formatting."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import parse_qs, urlparse

from .errors import ClipperError

_YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtu.be",
    "www.youtu.be",
}

_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_UNSAFE_CHARS_RE = re.compile(r"[^\w.\- ]+", re.UNICODE)
_TRUTHY = {"1", "true", "yes", "y", "on"}
_FALSY = {"0", "false", "no", "n", "off", ""}


def validate_youtube_url(url: str) -> str:
    """Validate a YouTube URL and return it normalised.

    Accepts ``watch?v=``, ``youtu.be/``, ``/live/``, ``/shorts/`` and
    ``/embed/`` forms. Raises :class:`ClipperError` for anything else so we
    never hand an arbitrary string to a subprocess.
    """

    if not isinstance(url, str) or not url.strip():
        raise ClipperError("No URL provided.", "Pass a YouTube URL as the first argument.")

    url = url.strip()
    if "://" not in url:
        url = "https://" + url

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ClipperError(f"Unsupported URL scheme: {parsed.scheme!r}", "Use an http(s) URL.")

    host = (parsed.hostname or "").lower()
    if host not in _YOUTUBE_HOSTS:
        raise ClipperError(
            f"Not a YouTube URL: {url}",
            "Supported hosts: youtube.com, youtu.be (watch, live, shorts, embed).",
        )

    if extract_video_id(url) is None:
        raise ClipperError(
            f"Could not find a video id in: {url}",
            "Expected something like https://www.youtube.com/watch?v=XXXXXXXXXXX",
        )
    return url


def extract_video_id(url: str) -> Optional[str]:
    """Return the 11 character YouTube video id, or ``None``."""

    if not isinstance(url, str):
        return None
    parsed = urlparse(url if "://" in url else "https://" + url)
    host = (parsed.hostname or "").lower()
    path = parsed.path or ""

    if host in {"youtu.be", "www.youtu.be"}:
        candidate = path.lstrip("/").split("/")[0]
        return candidate if _VIDEO_ID_RE.match(candidate) else None

    if path == "/watch":
        values = parse_qs(parsed.query).get("v", [])
        if values and _VIDEO_ID_RE.match(values[0]):
            return values[0]
        return None

    parts = [p for p in path.split("/") if p]
    if len(parts) >= 2 and parts[0] in {"live", "shorts", "embed", "v"}:
        candidate = parts[1]
        return candidate if _VIDEO_ID_RE.match(candidate) else None
    return None


def sanitize_filename(name: str, max_length: int = 80, fallback: str = "clip") -> str:
    """Make ``name`` safe to use as a single path component.

    Strips directory separators, traversal sequences, control characters and
    Windows-reserved names. The result never contains ``/``, ``\\`` or ``..``.
    """

    if not isinstance(name, str):
        name = str(name)
    name = unicodedata.normalize("NFKC", name)
    name = name.replace("/", " ").replace("\\", " ")
    name = "".join(ch for ch in name if ch.isprintable())
    name = _UNSAFE_CHARS_RE.sub(" ", name)
    name = re.sub(r"\s+", " ", name).strip(" .")
    name = name.replace(" ", "_")
    name = re.sub(r"_+", "_", name).strip("._-")

    if not name:
        return fallback
    if len(name) > max_length:
        name = name[:max_length].rstrip("._-") or fallback

    reserved = {
        "CON", "PRN", "AUX", "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }
    if name.upper() in reserved:
        name = f"{name}_file"
    return name


def ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def unique_dir(parent: Path, name: str) -> Path:
    """Return ``parent/name``, suffixed with ``-2``, ``-3``... if taken."""

    parent = Path(parent)
    candidate = parent / name
    counter = 2
    while candidate.exists():
        candidate = parent / f"{name}-{counter}"
        counter += 1
    return candidate


def format_timestamp(seconds: float) -> str:
    """``3738.4`` -> ``01:02:18``."""

    seconds = max(0.0, float(seconds))
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_clock(seconds: float) -> str:
    """``812.4`` -> ``13:32`` (short form used in logs)."""

    seconds = max(0.0, float(seconds))
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def srt_timestamp(seconds: float) -> str:
    """``1.5`` -> ``00:00:01,500``."""

    seconds = max(0.0, float(seconds))
    millis = int(round(seconds * 1000))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def vtt_timestamp(seconds: float) -> str:
    return srt_timestamp(seconds).replace(",", ".")


def ass_timestamp(seconds: float) -> str:
    """``1.5`` -> ``0:00:01.50`` (ASS uses centiseconds)."""

    seconds = max(0.0, float(seconds))
    centis = int(round(seconds * 100))
    hours, centis = divmod(centis, 360_000)
    minutes, centis = divmod(centis, 6_000)
    secs, centis = divmod(centis, 100)
    return f"{hours:d}:{minutes:02d}:{secs:02d}.{centis:02d}"


def parse_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in _TRUTHY:
        return True
    if text in _FALSY:
        return False
    raise ClipperError(f"Cannot interpret {value!r} as a boolean.")


def clamp(value: float, low: float, high: float) -> float:
    if high < low:
        low, high = high, low
    return max(low, min(high, value))


def write_json(path: Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    return path


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def human_duration(seconds: float) -> str:
    return format_timestamp(seconds)


def dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out
