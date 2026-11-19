"""Layered configuration.

Precedence (highest wins)::

    CLI overrides  >  environment / .env  >  config.yaml  >  built-in defaults

Everything the pipeline needs is exposed as typed dataclasses so the rest of
the code never reaches into raw dictionaries.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import yaml

from .errors import ConfigError
from .utils import parse_bool

DEFAULTS: Dict[str, Any] = {
    "ollama": {
        "base_url": "http://localhost:11434",
        "model": "qwen3:4b",
        "timeout": 120,
        "temperature": 0.2,
        "num_ctx": 8192,
        "max_retries": 3,
        "candidates_per_request": 12,
    },
    "whisper": {
        "model": "base",
        "device": "auto",          # auto | cuda | cpu
        "compute_type": "auto",    # auto | float16 | int8_float16 | int8 | float32
        "language": None,          # None -> auto detect
        "beam_size": 1,
        "vad_filter": True,
        "download_root": None,
    },
    "clip": {
        "target_duration": 30.0,
        "min_duration": 25.0,
        "max_duration": 35.0,
        "clips": 5,
        "auto_count": True,
        "max_candidates": 36,
        "min_gap_between_clips": 15.0,
        "lead_in": 0.25,
        "tail_out": 0.35,
        "skip_intro_seconds": 0.0,
        "skip_outro_seconds": 0.0,
    },
    "scoring": {
        "hook": 0.23,
        "interestingness": 0.21,
        "information_value": 0.08,
        "emotional_impact": 0.16,
        "shareability": 0.16,
        "clarity": 0.06,
        "standalone_context": 0.10,
    },
    "subtitles": {
        "enabled": True,
        "burn_in": True,
        "font_name": "DejaVu Sans",
        "font_size": 48,
        "max_lines": 2,
        "max_chars_per_line": 22,
        "position": "bottom",      # bottom | center | top
        "margin_ratio": 0.14,
        "primary_color": "#FFFFFF",
        "outline_color": "#000000",
        "back_color": "#000000",
        "outline": 3.0,
        "shadow": 1.0,
        "bold": True,
        "uppercase": False,
        "max_cue_duration": 3.2,
    },
    "video": {
        "formats": ["16:9", "9:16"],
        "landscape_resolution": "1920x1080",
        "vertical_resolution": "1080x1920",
        "video_codec": "libx264",
        "audio_codec": "aac",
        "crf": 20,
        "preset": "veryfast",
        "audio_bitrate": "192k",
        "fps": None,               # None -> keep source fps
        "smart_crop": True,
        "smart_crop_samples": 12,
        "loudness_normalize": True,
    },
    "download": {
        "format": "bestvideo[height<=1080]+bestaudio/best[height<=1080]/best",
        "max_duration": 21600,     # 6 h safety limit
        "retries": 3,
        "cookies_from_browser": None,
        "proxy": None,
    },
    "paths": {
        "output_dir": "./output",
        "workspace_dir": "./workspace",
    },
    "runtime": {
        "keep_temp_files": False,
        "use_cache": True,
        "log_level": "INFO",
        "ffmpeg_path": "ffmpeg",
        "ffprobe_path": "ffprobe",
    },
}

#: ``ENV_VAR -> dotted config path``
ENV_MAP: Dict[str, str] = {
    "OLLAMA_BASE_URL": "ollama.base_url",
    "OLLAMA_MODEL": "ollama.model",
    "OLLAMA_TIMEOUT": "ollama.timeout",
    "OLLAMA_TEMPERATURE": "ollama.temperature",
    "OLLAMA_NUM_CTX": "ollama.num_ctx",
    "WHISPER_MODEL": "whisper.model",
    "WHISPER_DEVICE": "whisper.device",
    "WHISPER_COMPUTE_TYPE": "whisper.compute_type",
    "WHISPER_LANGUAGE": "whisper.language",
    "TARGET_CLIP_DURATION": "clip.target_duration",
    "MIN_CLIP_DURATION": "clip.min_duration",
    "MAX_CLIP_DURATION": "clip.max_duration",
    "CLIPS_PER_VIDEO": "clip.clips",
    "AUTO_CLIP_COUNT": "clip.auto_count",
    "OUTPUT_DIR": "paths.output_dir",
    "WORKSPACE_DIR": "paths.workspace_dir",
    "VIDEO_FORMATS": "video.formats",
    "OUTPUT_FORMATS": "video.formats",
    "SMART_CROP": "video.smart_crop",
    "SUBTITLE_FONT_SIZE": "subtitles.font_size",
    "SUBTITLES_ENABLED": "subtitles.enabled",
    "BURN_IN_SUBTITLES": "subtitles.burn_in",
    "KEEP_TEMP_FILES": "runtime.keep_temp_files",
    "USE_CACHE": "runtime.use_cache",
    "LOG_LEVEL": "runtime.log_level",
    "FFMPEG_PATH": "runtime.ffmpeg_path",
    "FFPROBE_PATH": "runtime.ffprobe_path",
    "YTDLP_FORMAT": "download.format",
    "YTDLP_COOKIES_FROM_BROWSER": "download.cookies_from_browser",
    "HTTP_PROXY_URL": "download.proxy",
}

VALID_FORMATS = ("16:9", "9:16")
VALID_WHISPER_MODELS = (
    "tiny", "tiny.en", "base", "base.en", "small", "small.en",
    "medium", "medium.en", "large-v1", "large-v2", "large-v3", "distil-large-v3",
    "turbo", "large-v3-turbo",
)


# --------------------------------------------------------------------------- #
# dataclasses
# --------------------------------------------------------------------------- #
@dataclass
class OllamaConfig:
    base_url: str = "http://localhost:11434"
    model: str = "qwen3:4b"
    timeout: int = 120
    temperature: float = 0.2
    num_ctx: int = 8192
    max_retries: int = 3
    candidates_per_request: int = 12


@dataclass
class WhisperConfig:
    model: str = "base"
    device: str = "auto"
    compute_type: str = "auto"
    language: Optional[str] = None
    beam_size: int = 1
    vad_filter: bool = True
    download_root: Optional[str] = None


@dataclass
class ClipConfig:
    target_duration: float = 30.0
    min_duration: float = 25.0
    max_duration: float = 35.0
    clips: int = 5
    auto_count: bool = True
    max_candidates: int = 36
    min_gap_between_clips: float = 15.0
    lead_in: float = 0.25
    tail_out: float = 0.35
    skip_intro_seconds: float = 0.0
    skip_outro_seconds: float = 0.0


@dataclass
class ScoringWeights:
    hook: float = 0.23
    interestingness: float = 0.21
    information_value: float = 0.08
    emotional_impact: float = 0.16
    shareability: float = 0.16
    clarity: float = 0.06
    standalone_context: float = 0.10

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class SubtitleConfig:
    enabled: bool = True
    burn_in: bool = True
    font_name: str = "DejaVu Sans"
    font_size: int = 48
    max_lines: int = 2
    max_chars_per_line: int = 22
    position: str = "bottom"
    margin_ratio: float = 0.14
    primary_color: str = "#FFFFFF"
    outline_color: str = "#000000"
    back_color: str = "#000000"
    outline: float = 3.0
    shadow: float = 1.0
    bold: bool = True
    uppercase: bool = False
    max_cue_duration: float = 3.2


@dataclass
class VideoConfig:
    formats: List[str] = field(default_factory=lambda: ["16:9", "9:16"])
    landscape_resolution: str = "1920x1080"
    vertical_resolution: str = "1080x1920"
    video_codec: str = "libx264"
    audio_codec: str = "aac"
    crf: int = 20
    preset: str = "veryfast"
    audio_bitrate: str = "192k"
    fps: Optional[float] = None
    smart_crop: bool = True
    smart_crop_samples: int = 12
    loudness_normalize: bool = True

    def resolution_for(self, fmt: str) -> tuple[int, int]:
        raw = self.vertical_resolution if fmt == "9:16" else self.landscape_resolution
        return parse_resolution(raw)


@dataclass
class DownloadConfig:
    format: str = DEFAULTS["download"]["format"]
    max_duration: int = 21600
    retries: int = 3
    cookies_from_browser: Optional[str] = None
    proxy: Optional[str] = None


@dataclass
class PathsConfig:
    output_dir: Path = Path("./output")
    workspace_dir: Path = Path("./workspace")


@dataclass
class RuntimeConfig:
    keep_temp_files: bool = False
    use_cache: bool = True
    log_level: str = "INFO"
    ffmpeg_path: str = "ffmpeg"
    ffprobe_path: str = "ffprobe"


@dataclass
class AppConfig:
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    clip: ClipConfig = field(default_factory=ClipConfig)
    scoring: ScoringWeights = field(default_factory=ScoringWeights)
    subtitles: SubtitleConfig = field(default_factory=SubtitleConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    download: DownloadConfig = field(default_factory=DownloadConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    #: Where the prompt templates live.
    prompts_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "prompts")

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "ollama": asdict(self.ollama),
            "whisper": asdict(self.whisper),
            "clip": asdict(self.clip),
            "scoring": asdict(self.scoring),
            "subtitles": asdict(self.subtitles),
            "video": asdict(self.video),
            "download": asdict(self.download),
            "paths": {
                "output_dir": str(self.paths.output_dir),
                "workspace_dir": str(self.paths.workspace_dir),
            },
            "runtime": asdict(self.runtime),
        }
        return data


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def parse_resolution(value: str) -> tuple[int, int]:
    """``"1080x1920"`` -> ``(1080, 1920)``."""

    text = str(value).lower().replace(" ", "").replace("*", "x").replace(":", "x")
    parts = text.split("x")
    if len(parts) != 2:
        raise ConfigError(f"Invalid resolution: {value!r}", "Use WIDTHxHEIGHT, e.g. 1080x1920.")
    try:
        width, height = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise ConfigError(f"Invalid resolution: {value!r}", "Use WIDTHxHEIGHT, e.g. 1080x1920.") from exc
    if width <= 0 or height <= 0:
        raise ConfigError(f"Invalid resolution: {value!r}")
    # H.264 needs even dimensions.
    return width - (width % 2), height - (height % 2)


def parse_formats(value: Any) -> List[str]:
    """Accept ``"both"``, ``"9:16"``, ``"16:9,9:16"`` or a list."""

    if value is None:
        return list(DEFAULTS["video"]["formats"])
    if isinstance(value, str):
        items = [p.strip() for p in value.replace(";", ",").split(",") if p.strip()]
    elif isinstance(value, (list, tuple)):
        items = [str(p).strip() for p in value if str(p).strip()]
    else:
        raise ConfigError(f"Invalid video formats: {value!r}")

    out: List[str] = []
    for item in items:
        low = item.lower()
        if low in {"both", "all"}:
            out.extend(VALID_FORMATS)
            continue
        alias = {
            "16x9": "16:9", "landscape": "16:9", "youtube": "16:9", "horizontal": "16:9",
            "9x16": "9:16", "vertical": "9:16", "shorts": "9:16", "reels": "9:16", "tiktok": "9:16",
        }.get(low, item)
        if alias not in VALID_FORMATS:
            raise ConfigError(
                f"Unsupported output format: {item!r}",
                "Valid values: 16:9, 9:16, both.",
            )
        out.append(alias)
    if not out:
        raise ConfigError("No output formats selected.", "Use --format 16:9, 9:16 or both.")
    # Preserve order, drop duplicates.
    return [f for i, f in enumerate(out) if f not in out[:i]]


def load_dotenv(path: Path) -> Dict[str, str]:
    """Minimal ``.env`` parser (no external dependency).

    Supports ``KEY=value``, ``export KEY=value``, ``#`` comments and quoted
    values. Existing process environment variables win, so ``OLLAMA_MODEL=x
    python main.py ...`` keeps working.
    """

    values: Dict[str, str] = {}
    path = Path(path)
    if not path.is_file():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].strip()
        if key:
            values[key] = value
    return values


def _set_dotted(target: Dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = target
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):  # pragma: no cover - defensive
            raise ConfigError(f"Cannot set {dotted}: {part} is not a section.")
    node[parts[-1]] = value


def _deep_merge(base: Dict[str, Any], overlay: Mapping[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _coerce(section: str, key: str, value: Any) -> Any:
    """Coerce a raw string (env/CLI) into the type of the default value."""

    if section == "video" and key == "formats":
        return parse_formats(value)
    default = DEFAULTS.get(section, {}).get(key)
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null", "auto-detect"}:
        if default is None:
            return None
    if isinstance(default, bool):
        return parse_bool(value, default)
    if isinstance(default, int) and not isinstance(default, bool):
        try:
            return int(float(value))
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{section}.{key} must be an integer (got {value!r}).") from exc
    if isinstance(default, float):
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{section}.{key} must be a number (got {value!r}).") from exc
    if isinstance(default, list):
        if isinstance(value, str):
            return [p.strip() for p in value.split(",") if p.strip()]
        return list(value)
    return value


def _validate(data: Dict[str, Any]) -> None:
    clip = data["clip"]
    if clip["min_duration"] <= 0:
        raise ConfigError("clip.min_duration must be > 0.")
    if clip["max_duration"] < clip["min_duration"]:
        raise ConfigError(
            f"clip.max_duration ({clip['max_duration']}) is below clip.min_duration "
            f"({clip['min_duration']}).",
            "Fix MIN_CLIP_DURATION / MAX_CLIP_DURATION.",
        )
    if not (clip["min_duration"] <= clip["target_duration"] <= clip["max_duration"]):
        raise ConfigError(
            f"clip.target_duration ({clip['target_duration']}) must sit between "
            f"min ({clip['min_duration']}) and max ({clip['max_duration']}).",
            "Pass --duration together with --min/--max, or edit config.yaml.",
        )
    if clip["clips"] < 1:
        raise ConfigError("clip.clips must be >= 1.", "Use --clips 1 or higher.")
    if int(data["whisper"]["beam_size"]) < 1:
        raise ConfigError("whisper.beam_size must be >= 1.")

    whisper_model = str(data["whisper"]["model"])
    if whisper_model not in VALID_WHISPER_MODELS and not Path(whisper_model).exists():
        raise ConfigError(
            f"Unknown Whisper model: {whisper_model!r}",
            "Valid: " + ", ".join(VALID_WHISPER_MODELS) + " (or a local model path).",
        )
    if data["whisper"]["device"] not in {"auto", "cpu", "cuda"}:
        raise ConfigError("whisper.device must be auto, cpu or cuda.")

    if data["subtitles"]["position"] not in {"bottom", "center", "top"}:
        raise ConfigError("subtitles.position must be bottom, center or top.")
    if int(data["subtitles"]["max_lines"]) < 1:
        raise ConfigError("subtitles.max_lines must be >= 1.")
    if int(data["subtitles"]["font_size"]) < 8:
        raise ConfigError("subtitles.font_size looks too small (minimum 8).")

    data["video"]["formats"] = parse_formats(data["video"]["formats"])
    parse_resolution(data["video"]["landscape_resolution"])
    parse_resolution(data["video"]["vertical_resolution"])

    weights = data["scoring"]
    total = sum(float(v) for v in weights.values())
    if total <= 0:
        raise ConfigError("scoring weights must sum to a positive number.")

    if not str(data["ollama"]["base_url"]).startswith(("http://", "https://")):
        raise ConfigError(
            f"ollama.base_url must be an http(s) URL (got {data['ollama']['base_url']!r}).",
            "Default: http://localhost:11434",
        )
    if not str(data["ollama"]["model"]).strip():
        raise ConfigError("ollama.model must not be empty.", "Try OLLAMA_MODEL=qwen3:4b")


def load_config(
    config_path: Optional[Path] = None,
    env_path: Optional[Path] = None,
    overrides: Optional[Mapping[str, Any]] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> AppConfig:
    """Build an :class:`AppConfig` from every configuration layer."""

    data = copy.deepcopy(DEFAULTS)

    # 1. config.yaml
    if config_path is not None:
        config_path = Path(config_path)
        if config_path.is_file():
            try:
                loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            except yaml.YAMLError as exc:
                raise ConfigError(f"Could not parse {config_path}: {exc}") from exc
            if not isinstance(loaded, dict):
                raise ConfigError(f"{config_path} must contain a YAML mapping.")
            data = _deep_merge(data, loaded)

    # 2. .env then real environment (real environment wins)
    env: Dict[str, str] = {}
    if env_path is not None:
        env.update(load_dotenv(Path(env_path)))
    source_environ = os.environ if environ is None else environ
    for key in ENV_MAP:
        if key in source_environ:
            env[key] = source_environ[key]
    for key, dotted in ENV_MAP.items():
        if key in env:
            section, _, leaf = dotted.partition(".")
            _set_dotted(data, dotted, _coerce(section, leaf, env[key]))

    # 3. CLI overrides (already typed)
    for dotted, value in (overrides or {}).items():
        if value is None:
            continue
        section, _, leaf = dotted.partition(".")
        _set_dotted(data, dotted, _coerce(section, leaf, value))

    _validate(data)

    paths = data["paths"]
    cfg = AppConfig(
        ollama=OllamaConfig(**data["ollama"]),
        whisper=WhisperConfig(**data["whisper"]),
        clip=ClipConfig(**data["clip"]),
        scoring=ScoringWeights(**data["scoring"]),
        subtitles=SubtitleConfig(**data["subtitles"]),
        video=VideoConfig(**data["video"]),
        download=DownloadConfig(**data["download"]),
        paths=PathsConfig(
            output_dir=Path(str(paths["output_dir"])).expanduser(),
            workspace_dir=Path(str(paths["workspace_dir"])).expanduser(),
        ),
        runtime=RuntimeConfig(**data["runtime"]),
    )
    return cfg
