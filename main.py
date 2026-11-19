#!/usr/bin/env python3
"""Local AI video clipping agent - command line entry point.

    python main.py "https://www.youtube.com/watch?v=XXXXXXXXXXX"

Downloads a YouTube video or live recording, transcribes it, asks Qwen 2.5
(through a local Ollama) which ~30 second moment is strongest, then renders
captioned 16:9 and 9:16 versions into ./output.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from app import __version__
from app.config import AppConfig, load_config, parse_formats
from app.errors import ClipperError
from app.logger import get_logger, setup_logging
from app.pipeline import JobResult, Pipeline
from app.preflight import run_preflight
from app.utils import format_clock, format_timestamp

ROOT = Path(__file__).resolve().parent
log = get_logger()

EPILOG = """
examples:
  python main.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
  python main.py URL --duration 30
  python main.py URL --format 9:16
  python main.py URL --format both --clips 3
  python main.py URL --model qwen2.5:14b --whisper medium
  python main.py URL --output ./output --keep-temp --debug
  python main.py --check                      # verify ffmpeg / ollama / models only

requirements:
  ollama serve            (and: ollama pull qwen3:4b)
  ffmpeg + ffprobe on PATH
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Find and render the best ~30 second clip from a YouTube video, locally.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("url", nargs="?", help="YouTube video, live recording or shorts URL")

    clip_group = parser.add_argument_group("clip selection")
    clip_group.add_argument("--duration", type=float, metavar="SEC", help="target clip duration (default 30)")
    clip_group.add_argument("--min-duration", type=float, metavar="SEC", help="minimum clip duration (default 25)")
    clip_group.add_argument("--max-duration", type=float, metavar="SEC", help="maximum clip duration (default 35)")
    clip_group.add_argument("--clips", type=int, metavar="N", help="exact number of clips (overrides automatic count)")
    clip_group.add_argument(
        "--max-candidates", type=int, metavar="N",
        help="how many candidate windows to score with the LLM (default 36)",
    )
    clip_group.add_argument(
        "--skip-intro", type=float, metavar="SEC",
        help="ignore the first N seconds of the video when searching",
    )
    clip_group.add_argument(
        "--skip-outro", type=float, metavar="SEC",
        help="ignore the last N seconds of the video when searching",
    )

    output_group = parser.add_argument_group("output")
    output_group.add_argument(
        "--format", dest="formats", metavar="FMT",
        help="output formats: 16:9, 9:16 or both (default: both)",
    )
    output_group.add_argument("--output", metavar="DIR", help="output directory (default ./output)")
    output_group.add_argument("--workspace", metavar="DIR", help="working directory (default ./workspace)")
    output_group.add_argument("--no-subtitles", action="store_true", help="do not generate or burn captions")
    output_group.add_argument("--no-burn-in", action="store_true", help="write .srt/.vtt but do not burn captions in")
    output_group.add_argument("--font-size", type=int, metavar="PX", help="caption font size (default 48)")
    output_group.add_argument(
        "--subtitle-position", choices=("bottom", "center", "top"), help="caption placement (default bottom)",
    )
    output_group.add_argument("--no-smart-crop", action="store_true", help="always centre-crop the 9:16 version")

    model_group = parser.add_argument_group("models")
    model_group.add_argument("--model", metavar="NAME", help="Ollama model (default qwen3:4b)")
    model_group.add_argument("--ollama-url", metavar="URL", help="Ollama base URL (default http://localhost:11434)")
    model_group.add_argument(
        "--whisper", metavar="NAME",
        help="Whisper model: tiny, base, small, medium, large-v3 (default base)",
    )
    model_group.add_argument("--device", choices=("auto", "cpu", "cuda"), help="Whisper device (default auto)")
    model_group.add_argument("--beam-size", type=int, metavar="N", help="Whisper beam size (1 is faster, may slightly reduce accuracy)")
    model_group.add_argument("--language", metavar="CODE", help="force a transcription language, e.g. en")

    misc = parser.add_argument_group("misc")
    misc.add_argument("--config", metavar="FILE", default=str(ROOT / "config.yaml"), help="path to config.yaml")
    misc.add_argument("--env", metavar="FILE", default=str(ROOT / ".env"), help="path to a .env file")
    misc.add_argument("--keep-temp", action="store_true", help="keep workspace intermediates (debugging)")
    misc.add_argument("--no-cache", action="store_true", help="ignore cached download/transcript/analysis")
    misc.add_argument("--debug", action="store_true", help="verbose logging")
    misc.add_argument("--quiet", action="store_true", help="only warnings and errors")
    misc.add_argument("--check", action="store_true", help="run dependency checks and exit")
    misc.add_argument("--json", dest="json_summary", action="store_true", help="print the result summary as JSON")
    misc.add_argument("--version", action="version", version=f"ai-video-clipper {__version__}")
    return parser


def overrides_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    """Map CLI flags onto dotted config paths (``None`` values are ignored)."""

    overrides: Dict[str, Any] = {
        "clip.target_duration": args.duration,
        "clip.min_duration": args.min_duration,
        "clip.max_duration": args.max_duration,
        "clip.clips": args.clips,
        "clip.max_candidates": args.max_candidates,
        "clip.skip_intro_seconds": args.skip_intro,
        "clip.skip_outro_seconds": args.skip_outro,
        "paths.output_dir": args.output,
        "paths.workspace_dir": args.workspace,
        "ollama.model": args.model,
        "ollama.base_url": args.ollama_url,
        "whisper.model": args.whisper,
        "whisper.device": args.device,
        "whisper.beam_size": args.beam_size,
        "whisper.language": args.language,
        "subtitles.font_size": args.font_size,
        "subtitles.position": args.subtitle_position,
    }
    if args.formats:
        overrides["video.formats"] = parse_formats(args.formats)
    if args.no_subtitles:
        overrides["subtitles.enabled"] = False
    if args.no_burn_in:
        overrides["subtitles.burn_in"] = False
    if args.no_smart_crop:
        overrides["video.smart_crop"] = False
    if args.keep_temp:
        overrides["runtime.keep_temp_files"] = True
    if args.no_cache:
        overrides["runtime.use_cache"] = False
    if args.debug:
        overrides["runtime.log_level"] = "DEBUG"
    elif args.quiet:
        overrides["runtime.log_level"] = "WARNING"

    # A target duration on its own should widen the window around it.
    if args.duration and not args.min_duration and not args.max_duration:
        overrides["clip.min_duration"] = max(3.0, float(args.duration) - 5.0)
        overrides["clip.max_duration"] = float(args.duration) + 5.0
    return {k: v for k, v in overrides.items() if v is not None}


def _auto_clip_count(config: AppConfig, args: argparse.Namespace) -> Optional[int]:
    return max(1, int(args.clips)) if args.clips is not None else None


def summarise(result: JobResult) -> str:
    lines: List[str] = []
    lines.append("")
    lines.append("=" * 68)
    lines.append(f"  Source : {result.source.title}")
    lines.append(f"  Length : {format_timestamp(result.source.duration)}")
    lines.append(f"  Output : {result.output_root}")
    lines.append(f"  Took   : {format_timestamp(result.elapsed_seconds)}")
    for artifacts in result.clips:
        clip = artifacts.clip
        lines.append("-" * 68)
        lines.append(f"  Clip {clip.index}: {clip.label}  ({clip.duration:.1f}s, score {clip.score:.1f}/10)")
        if artifacts.metadata.get("title"):
            lines.append(f"    Title    : {artifacts.metadata['title']}")
        if clip.hook:
            lines.append(f"    Hook     : {clip.hook}")
        if clip.reason:
            lines.append(f"    Why      : {_wrap(clip.reason, 58, indent=15)}")
        hashtags = artifacts.metadata.get("hashtags") or []
        if hashtags:
            lines.append(f"    Hashtags : {' '.join(hashtags[:8])}")
        for render in artifacts.renders:
            lines.append(
                f"    {render.format:<5}    : {render.path.name} ({render.width}x{render.height})"
            )
    lines.append("=" * 68)
    return "\n".join(lines)


def _wrap(text: str, width: int, indent: int = 0) -> str:
    import textwrap

    wrapped = textwrap.wrap(text, width=width) or [""]
    pad = " " * indent
    return ("\n" + pad).join(wrapped)


def result_to_json(result: JobResult) -> Dict[str, Any]:
    return {
        "job_id": result.job_id,
        "url": result.url,
        "source_title": result.source.title,
        "source_duration": round(result.source.duration, 2),
        "output_root": str(result.output_root),
        "elapsed_seconds": round(result.elapsed_seconds, 2),
        "clips": [
            {
                "index": art.clip.index,
                "start": round(art.clip.start, 2),
                "end": round(art.clip.end, 2),
                "duration": round(art.clip.duration, 2),
                "score": round(art.clip.score, 2),
                "title": art.metadata.get("title", art.clip.title),
                "hook": art.clip.hook,
                "reason": art.clip.reason,
                "output_dir": str(art.output_dir),
                "files": [str(r.path) for r in art.renders],
            }
            for art in result.clips
        ],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.url and not args.check:
        parser.print_help()
        return 2

    try:
        config = load_config(
            config_path=Path(args.config) if args.config else None,
            env_path=Path(args.env) if args.env else None,
            overrides=overrides_from_args(args),
        )
    except ClipperError as exc:
        setup_logging("INFO")
        log.error("Configuration error: %s", exc)
        return 2

    setup_logging(config.runtime.log_level)

    try:
        if args.check:
            report = run_preflight(config)
            print("")
            print("All dependencies are ready:")
            print(f"  ffmpeg        : {report.ffmpeg_version}")
            print(f"  yt-dlp        : {report.yt_dlp_version}")
            print(f"  faster-whisper: installed (model {config.whisper.model})")
            print(f"  ollama        : {config.ollama.base_url} (version {report.ollama_version or 'unknown'})")
            print(f"  llm model     : {report.ollama_model}")
            return 0

        pipeline = Pipeline(config)
        result = pipeline.run(args.url, clips=_auto_clip_count(config, args))

        if args.json_summary:
            import json

            print(json.dumps(result_to_json(result), indent=2, ensure_ascii=False))
        else:
            print(summarise(result))
        return 0

    except ClipperError as exc:
        log.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        log.warning("Interrupted - partial files are kept in the workspace for a resume.")
        return 130
    except Exception as exc:  # unexpected: show the type so bug reports are useful
        log.error("Unexpected %s: %s", type(exc).__name__, exc)
        if config.runtime.log_level == "DEBUG":
            raise
        log.error("Re-run with --debug for the full traceback.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
