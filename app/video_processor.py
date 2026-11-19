"""Video cutting, reframing and rendering (FFmpeg).

Three jobs:

1. cut the selected segment losslessly-ish into ``original_clip.mp4``;
2. reframe it for each requested aspect ratio (16:9 letterbox-safe scaling,
   9:16 full-frame fit over a blurred fill background);
3. burn the captions in and encode a social-ready H.264/AAC MP4.

Only numbers that have already been validated reach the FFmpeg argument list,
and every call uses an argument vector (never a shell string).
"""

from __future__ import annotations

import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .config import AppConfig
from .errors import RenderError
from .logger import get_logger
from .media import FFmpeg, MediaInfo
from .utils import clamp

log = get_logger("video")


@dataclass
class RenderResult:
    """One rendered output file."""

    format: str
    path: Path
    width: int
    height: int
    duration: float
    crop_center: Optional[float] = None
    crop_strategy: str = ""
    burned_subtitles: bool = False


class VideoProcessor:
    """FFmpeg-driven clip extraction and composition."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.ffmpeg = FFmpeg(config.runtime.ffmpeg_path, config.runtime.ffprobe_path)

    # -- cutting ---------------------------------------------------------- #
    def cut_clip(self, source: Path, start: float, end: float, destination: Path) -> Path:
        """Extract ``[start, end]`` into its own file (re-encoded, so the cut is exact)."""

        duration = float(end) - float(start)
        if duration <= 0:
            raise RenderError(f"Refusing to cut a non-positive duration ({duration:.2f}s).")

        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        video = self.config.video

        args: List[str] = [
            "-ss", f"{max(0.0, float(start)):.3f}",
            "-i", str(source),
            "-t", f"{duration:.3f}",
            "-map", "0:v:0",
            "-map", "0:a:0?",
            "-c:v", str(video.video_codec),
            "-preset", str(video.preset),
            "-crf", str(int(video.crf)),
            "-pix_fmt", "yuv420p",
            "-c:a", str(video.audio_codec),
            "-b:a", str(video.audio_bitrate),
            "-ac", "2",
            "-movflags", "+faststart",
            "-avoid_negative_ts", "make_zero",
            str(destination),
        ]
        log.info("Extracting clip segment (%.2fs)", duration)
        self.ffmpeg.run(args, description="clip extraction")
        if not destination.is_file() or destination.stat().st_size == 0:
            raise RenderError(f"Clip extraction produced no output: {destination}")
        return destination

    # -- smart crop ------------------------------------------------------- #
    def detect_subject_center(
        self,
        clip_path: Path,
        workspace: Path,
        *,
        info: Optional[MediaInfo] = None,
    ) -> Tuple[float, str]:
        """Find the horizontal centre of the main subject as a 0.0-1.0 ratio.

        Uses OpenCV face detection over sampled frames when OpenCV is
        installed; falls back to the centre of the frame otherwise. The return
        value is always safe to use for cropping.
        """

        if not self.config.video.smart_crop:
            return 0.5, "center (smart crop disabled)"

        info = info or self.ffmpeg.probe(clip_path)
        try:
            import cv2  # type: ignore
        except ImportError:
            log.info("OpenCV not installed - using centre crop for 9:16")
            return 0.5, "center (opencv unavailable)"

        frames_dir = Path(workspace) / "frames"
        if frames_dir.exists():
            shutil.rmtree(frames_dir, ignore_errors=True)
        try:
            frames = self.ffmpeg.extract_frames(
                clip_path,
                frames_dir,
                start=0.0,
                duration=max(1.0, info.duration),
                count=int(self.config.video.smart_crop_samples),
            )
        except RenderError as exc:
            log.debug("Frame sampling failed (%s) - centre crop", exc)
            return 0.5, "center (frame sampling failed)"

        if not frames:
            return 0.5, "center (no frames sampled)"

        try:
            cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            cascade = cv2.CascadeClassifier(str(cascade_path))
            profile_path = Path(cv2.data.haarcascades) / "haarcascade_profileface.xml"
            profile = cv2.CascadeClassifier(str(profile_path))
        except Exception as exc:  # pragma: no cover - depends on the install
            log.debug("Could not load OpenCV cascades (%s)", exc)
            return 0.5, "center (cascade unavailable)"

        centers: List[float] = []
        weights: List[float] = []
        for frame_path in frames:
            image = cv2.imread(str(frame_path))
            if image is None:
                continue
            height, width = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            detections = cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(40, 40))
            if len(detections) == 0 and not profile.empty():
                detections = profile.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(40, 40))
            for (x, y, w, h) in detections:
                centers.append((float(x) + float(w) / 2.0) / float(width))
                weights.append(float(w) * float(h))

        if not frames_kept_for_debug(self.config):
            shutil.rmtree(frames_dir, ignore_errors=True)

        if not centers:
            log.info("No face detected - using centre crop for 9:16")
            return 0.5, "center (no subject detected)"

        # Weighted mean, biased towards the largest (closest) face.
        total_weight = sum(weights) or 1.0
        center = sum(c * w for c, w in zip(centers, weights)) / total_weight
        # Pull slightly towards the middle so a jittery detection cannot push
        # the subject to the very edge of the crop.
        center = 0.75 * center + 0.25 * 0.5
        center = clamp(center, 0.12, 0.88)
        log.info("Smart crop: subject centred at %.0f%% of frame width (%d detections)", center * 100, len(centers))
        return center, f"face-tracked ({len(centers)} detections)"

    # -- rendering -------------------------------------------------------- #
    def render(
        self,
        clip_path: Path,
        destination: Path,
        fmt: str,
        *,
        subtitle_path: Optional[Path] = None,
        subject_center: float = 0.5,
        crop_strategy: str = "",
        info: Optional[MediaInfo] = None,
    ) -> RenderResult:
        """Render ``clip_path`` into ``fmt`` (``"16:9"`` or ``"9:16"``)."""

        clip_path = Path(clip_path)
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        info = info or self.ffmpeg.probe(clip_path)
        width, height = self.config.video.resolution_for(fmt)

        if info.width <= 0 or info.height <= 0:
            raise RenderError(f"Could not read the clip's dimensions: {clip_path}")

        filters, used_center, strategy = self._build_filters(
            info, width, height, fmt, subject_center, crop_strategy
        )

        burn = bool(subtitle_path and self.config.subtitles.enabled and self.config.subtitles.burn_in)
        cwd: Optional[Path] = None
        if burn:
            subtitle_path = Path(subtitle_path)  # type: ignore[arg-type]
            if not subtitle_path.is_file():
                log.warning("Subtitle file missing (%s) - rendering without captions", subtitle_path)
                burn = False
            else:
                # Running from the subtitle's directory avoids all filter-path
                # escaping problems (Windows drive letters, colons, spaces).
                cwd = subtitle_path.parent
                filters.append(f"ass={_filter_escape(subtitle_path.name)}")

        filters.append("format=yuv420p")
        video_filter = ",".join(filters)

        args: List[str] = ["-i", str(clip_path.resolve()), "-vf", video_filter]
        if self.config.video.fps:
            args += ["-r", f"{float(self.config.video.fps):.3f}"]
        if info.has_audio and self.config.video.loudness_normalize:
            args += ["-af", "loudnorm=I=-16:TP=-1.5:LRA=11"]
        args += [
            "-c:v", str(self.config.video.video_codec),
            "-preset", str(self.config.video.preset),
            "-crf", str(int(self.config.video.crf)),
            "-profile:v", "high",
            "-pix_fmt", "yuv420p",
        ]
        if info.has_audio:
            args += ["-c:a", str(self.config.video.audio_codec), "-b:a", str(self.config.video.audio_bitrate), "-ac", "2"]
        else:
            args += ["-an"]
        args += ["-movflags", "+faststart", str(destination.resolve())]

        log.info("Rendering %s version (%dx%d)", fmt, width, height)
        self.ffmpeg.run(args, description=f"{fmt} render", cwd=cwd)
        if not destination.is_file() or destination.stat().st_size == 0:
            raise RenderError(f"Render produced no output: {destination}")

        rendered = self.ffmpeg.probe(destination)
        return RenderResult(
            format=fmt,
            path=destination,
            width=rendered.width or width,
            height=rendered.height or height,
            duration=rendered.duration,
            crop_center=used_center,
            crop_strategy=strategy,
            burned_subtitles=burn,
        )

    # -- filter construction ---------------------------------------------- #
    def _build_filters(
        self,
        info: MediaInfo,
        out_width: int,
        out_height: int,
        fmt: str,
        subject_center: float,
        crop_strategy: str,
    ) -> Tuple[List[str], Optional[float], str]:
        target_aspect = out_width / out_height
        source_aspect = info.width / info.height

        if fmt == "9:16" or target_aspect < 1.0:
            return self._vertical_filters(
                info, out_width, out_height, target_aspect, source_aspect, subject_center, crop_strategy
            )
        return self._landscape_filters(info, out_width, out_height, target_aspect, source_aspect)

    def _vertical_filters(
        self,
        info: MediaInfo,
        out_width: int,
        out_height: int,
        target_aspect: float,
        source_aspect: float,
        subject_center: float,
        crop_strategy: str,
    ) -> Tuple[List[str], Optional[float], str]:
        """Fit the complete source frame over a blurred 9:16 background."""

        if abs(source_aspect - target_aspect) <= 0.01:
            return [f"scale={out_width}:{out_height}:flags=lanczos"], None, "scaled full frame"

        filters = [
            f"split=2[bg][fg];"
            f"[bg]scale={out_width}:{out_height}:force_original_aspect_ratio=increase:flags=lanczos,"
            f"crop={out_width}:{out_height},boxblur=24:2[blurred];"
            f"[fg]scale={out_width}:{out_height}:force_original_aspect_ratio=decrease:flags=lanczos[full];"
            f"[blurred][full]overlay=(W-w)/2:(H-h)/2"
        ]
        log.info("9:16 render: keeping the full source frame over a blurred background")
        return filters, None, "blurred background (full frame)"

    def _landscape_filters(
        self,
        info: MediaInfo,
        out_width: int,
        out_height: int,
        target_aspect: float,
        source_aspect: float,
    ) -> Tuple[List[str], Optional[float], str]:
        filters = [
            f"scale={out_width}:{out_height}:force_original_aspect_ratio=decrease:flags=lanczos"
        ]
        if abs(source_aspect - target_aspect) > 0.01:
            # Blurred pillarbox looks far better than black bars for vertical
            # or square sources placed in a 16:9 frame.
            filters = [
                f"split=2[bg][fg];"
                f"[bg]scale={out_width}:{out_height}:force_original_aspect_ratio=increase,"
                f"crop={out_width}:{out_height},boxblur=24:2[blurred];"
                f"[fg]scale={out_width}:{out_height}:force_original_aspect_ratio=decrease:flags=lanczos[scaled];"
                f"[blurred][scaled]overlay=(W-w)/2:(H-h)/2"
            ]
            return filters, None, "blurred pillarbox"
        filters.append(f"pad={out_width}:{out_height}:(ow-iw)/2:(oh-ih)/2:color=black")
        return filters, None, "scaled"


def frames_kept_for_debug(config: AppConfig) -> bool:
    return bool(config.runtime.keep_temp_files)


def _filter_escape(value: str) -> str:
    """Escape a value used inside an FFmpeg filter argument."""

    return (
        str(value)
        .replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace("'", "\\'")
        .replace("[", "\\[")
        .replace("]", "\\]")
        .replace(",", "\\,")
    )
