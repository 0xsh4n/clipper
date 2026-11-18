"""End-to-end pipeline wiring with every external dependency stubbed out.

No network, no FFmpeg, no Ollama, no Whisper: the stubs stand in for the
download, transcription, LLM and render stages so the orchestration, caching
and output layout can be verified in CI.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from app import pipeline as pipeline_module
from app.config import load_config
from app.downloader import SourceVideo
from app.media import MediaInfo
from app.pipeline import Pipeline, automatic_clip_count
from app.preflight import PreflightReport
from app.transcript import Transcript
from app.video_processor import RenderResult

SCORES = {
    "hook": 9,
    "interestingness": 8,
    "information_value": 8,
    "emotional_impact": 7,
    "shareability": 8,
    "clarity": 9,
    "standalone_context": 8,
}


@pytest.mark.parametrize("duration,count", [(60, 1), (1800, 3), (3600, 5), (7200, 10), (21600, 30), (86400, 30)])
def test_automatic_clip_count(duration, count):
    assert automatic_clip_count(duration) == count


class StubLLM:
    """Answers the three prompt stages by looking at what was asked."""

    def __init__(self) -> None:
        self.calls: List[str] = []

    def generate_json(self, prompt: str, **_: Any) -> Any:
        self.calls.append(prompt)
        if "SHORTLISTED SEGMENTS" in prompt:
            ids = self._ids(prompt)
            return {
                "selected": True,
                "selections": [
                    {
                        "id": ids[0],
                        "title": "Data quality beats compute",
                        "hook": "Most people think that training a model needs a huge cluster of GPUs.",
                        "reason": "A contrarian claim backed by a concrete measurement.",
                        "score": 9.2,
                    }
                ],
            }
        if "CLIP TRANSCRIPT" in prompt:
            return {
                "title": "Data quality beats compute",
                "description": "Why a single laptop gets you most of the way.",
                "hashtags": ["#shorts", "#ai"],
                "alt_titles": ["The GPU myth"],
            }
        return {"scores": [{"id": i, **SCORES, "title": f"c{i}"} for i in self._ids(prompt)]}

    @staticmethod
    def _ids(prompt: str) -> List[int]:
        """Read the candidate ids out of the rendered prompt."""

        anchor = max(prompt.find("CANDIDATE SEGMENTS"), prompt.find("SHORTLISTED SEGMENTS"))
        start = prompt.find("[", anchor)
        depth = 0
        for index in range(start, len(prompt)):
            if prompt[index] == "[":
                depth += 1
            elif prompt[index] == "]":
                depth -= 1
                if depth == 0:
                    payload = json.loads(prompt[start : index + 1])
                    return [int(entry["id"]) for entry in payload]
        raise AssertionError("no candidate payload found in the prompt")


@pytest.fixture()
def stub_pipeline(tmp_path, monkeypatch, transcript: Transcript):
    """A Pipeline whose external stages are replaced by fast stubs."""

    source_file = tmp_path / "source.mp4"
    source_file.write_bytes(b"not really a video")

    config = load_config(
        overrides={
            "paths.output_dir": str(tmp_path / "output"),
            "paths.workspace_dir": str(tmp_path / "workspace"),
            "clip.max_candidates": 6,
            "ollama.candidates_per_request": 3,
        },
        environ={},
    )

    monkeypatch.setattr(
        pipeline_module, "run_preflight", lambda cfg, require_llm=True: PreflightReport()
    )

    pipeline = Pipeline(config)
    stub_llm = StubLLM()
    pipeline.analyzer.client = stub_llm
    pipeline.metadata.analyzer = pipeline.analyzer

    source = SourceVideo(
        path=source_file,
        video_id="dQw4w9WgXcQ",
        title="How data quality beats compute",
        duration=transcript.duration,
        url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        uploader="Example Channel",
        width=1920,
        height=1080,
        fps=30.0,
    )

    downloads: Dict[str, int] = {"count": 0}

    def fake_download(url, workspace, use_cache=True):
        downloads["count"] += 1
        return source

    transcriptions: Dict[str, int] = {"count": 0}

    def fake_transcribe(audio_path, workspace, use_cache=True, expected_duration=0.0):
        transcriptions["count"] += 1
        return transcript

    def fake_extract_audio(source_path, workspace, use_cache=True):
        audio = Path(workspace) / "audio" / "audio.wav"
        audio.parent.mkdir(parents=True, exist_ok=True)
        audio.write_bytes(b"RIFF")
        return audio

    def fake_cut(src, start, end, destination):
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"clip")
        return destination

    def fake_probe(path):
        return MediaInfo(
            path=Path(path), duration=30.0, width=1920, height=1080, fps=30.0, has_audio=True
        )

    renders: List[Dict[str, Any]] = []

    def fake_render(clip_path, destination, fmt, **kwargs):
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"render")
        width, height = config.video.resolution_for(fmt)
        renders.append({"format": fmt, "subtitle_path": kwargs.get("subtitle_path")})
        return RenderResult(
            format=fmt,
            path=destination,
            width=width,
            height=height,
            duration=30.0,
            crop_center=kwargs.get("subject_center"),
            crop_strategy=kwargs.get("crop_strategy", ""),
            burned_subtitles=kwargs.get("subtitle_path") is not None,
        )

    monkeypatch.setattr(pipeline.downloader, "download", fake_download)
    monkeypatch.setattr(pipeline.transcriber, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(pipeline.transcriber, "transcribe", fake_transcribe)
    monkeypatch.setattr(pipeline.video, "cut_clip", fake_cut)
    monkeypatch.setattr(pipeline.video, "render", fake_render)
    monkeypatch.setattr(pipeline.video, "detect_subject_center", lambda *a, **k: (0.6, "face-tracked"))
    monkeypatch.setattr(pipeline.ffmpeg, "probe", fake_probe)

    return {
        "pipeline": pipeline,
        "config": config,
        "llm": stub_llm,
        "renders": renders,
        "downloads": downloads,
        "transcriptions": transcriptions,
        "url": source.url,
    }


def test_full_run_writes_the_documented_output_layout(stub_pipeline):
    result = stub_pipeline["pipeline"].run(stub_pipeline["url"], clips=1)

    assert len(result.clips) == 1
    artifacts = result.clips[0]
    output_dir = artifacts.output_dir
    assert output_dir.name == "clip_01"

    expected = {
        "original_clip.mp4",
        "shorts_9x16.mp4",
        "youtube_16x9.mp4",
        "subtitles.srt",
        "subtitles.vtt",
        "transcript.json",
        "clip_transcript.json",
        "clip_analysis.json",
        "metadata.json",
    }
    assert expected <= {p.name for p in output_dir.iterdir()}

    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["title"] == "Data quality beats compute"
    assert metadata["hashtags"] == ["#shorts", "#ai"]
    assert metadata["source_url"] == stub_pipeline["url"]
    assert 25.0 <= metadata["clip_duration"] <= 35.0

    analysis = json.loads((output_dir / "clip_analysis.json").read_text(encoding="utf-8"))
    assert analysis["selection_method"] == "qwen-final-selection"
    assert analysis["top_candidates"]
    assert analysis["reason"]


def test_run_avoids_the_intro_and_the_sponsor_read(stub_pipeline):
    result = stub_pipeline["pipeline"].run(stub_pipeline["url"], clips=1)
    clip = result.clips[0].clip
    assert "SAVE20" not in clip.text
    assert clip.start > 0.0


def test_both_formats_are_rendered_with_their_own_captions(stub_pipeline):
    stub_pipeline["pipeline"].run(stub_pipeline["url"], clips=1)
    rendered = {entry["format"]: entry for entry in stub_pipeline["renders"]}
    assert set(rendered) == {"16:9", "9:16"}
    assert rendered["9:16"]["subtitle_path"].name.endswith("_9x16.ass")
    assert rendered["16:9"]["subtitle_path"].name.endswith("_16x9.ass")


def test_second_run_reuses_the_cached_analysis(stub_pipeline):
    pipeline = stub_pipeline["pipeline"]
    pipeline.run(stub_pipeline["url"], clips=1)
    first_calls = len(stub_pipeline["llm"].calls)

    pipeline.run(stub_pipeline["url"], clips=1)
    # Candidate scoring and final selection come from the workspace cache; only
    # the metadata prompt runs again for the new export.
    assert len(stub_pipeline["llm"].calls) == first_calls + 1


def test_second_run_does_not_overwrite_the_first_export(stub_pipeline):
    pipeline = stub_pipeline["pipeline"]
    first = pipeline.run(stub_pipeline["url"], clips=1).clips[0].output_dir
    second = pipeline.run(stub_pipeline["url"], clips=1).clips[0].output_dir
    assert first != second
    assert (first / "metadata.json").is_file() and (second / "metadata.json").is_file()


def test_no_cache_forces_fresh_analysis(tmp_path, stub_pipeline, monkeypatch):
    pipeline = stub_pipeline["pipeline"]
    pipeline.run(stub_pipeline["url"], clips=1)
    calls = len(stub_pipeline["llm"].calls)

    pipeline.config.runtime.use_cache = False
    pipeline.run(stub_pipeline["url"], clips=1)
    assert len(stub_pipeline["llm"].calls) > calls + 1


def test_multiple_clips_are_distinct(stub_pipeline, monkeypatch):
    pipeline = stub_pipeline["pipeline"]

    def many_selections(prompt: str, **_: Any) -> Any:
        if "SHORTLISTED SEGMENTS" in prompt:
            ids = StubLLM._ids(prompt)
            return {
                "selected": True,
                "selections": [
                    {"id": i, "title": f"Clip {i}", "hook": "h", "reason": "r", "score": 9.0 - n}
                    for n, i in enumerate(ids[:3])
                ],
            }
        return StubLLM().generate_json(prompt)

    monkeypatch.setattr(pipeline.analyzer.client, "generate_json", many_selections)
    result = pipeline.run(stub_pipeline["url"], clips=2)

    assert len(result.clips) >= 1
    starts = [art.clip.start for art in result.clips]
    assert len(set(starts)) == len(starts)
    if len(result.clips) > 1:
        assert all(art.output_dir.name.startswith("clip_") for art in result.clips)


def test_workspace_temp_files_are_cleaned(stub_pipeline):
    result = stub_pipeline["pipeline"].run(stub_pipeline["url"], clips=1)
    assert not (result.workspace / "audio").exists()
    assert not (result.workspace / "clips").exists()
    # Transcript and analysis survive so a re-run can resume.
    assert (result.workspace / "transcript").exists()
    assert (result.workspace / "analysis" / "candidates.json").is_file()


def test_keep_temp_files_preserves_the_workspace(stub_pipeline):
    pipeline = stub_pipeline["pipeline"]
    pipeline.config.runtime.keep_temp_files = True
    result = pipeline.run(stub_pipeline["url"], clips=1)
    assert (result.workspace / "audio").exists()


def test_invalid_url_is_rejected_before_any_work(stub_pipeline):
    from app.errors import ClipperError

    with pytest.raises(ClipperError):
        stub_pipeline["pipeline"].run("https://vimeo.com/12345")
    assert stub_pipeline["downloads"]["count"] == 0
