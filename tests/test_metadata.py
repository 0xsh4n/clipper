"""metadata.json / clip_analysis.json shape."""

from __future__ import annotations

import json
from pathlib import Path

from app.candidate_generator import Candidate
from app.clip_selector import Clip
from app.downloader import SourceVideo
from app.metadata_generator import MetadataGenerator
from app.video_processor import RenderResult


def make_source() -> SourceVideo:
    return SourceVideo(
        path=Path("source.mp4"),
        video_id="dQw4w9WgXcQ",
        title="How data quality beats compute",
        duration=3600.0,
        url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        uploader="Example Channel",
    )


def make_clip() -> Clip:
    return Clip(
        index=1,
        start=812.4,
        end=842.1,
        title="The surprising truth about scaling",
        hook="Most people think you need a cluster.",
        reason="A contrarian, self-contained claim backed by a concrete number.",
        score=9.1,
        candidate_id=7,
        scores={"hook": 9.0, "clarity": 8.0},
        text="Most people think you need a cluster. It turns out ninety percent of the quality...",
    )


def test_metadata_without_an_llm_has_every_required_field(tmp_path, config):
    renders = [
        RenderResult(
            format="9:16",
            path=tmp_path / "shorts_9x16.mp4",
            width=1080,
            height=1920,
            duration=29.7,
            crop_center=0.62,
            crop_strategy="face-tracked",
            burned_subtitles=True,
        )
    ]
    generator = MetadataGenerator(config, analyzer=None)
    metadata = generator.build_metadata(
        source=make_source(),
        clip=make_clip(),
        renders=renders,
        subtitle_files={"srt": tmp_path / "subtitles.srt"},
        output_dir=tmp_path,
        transcript_language="en",
    )

    for key in (
        "source_url", "source_title", "source_duration", "clip_start", "clip_end",
        "clip_duration", "title", "description", "hook", "reason", "score", "generated_at",
    ):
        assert key in metadata, key
    assert metadata["clip_duration"] == 29.7
    assert metadata["outputs"][0]["format"] == "9:16"
    assert metadata["models"]["llm"] == config.ollama.model

    on_disk = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert on_disk == metadata


def test_analysis_report_includes_the_candidate_ranking(tmp_path, config):
    candidates = [
        Candidate(id=i, start=i * 60.0, end=i * 60.0 + 30.0, text=f"c{i}", overall=5.0 + i)
        for i in range(4)
    ]
    analysis = MetadataGenerator(config, analyzer=None).build_analysis(
        clip=make_clip(),
        candidates=candidates,
        output_dir=tmp_path,
        source=make_source(),
        selection_payload={"fallback": False},
        top_n=3,
    )

    assert analysis["selected"] is True
    assert analysis["candidates_considered"] == 4
    assert len(analysis["top_candidates"]) == 3
    assert analysis["top_candidates"][0]["overall"] >= analysis["top_candidates"][1]["overall"]
    assert analysis["scoring_weights"]["hook"] == config.scoring.hook
    assert analysis["selection_method"] == "qwen-final-selection"
    assert (tmp_path / "clip_analysis.json").is_file()


def test_fallback_selection_is_recorded(tmp_path, config):
    analysis = MetadataGenerator(config, analyzer=None).build_analysis(
        clip=make_clip(),
        candidates=[],
        output_dir=tmp_path,
        source=make_source(),
        selection_payload={"fallback": True},
    )
    assert analysis["selection_method"] == "heuristic-fallback"


def test_transcripts_are_written(tmp_path, config, transcript):
    clip = Clip(index=1, start=10.0, end=40.0, text="x")
    paths = MetadataGenerator(config, analyzer=None).write_transcripts(
        transcript=transcript, clip=clip, output_dir=tmp_path
    )
    full = json.loads(paths["transcript"].read_text(encoding="utf-8"))
    clip_only = json.loads(paths["clip_transcript"].read_text(encoding="utf-8"))

    assert full["segment_count"] == len(transcript)
    assert clip_only["segments"]
    assert clip_only["segments"][0]["start"] >= 0.0
    assert clip_only["segments"][0]["source_start"] >= 10.0 - 1e-6
