from __future__ import annotations

import json

from app.transcription import Transcriber


def test_transcript_cache_tracks_whisper_settings(tmp_path, transcript, config):
    path = transcript.save(tmp_path / "transcript.json")
    settings = tmp_path / "transcription_settings.json"
    settings.write_text(json.dumps({"model": "small", "language": None, "beam_size": 5}), encoding="utf-8")
    transcriber = Transcriber(config)
    assert transcriber._load_cached(path, settings_path=settings) is not None
    config.whisper.beam_size = 1
    assert transcriber._load_cached(path, settings_path=settings) is None
