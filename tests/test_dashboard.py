from __future__ import annotations

import threading
import time

from app import dashboard


def test_dashboard_starts_one_background_job_and_rejects_second(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"paths:\n  output_dir: {tmp_path / 'out'}\n  workspace_dir: {tmp_path / 'work'}\n",
        encoding="utf-8",
    )
    entered = threading.Event()
    release = threading.Event()

    def run(self, url, *, clips=None):
        entered.set()
        release.wait(3)
        raise RuntimeError("finished test job")

    monkeypatch.setattr(dashboard.Pipeline, "run", run)
    client = dashboard.create_app(config_path, tmp_path / ".env").test_client()
    payload = {"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "duration": 30,
               "formats": "both", "clip_mode": "exact", "clips": 2,
               "whisper": "small", "device": "cpu", "ollama_model": "qwen2.5"}
    first = client.post("/api/jobs", json=payload)
    assert first.status_code == 202
    assert entered.wait(1)
    second = client.post("/api/jobs", json=payload)
    assert second.status_code == 409
    job_id = first.json["job_id"]
    assert client.get("/api/jobs/" + job_id).json["status"] == "running"
    release.set()
    for _ in range(30):
        status = client.get("/api/jobs/" + job_id).json["status"]
        if status == "failed":
            break
        time.sleep(0.02)
    assert status == "failed"


def test_dashboard_rejects_invalid_clip_count(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}", encoding="utf-8")
    client = dashboard.create_app(config_path, tmp_path / ".env").test_client()
    response = client.post("/api/jobs", json={"url": "https://youtu.be/dQw4w9WgXcQ",
                                               "clip_mode": "exact", "clips": 0})
    assert response.status_code == 400


def test_dashboard_file_routes_redact_source_and_model_urls(tmp_path):
    output = tmp_path / "out"
    clip = output / "demo" / "clip_01"
    clip.mkdir(parents=True)
    (clip / "metadata.json").write_text(
        '{"title":"Demo","source_url":"https://youtube.test/?token=secret",'
        '"models":{"llm_host":"http://user:secret@host"}}', encoding="utf-8")
    (clip / "clip_analysis.json").write_text(
        '{"source":{"url":"https://youtube.test/?token=secret","title":"Demo"}}',
        encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"paths:\n  output_dir: {output}\n  workspace_dir: {tmp_path / 'work'}\n",
                           encoding="utf-8")
    client = dashboard.create_app(config_path, tmp_path / ".env").test_client()
    metadata = client.get("/api/files/demo/clip_01/metadata.json").json
    analysis = client.get("/api/files/demo/clip_01/clip_analysis.json").json
    assert "source_url" not in metadata
    assert "llm_host" not in metadata["models"]
    assert "url" not in analysis["source"]
