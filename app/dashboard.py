"""Local Flask dashboard backed by the existing clipping pipeline."""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict

from flask import Flask, abort, jsonify, request, send_file

from .config import load_config
from .errors import ClipperError
from .pipeline import Pipeline

_URL_RE = re.compile(r"https?://[^\s]+", re.I)
_SECRET_RE = re.compile(r"(?i)\b(token|api[_-]?key|password|secret|cookie)=([^\s&]+)")


def create_app(config_path: Path | None = None, env_path: Path | None = None) -> Flask:
    app = Flask(__name__)
    config = load_config(config_path=config_path, env_path=env_path)
    history_path = Path(config.paths.workspace_dir) / "dashboard_jobs.json"
    lock = threading.RLock()
    jobs: Dict[str, Dict[str, Any]] = {}
    running = {"id": None}
    history_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        for job in json.loads(history_path.read_text(encoding="utf-8")):
            if job.get("status") == "running":
                job["status"] = "interrupted"
            jobs[job["id"]] = job
    except (OSError, ValueError, TypeError):
        pass

    def persist() -> None:
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_text(json.dumps(list(jobs.values()), indent=2), encoding="utf-8")

    def safe_message(message: str) -> str:
        return _SECRET_RE.sub(r"\1=[REDACTED]", _URL_RE.sub("[URL]", message))[:500]

    class ProgressHandler(logging.Handler):
        def __init__(self, job_id: str) -> None:
            super().__init__(logging.INFO)
            self.job_id = job_id

        def emit(self, record: logging.LogRecord) -> None:
            message = safe_message(record.getMessage())
            with lock:
                job = jobs.get(self.job_id)
                if not job:
                    return
                job["stage"] = message
                job["messages"].append({"time": time.time(), "level": record.levelname, "message": message})
                job["messages"] = job["messages"][-200:]
                if record.levelno >= logging.ERROR:
                    job["error"] = message
                persist()

    def worker(job_id: str, url: str, overrides: Dict[str, Any]) -> None:
        handler = ProgressHandler(job_id)
        from .logger import get_logger

        logger = get_logger()
        logger.addHandler(handler)
        try:
            job_config = load_config(
                config_path=config_path,
                env_path=env_path,
                overrides={k: v for k, v in overrides.items() if k != "clip.requested_clips"},
            )
            result = Pipeline(job_config).run(url, clips=overrides.get("clip.requested_clips"))
            clips = []
            for artifact in result.clips:
                metadata = dict(artifact.metadata)
                metadata.pop("source_url", None)
                if isinstance(metadata.get("models"), dict):
                    metadata["models"].pop("llm_host", None)
                file_paths = [r.path for r in artifact.renders] + [
                    artifact.source_clip,
                    artifact.output_dir / "metadata.json",
                    artifact.output_dir / "clip_analysis.json",
                    artifact.output_dir / "subtitles.srt",
                    artifact.output_dir / "subtitles.vtt",
                ]
                files = [str(path.relative_to(job_config.paths.output_dir)) for path in file_paths
                         if path and path.is_file()]
                clips.append({"index": artifact.clip.index, "start": artifact.clip.start,
                              "end": artifact.clip.end, "duration": artifact.clip.duration,
                              "score": artifact.clip.score, "metadata": metadata,
                              "output_dir": str(artifact.output_dir), "files": files})
            with lock:
                jobs[job_id].update(status="completed", stage="Completed", clips=clips,
                                    source_title=result.source.title, output_root=str(result.output_root))
                running["id"] = None
                persist()
        except Exception as exc:
            with lock:
                jobs[job_id].update(status="failed", stage="Failed", error=safe_message(str(exc)) or type(exc).__name__)
                running["id"] = None
                persist()
        finally:
            logger.removeHandler(handler)

    @app.get("/")
    def index():
        return _PAGE

    @app.get("/api/jobs")
    def list_jobs():
        with lock:
            return jsonify({"jobs": list(reversed(list(jobs.values())))})

    @app.post("/api/jobs")
    def start_job():
        payload = request.get_json(silent=True) or request.form
        url = str(payload.get("url", "")).strip()
        if not url:
            return jsonify(error="A video URL is required."), 400
        try:
            overrides: Dict[str, Any] = {
                "video.formats": payload.get("formats", config.video.formats),
                "clip.target_duration": float(payload.get("duration", config.clip.target_duration)),
                "whisper.model": payload.get("whisper", config.whisper.model),
                "whisper.language": payload.get("language") or None,
                "whisper.device": payload.get("device", config.whisper.device),
                "ollama.model": payload.get("ollama_model", config.ollama.model),
            }
            target_duration = float(overrides["clip.target_duration"])
            overrides["clip.min_duration"] = max(3.0, target_duration - 5.0)
            overrides["clip.max_duration"] = target_duration + 5.0
            mode = payload.get("clip_mode", "auto")
            if mode == "exact":
                count = int(payload.get("clips", config.clip.clips))
                if count < 1:
                    return jsonify(error="Clip count must be at least 1."), 400
                overrides["clip.requested_clips"] = count
            elif mode != "auto":
                return jsonify(error="Clip mode must be auto or exact."), 400
            cfg_overrides = {k: v for k, v in overrides.items() if k != "clip.requested_clips"}
            load_config(config_path=config_path, env_path=env_path, overrides=cfg_overrides)
        except (ValueError, ClipperError) as exc:
            return jsonify(error=safe_message(str(exc))), 400
        with lock:
            if running["id"]:
                return jsonify(error="A job is already running.", job_id=running["id"]), 409
            job_id = uuid.uuid4().hex[:12]
            jobs[job_id] = {"id": job_id, "status": "running", "stage": "Queued",
                            "created_at": time.time(), "messages": [], "clips": []}
            running["id"] = job_id
            persist()
        thread = threading.Thread(target=worker, args=(job_id, url, overrides), daemon=True)
        thread.start()
        return jsonify({"job_id": job_id}), 202

    @app.get("/api/jobs/<job_id>")
    def get_job(job_id: str):
        with lock:
            job = jobs.get(job_id)
            if not job:
                abort(404)
            return jsonify(job)

    @app.get("/api/files/<path:relative>")
    def get_file(relative: str):
        root = Path(config.paths.output_dir).resolve()
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            abort(404)
        if path.name in {"metadata.json", "clip_analysis.json"}:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                data.pop("source_url", None)
                if isinstance(data.get("source"), dict):
                    data["source"].pop("url", None)
                if isinstance(data.get("models"), dict):
                    data["models"].pop("llm_host", None)
                return jsonify(data)
            except (ValueError, OSError):
                abort(404)
        return send_file(path, as_attachment=request.args.get("download") == "1")

    return app


_PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>AI Video Clipper</title>
<style>body{font:16px system-ui;max-width:900px;margin:2rem auto;padding:0 1rem;color:#202124}input,select,button{font:inherit;padding:.55rem;margin:.25rem}label{display:inline-block;margin:.45rem}button{cursor:pointer}article{border:1px solid #ddd;border-radius:8px;padding:1rem;margin:1rem 0}pre{white-space:pre-wrap;background:#f5f5f5;padding:.75rem}.grid{display:flex;flex-wrap:wrap;gap:.5rem}</style></head><body>
<h1>AI Video Clipper</h1><form id="form"><label>Video URL <input name="url" type="url" required style="width:28rem"></label><div class="grid">
<label>Format <select name="formats"><option value="both">Both</option><option value="16:9">16:9</option><option value="9:16">9:16</option></select></label><label>Clip duration <input name="duration" type="number" value="30" min="3"></label>
<label>Clip count <select name="clip_mode" id="mode"><option value="auto">Automatic (~5/hour, max 30)</option><option value="exact">Exact</option></select></label><label id="countLabel" hidden>Count <input name="clips" type="number" value="5" min="1"></label>
<label>Whisper <select name="whisper"><option>small</option><option>tiny</option><option>base</option><option>medium</option><option>large-v3</option></select></label><label>Language <input name="language" placeholder="auto"></label>
<label>Device <select name="device"><option>auto</option><option>cpu</option><option>cuda</option></select></label><label>Ollama model <input name="ollama_model" value="qwen2.5"></label></div><button>Start clipping</button></form><p id="notice"></p><h2>Jobs</h2><div id="jobs"></div>
<script>const form=document.querySelector('#form'),mode=document.querySelector('#mode'),esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));mode.onchange=()=>document.querySelector('#countLabel').hidden=mode.value!=='exact';form.onsubmit=async e=>{e.preventDefault();let r=await fetch('/api/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(Object.fromEntries(new FormData(form)))}),d=await r.json();document.querySelector('#notice').textContent=r.ok?'Started job '+d.job_id:(d.error||'Could not start job');refresh()};async function refresh(){let d=await(await fetch('/api/jobs')).json();document.querySelector('#jobs').innerHTML=d.jobs.map(j=>`<article><b>${esc(j.status)}</b> — ${esc(j.stage)}<pre>${esc((j.messages||[]).slice(-12).map(x=>x.message).join('\\n'))}${j.error?'\\nError: '+esc(j.error):''}</pre>${(j.clips||[]).map(c=>`<h3>Clip ${c.index}: ${esc(c.metadata.title)}</h3><p>${c.start.toFixed(1)}–${c.end.toFixed(1)} sec · score ${c.score}</p><p>${Object.entries(c.metadata).filter(([k])=>['description','reason','hook','hashtags'].includes(k)).map(([k,v])=>esc(k+': '+(Array.isArray(v)?v.join(' '):v))).join('<br>')}</p>${c.files.map(f=>`<a href="/api/files/${encodeURIComponent(f)}">${esc(f.split('/').pop())}</a> · <a href="/api/files/${encodeURIComponent(f)}?download=1">download</a>`).join('<br>')}`).join('')}</article>`).join('')||'No jobs yet.'}refresh();setInterval(refresh,2000);</script></body></html>"""


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the local AI Video Clipper dashboard")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent.parent / "config.yaml")
    parser.add_argument("--env", type=Path, default=Path(__file__).resolve().parent.parent / ".env")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    create_app(args.config, args.env).run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
