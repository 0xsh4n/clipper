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


_PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta name="theme-color" content="#0b1020"><title>Clipcraft · AI Video Clipper</title>
<style>
:root{color-scheme:dark;--bg:#090d18;--panel:#111827;--panel2:#151e30;--line:#26334a;--muted:#97a6bd;--text:#f3f6fc;--accent:#a3e635;--blue:#65d5ff;--danger:#fb7185}*{box-sizing:border-box}body{margin:0;background:radial-gradient(ellipse at 50% -25%,#233152 0,transparent 48%),var(--bg);color:var(--text);font:15px/1.5 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}button,input,select{font:inherit}button{cursor:pointer}.shell{max-width:1120px;margin:auto;padding:34px 24px 64px}.topbar{display:flex;justify-content:space-between;align-items:center;margin-bottom:42px}.brand{display:flex;align-items:center;gap:11px;font-weight:750;letter-spacing:-.03em}.brandmark{display:grid;place-items:center;width:38px;height:38px;border-radius:12px;background:var(--accent);color:#17210c;font-size:20px}.local{color:var(--muted);font-size:12px;border:1px solid var(--line);border-radius:30px;padding:6px 11px}.hero{max-width:740px;margin:0 auto 28px;text-align:center}.eyebrow{color:var(--accent);font-size:12px;font-weight:750;letter-spacing:.15em;text-transform:uppercase}.hero h1{font-size:clamp(34px,6vw,58px);line-height:1.03;letter-spacing:-.055em;margin:13px 0}.hero p{color:var(--muted);font-size:17px;margin:0 auto;max-width:600px}.card{background:linear-gradient(145deg,rgba(21,30,48,.97),rgba(14,21,35,.98));border:1px solid var(--line);border-radius:20px;box-shadow:0 22px 70px #0004}.form-card{max-width:900px;margin:auto;padding:24px}.url-label,.field label{display:block;color:#c6d1e2;font-size:12px;font-weight:700;margin-bottom:8px}.url-row{display:flex;gap:10px}.url-row input{flex:1;min-width:0}.input,.field select{width:100%;border:1px solid #34425b;background:#0c1322;color:var(--text);border-radius:10px;padding:12px 13px;outline:none;transition:border .15s,box-shadow .15s}.input:focus,.field select:focus{border-color:var(--blue);box-shadow:0 0 0 3px #65d5ff20}.input::placeholder{color:#697991}.fields{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:15px;margin:20px 0}.field{min-width:0}.hint{color:var(--muted);font-size:11px;margin:6px 0 0}.primary{border:0;border-radius:10px;background:var(--accent);color:#17210c;font-weight:800;padding:12px 19px;white-space:nowrap;transition:transform .15s,filter .15s}.primary:hover{filter:brightness(1.07);transform:translateY(-1px)}.primary:disabled{opacity:.55;cursor:wait;transform:none}.notice{min-height:23px;margin:13px 2px 0;color:var(--blue);font-size:13px}.section-head{display:flex;justify-content:space-between;align-items:end;margin:46px 0 14px}.section-head h2{font-size:22px;letter-spacing:-.03em;margin:0}.section-head p{color:var(--muted);font-size:12px;margin:0}.jobs{display:grid;gap:14px}.job{padding:19px 21px}.job-head{display:flex;justify-content:space-between;gap:12px;align-items:center}.job-title{font-weight:750;display:flex;gap:10px;align-items:center}.badge{font-size:11px;text-transform:uppercase;letter-spacing:.06em;padding:4px 9px;border-radius:20px;background:#26334a;color:#d9e3f1}.badge.running{background:#302b16;color:#facc15}.badge.completed{background:#142e22;color:var(--accent)}.badge.failed{background:#321c27;color:var(--danger)}.stage{color:var(--muted);font-size:13px;margin:6px 0 0}.log{margin:13px 0 0;padding:12px 14px;background:#0a101c;border:1px solid #202c40;border-radius:10px;color:#aab8cb;white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.6 ui-monospace,SFMono-Regular,Consolas,monospace;max-height:190px;overflow:auto}.clips{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:12px;margin-top:15px}.clip{padding:14px;border-radius:13px;background:#0c1423;border:1px solid #28364e;min-width:0}.clip h3{font-size:15px;margin:0 0 6px;letter-spacing:-.02em}.clip-meta{color:var(--muted);font-size:12px}.score{color:var(--accent);font-weight:750}.clip p{font-size:13px;color:#bfcbda;margin:10px 0}.clip video{display:block;width:100%;max-height:280px;background:#05070b;border-radius:9px;margin:12px 0;object-fit:contain}.links{display:flex;flex-wrap:wrap;gap:7px}.links a{color:var(--blue);font-size:12px;text-decoration:none;border:1px solid #2c4055;border-radius:7px;padding:5px 8px}.links a:hover{background:#17263a}.empty{padding:28px;text-align:center;color:var(--muted);border:1px dashed #34425b;border-radius:14px}.error{color:var(--danger)}@media(max-width:760px){.fields{grid-template-columns:repeat(2,minmax(0,1fr))}.url-row{flex-direction:column}.primary{width:100%}.shell{padding:22px 15px 45px}.topbar{margin-bottom:34px}.form-card{padding:17px}}@media(max-width:430px){.fields{grid-template-columns:1fr}.section-head{align-items:start;flex-direction:column;gap:4px}}
</style></head><body><main class="shell"><header class="topbar"><div class="brand"><span class="brandmark">✦</span><span>clipcraft</span></div><span class="local">● &nbsp;Local processing</span></header>
<section class="hero"><div class="eyebrow">Find the moment worth sharing</div><h1>Turn long videos into<br>short-form gold.</h1><p>Drop in a video link. AI finds the strongest hooks and moments, then prepares clips ready for Shorts, Reels and TikTok.</p></section>
<form class="card form-card" id="form"><label class="url-label" for="url">Paste a video link</label><div class="url-row"><input class="input" id="url" name="url" type="url" placeholder="YouTube video URL…" required><button class="primary" id="submit">✦ &nbsp;Find my clips</button></div>
<div class="fields"><div class="field"><label for="formats">Export format</label><select id="formats" name="formats"><option value="both">Vertical + landscape</option><option value="9:16">Vertical · 9:16</option><option value="16:9">Landscape · 16:9</option></select></div><div class="field"><label for="duration">Clip length</label><select class="input" id="duration" name="duration"><option value="30">30 seconds</option><option value="45">45 seconds</option><option value="60">60 seconds</option></select></div><div class="field"><label for="whisper">Transcription speed</label><select id="whisper" name="whisper"><option value="base" selected>Base · fastest</option><option value="small">Small · balanced</option><option value="tiny">Tiny · lightest</option><option value="medium">Medium · accurate</option><option value="large-v3">Large v3 · highest accuracy</option></select></div><div class="field"><label for="ollama_model">Clip analysis model</label><select id="ollama_model" name="ollama_model"><option value="qwen3:4b" selected>Qwen 3 · 4B installed</option><option value="qwen2.5:3b">Qwen 2.5 · 3B</option><option value="qwen2.5">Qwen 2.5</option></select><p class="hint">Qwen 3 is selected for stronger clip analysis.</p></div><div class="field"><label for="mode">Clip quantity</label><select id="mode" name="clip_mode"><option value="auto">Auto · about 5 per hour</option><option value="exact">Choose a number</option></select></div><div class="field" id="countField" hidden><label for="clips">Number of clips</label><input class="input" id="clips" name="clips" type="number" value="5" min="1" max="30"></div><div class="field"><label for="language">Spoken language</label><input class="input" id="language" name="language" placeholder="Auto-detect"></div><div class="field"><label for="device">Compute device</label><select id="device" name="device"><option value="auto">Automatic</option><option value="cuda">NVIDIA GPU</option><option value="cpu">CPU</option></select></div></div><p class="hint">Videos are processed locally on this computer. Job history and exports stay on this machine.</p><div class="notice" id="notice" role="status"></div></form>
<div class="section-head"><div><h2>Your clip jobs</h2><p>Progress updates automatically while a video is processing.</p></div><p id="jobCount"></p></div><div class="jobs" id="jobs"><div class="empty">Your finished clips will show up here.</div></div></main>
<script>
const form=document.querySelector('#form'),mode=document.querySelector('#mode'),notice=document.querySelector('#notice'),submit=document.querySelector('#submit'),esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
mode.onchange=()=>{document.querySelector('#countField').hidden=mode.value!=='exact'};
form.onsubmit=async e=>{e.preventDefault();submit.disabled=true;submit.textContent='Starting…';notice.textContent='Adding your video to the queue…';try{const r=await fetch('/api/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(Object.fromEntries(new FormData(form)))}),d=await r.json();notice.textContent=r.ok?'Job '+d.job_id+' started. You can follow its progress below.':(d.error||'Could not start job');notice.className='notice'+(r.ok?'':' error');await refresh()}catch{notice.textContent='Could not connect to the local dashboard.';notice.className='notice error'}finally{submit.disabled=false;submit.innerHTML='✦ &nbsp;Find my clips'}};
async function refresh(){try{const d=await(await fetch('/api/jobs')).json(),jobs=d.jobs||[];document.querySelector('#jobCount').textContent=jobs.length?`${jobs.length} job${jobs.length===1?'':'s'}`:'';document.querySelector('#jobs').innerHTML=jobs.map(j=>{const files=j.clips||[];return `<article class="card job"><div class="job-head"><div><div class="job-title"><span class="badge ${esc(j.status)}">${esc(j.status)}</span>${esc(j.source_title||('Job '+j.id))}</div><div class="stage">${esc(j.stage||'Waiting')}</div></div></div>${j.messages?.length?`<details><summary class="stage" style="cursor:pointer;margin-top:10px">Show processing log</summary><pre class="log">${esc(j.messages.slice(-12).map(x=>x.message).join('\n'))}${j.error?'\nError: '+esc(j.error):''}</pre></details>`:''}${j.error&&!j.messages?.length?`<p class="error">${esc(j.error)}</p>`:''}${files.length?`<div class="clips">${files.map(c=>{const title=c.metadata?.title||`Clip ${c.index}`,mp4=c.files.find(f=>f.toLowerCase().endsWith('.mp4')),desc=c.metadata?.caption||c.metadata?.description||c.metadata?.hook||c.metadata?.reason||'';return `<section class="clip"><h3>${esc(title)}</h3><div class="clip-meta">${Number(c.start).toFixed(1)}–${Number(c.end).toFixed(1)} sec &nbsp;·&nbsp; <span class="score">★ ${Number(c.score).toFixed(1)} / 10</span></div>${mp4?`<video controls preload="none" src="/api/files/${encodeURIComponent(mp4)}"></video>`:''}${desc?`<p><strong>Post caption:</strong> ${esc(desc)}</p>`:''}<p><strong>Hashtags:</strong> ${(c.metadata?.hashtags||[]).map(tag=>`<span style="color:#a3e635;background:#1b2b21;border-radius:20px;padding:3px 7px;font-size:11px">${esc(tag)}</span>`).join(' ')}</p><div class="links">${c.files.map(f=>`<a href="/api/files/${encodeURIComponent(f)}?download=1">↓ ${esc(f.split('/').pop())}</a>`).join('')}</div></section>`}).join('')}</div>`:''}</article>`}).join('')||'<div class="empty">Your finished clips will show up here.</div>'}catch{}}
refresh();setInterval(refresh,2500);
</script></body></html>"""


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
