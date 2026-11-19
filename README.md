# AI Video Clipper

Local AI agent that takes a YouTube URL, finds the strongest ~30 second moment in
the whole video, and renders captioned 16:9 and 9:16 versions into `output/`.

All reasoning happens on your machine with **Ollama + Qwen 2.5**. No OpenAI,
Anthropic, Gemini or any other cloud LLM is used, and no API keys exist anywhere
in the project.

```
ollama serve
python main.py "https://www.youtube.com/watch?v=XXXXXXXXXXX"
```

```
YouTube URL
   -> yt-dlp download (video + audio, live VODs supported)
   -> FFmpeg audio extraction (16 kHz mono WAV)
   -> faster-whisper transcription (word-level timestamps)
   -> overlapping candidate windows + objective metadata
   -> Qwen 2.5 scores every candidate (hook, interest, value, emotion,
      standalone context, shareability, clarity)
   -> Qwen 2.5 picks the final clip and explains why
   -> sentence-aware boundary snapping to ~30 s
   -> SRT / VTT / burned-in captions
   -> 1920x1080 landscape + 1080x1920 subject-aware vertical crop
   -> output/<video>/clip_01/ with metadata.json and clip_analysis.json
```

---

## Table of contents

1. [Requirements](#1-requirements)
2. [Python version](#2-python-version)
3. [FFmpeg installation](#3-ffmpeg-installation)
4. [Ollama installation](#4-ollama-installation)
5. [Ollama model setup](#5-ollama-model-setup)
6. [Whisper model setup](#6-whisper-model-setup)
7. [Installation](#7-installation)
8. [Configuration](#8-configuration)
9. [Running the application](#9-running-the-application)
10. [Example commands](#10-example-commands)
11. [Output structure](#11-output-structure)
12. [Troubleshooting](#12-troubleshooting)
13. [GPU acceleration](#13-gpu-acceleration)
14. [CPU fallback](#14-cpu-fallback)
15. [How to change the LLM](#15-how-to-change-the-llm)
16. [How to change video formats](#16-how-to-change-video-formats)
17. [How clip selection works](#17-how-clip-selection-works)
18. [Project structure](#18-project-structure)
19. [Tests](#19-tests)
20. [Local dashboard](#20-local-dashboard)
21. [Extending the pipeline](#21-extending-the-pipeline)

---

## 1. Requirements

| Component | Minimum | Notes |
|---|---|---|
| Python | 3.10 – 3.12 | 3.13+ may lack `ctranslate2` wheels, see [Troubleshooting](#12-troubleshooting) |
| FFmpeg + FFprobe | any recent build | must be on `PATH` (or set `FFMPEG_PATH`) |
| Ollama | 0.1.30+ | running locally (`ollama serve`) |
| Qwen 2.5 3B | already installed | ~1.9 GB; fast default on this laptop |
| Qwen 3 4B | `ollama pull qwen3:4b` | ~2.5 GB; stronger clip reasoning default |
| Qwen 2.5 3B | `ollama pull qwen2.5:3b` | ~1.9 GB; smaller fallback |
| RAM | 8 GB | 16 GB comfortable |
| Disk | ~5 GB + video size | models + workspace |
| GPU | optional | NVIDIA CUDA speeds up transcription a lot |

Works on **Windows 10/11** and **Linux** (and macOS). There is no OS-specific
code: paths go through `pathlib`, and FFmpeg is always called with an argument
list, never a shell string.

## 2. Python version

Use **Python 3.10, 3.11 or 3.12**.

```bash
python --version
```

`faster-whisper` depends on `ctranslate2`, which does not always publish wheels
for brand-new Python releases. If `pip install faster-whisper` fails with
"no matching distribution", install a 3.12 interpreter and create the virtual
environment with it (`py -3.12 -m venv .venv` on Windows).

## 3. FFmpeg installation

The app verifies both `ffmpeg` and `ffprobe` before doing any work.

**Windows (recommended):**

```powershell
winget install Gyan.FFmpeg
```

Then close and reopen the terminal and check:

```powershell
ffmpeg -version
```

**Windows (manual):** download a build from <https://www.gyan.dev/ffmpeg/builds/>
(`ffmpeg-release-essentials.zip`), extract to `C:\ffmpeg`, then either add
`C:\ffmpeg\bin` to *Path* (System Properties → Environment Variables), or point
the app at it in `.env`:

```
FFMPEG_PATH=C:\ffmpeg\bin\ffmpeg.exe
FFPROBE_PATH=C:\ffmpeg\bin\ffprobe.exe
```

**Windows (chocolatey / scoop):**

```powershell
choco install ffmpeg-full
scoop install ffmpeg
```

**Linux:**

```bash
sudo apt install ffmpeg          # Debian / Ubuntu / Kali
sudo dnf install ffmpeg-free     # Fedora
sudo pacman -S ffmpeg            # Arch
```

**macOS:**

```bash
brew install ffmpeg
```

## 4. Ollama installation

Download the installer from <https://ollama.com/download> (Windows, macOS) or:

```bash
curl -fsSL https://ollama.com/install.sh | sh     # Linux
```

Start the server (the Windows installer runs it in the background already):

```bash
ollama serve
```

Verify:

```bash
curl http://localhost:11434/api/version
```

## 5. Ollama model setup

```bash
ollama pull qwen3:4b
```

For a smaller fallback model, use Qwen 2.5 3B (about 1.9 GB):

```bash
ollama pull qwen2.5:3b
```

Other sizes work too:

```bash
ollama pull qwen2.5:14b      # noticeably better judgement, ~9 GB
ollama pull qwen2.5:3b       # faster on modest hardware
ollama pull qwen2.5:32b      # best quality, needs a lot of VRAM/RAM
```

Select one at runtime with `--model qwen2.5:14b` or `OLLAMA_MODEL=qwen2.5:14b`.

The app checks at startup that Ollama answers **and** that the requested model
is installed, and tells you exactly what to pull if it is not.

## 6. Whisper model setup

Nothing to download by hand: `faster-whisper` fetches the model on first use
(into the Hugging Face cache) and reuses it afterwards.

| Model | Size | Speed (CPU) | Quality | Use when |
|---|---|---|---|---|
| `tiny` | 75 MB | fastest | low | smoke tests |
| `base` | 145 MB | fastest practical | fair | **default**, quick runs on modest hardware |
| `small` | 480 MB | moderate | good | when transcript accuracy matters more |
| `medium` | 1.5 GB | slow on CPU | very good | GPU, or quality matters |
| `large-v3` | 3 GB | GPU recommended | best | accents, noisy audio |

```bash
python main.py URL --whisper medium
# or
WHISPER_MODEL=medium
```

Word-level timestamps are always enabled: they are what makes the captions tight
and the clip boundaries land between sentences instead of mid-word.

## 7. Installation

```bash
git clone <this repo>  # or copy the folder
cd ai-video-clipper

python -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows PowerShell

pip install -r requirements.txt
cp .env.example .env               # copy .env.example .env   on Windows
```

Then verify the whole toolchain:

```bash
python main.py --check
```

```
All dependencies are ready:
  ffmpeg        : ffmpeg version 6.1.1 ...
  yt-dlp        : 2024.08.06
  faster-whisper: installed (model base)
  ollama        : http://localhost:11434 (version 0.3.9)
  llm model     : qwen3:4b
```

`opencv-python-headless` in `requirements.txt` is optional: it powers the
face-aware 9:16 crop. Without it the vertical render falls back to a centre
crop and logs that it did.

## 8. Configuration

Three layers, highest priority first:

1. **CLI flags** — `--duration 30`, `--format 9:16`, `--model qwen2.5:14b`, ...
2. **Environment / `.env`** — see [`.env.example`](.env.example)
3. **`config.yaml`** — everything, including subtitle styling and encoder settings

Most used keys:

```
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5

WHISPER_MODEL=small

TARGET_CLIP_DURATION=30
MIN_CLIP_DURATION=25
MAX_CLIP_DURATION=35

OUTPUT_DIR=./output
VIDEO_FORMATS=16:9,9:16

KEEP_TEMP_FILES=false
```

Subtitle styling is data, not code — change it in `config.yaml`:

```yaml
subtitles:
  enabled: true
  burn_in: true
  font_name: DejaVu Sans    # "Arial" / "Impact" / "Bebas Neue" on Windows
  font_size: 48             # authored for a 1080 px short side, auto-scaled
  max_lines: 2
  max_chars_per_line: 22
  position: bottom          # bottom | center | top
  margin_ratio: 0.14        # keeps captions inside TikTok/Reels safe areas
  uppercase: false
```

`font_size` is authored for a 1080 px tall frame and scales with output height,
so captions stay equally readable at 1920x1080 and 1080x1920. Lines that still
would not fit between the horizontal margins are re-wrapped, and a cue that
cannot fit even then gets its own smaller size — captions never run off frame.

Invalid configuration is rejected before anything is downloaded, with a message
that names the key and the fix.

## 9. Running the application

```bash
ollama serve                                      # terminal 1 (once)
python main.py "https://www.youtube.com/watch?v=XXXXXXXXXXX"
```

Typical log:

```
[INFO] Starting video processing
[INFO] FFmpeg ready: ffmpeg version 6.1.1
[INFO] Ollama ready at http://localhost:11434 (version 0.3.9, model qwen3:4b)
[INFO] Downloading video
[INFO] Video duration: 01:42:18
[INFO] Extracting audio
[INFO] Transcribing (model: base)
[INFO] Transcript generated: 1483 segments, 19204 words, language=en
[INFO] Generating candidate clips
[INFO] Shortlisted 36 candidates spanning 02:15 - 1:38:40
[INFO] Analyzing 36 candidates with qwen3:4b (3 batches)
[INFO] Selecting the final clip with qwen3:4b
[INFO] Selected clip: 13:32 - 14:02 (29.8s, score 9.1) - The GPU myth
[INFO] Extracting clip segment (29.80s)
[INFO] Generating subtitles (11 cues)
[INFO] Smart crop: subject centred at 62% of frame width (9 detections)
[INFO] Rendering 16:9 version (1920x1080)
[INFO] Rendering 9:16 version (1080x1920)
[INFO] Saved clip 1 to output/The_GPU_myth-dQw4w9WgXcQ
[INFO] Finished in 00:06:41
```

Supported URL forms: `watch?v=`, `youtu.be/`, `/live/` (finished streams),
`/shorts/`, `/embed/`. A stream that is **still live** is rejected with an
explanation — clipping needs a complete recording.

## 10. Example commands

```bash
python main.py URL                          # 30 s, both formats, ~5 clips/hour (max 30)
python main.py URL --duration 30
python main.py URL --format 9:16            # Shorts / Reels / TikTok only
python main.py URL --format 16:9            # YouTube landscape only
python main.py URL --format both
python main.py URL --model qwen2.5          # any installed Ollama model
python main.py URL --whisper small
python main.py URL --output ./output
python main.py --help
```

More:

```bash
python main.py URL --clips 3                       # exactly three strongest moments
python main.py URL --beam-size 1                   # faster transcription; may slightly reduce accuracy
python main.py URL --duration 45                   # window widens to 40-50 s
python main.py URL --min-duration 20 --max-duration 40
python main.py URL --skip-intro 120 --skip-outro 60  # ignore intro/outro
python main.py URL --whisper medium --device cuda
python main.py URL --language en                   # skip language detection
python main.py URL --no-subtitles                  # no captions at all
python main.py URL --no-burn-in                    # .srt/.vtt only, clean video
python main.py URL --font-size 60 --subtitle-position center
python main.py URL --no-smart-crop                 # always centre-crop 9:16
python main.py URL --keep-temp --debug             # keep intermediates, verbose
python main.py URL --no-cache                      # redo download/transcript/analysis
python main.py URL --json                          # machine-readable summary
python main.py --check                             # dependency check only
```

## 11. Output structure

```
output/
└── How_data_quality_beats_compute-dQw4w9WgXcQ/
    ├── clip_01/
    │   ├── original_clip.mp4    exact cut, no reframing, no captions
    │   ├── youtube_16x9.mp4     1920x1080, captions burned in
    │   └── metadata.json       title, description, hashtags, timings
    └── clip_02/                 one numbered directory per clip
```

By default the app requests about five clips per source hour (five from one
hour, ten from two hours), capped at 30. Set `clip.auto_count: false` in
`config.yaml` to use the fixed `clip.clips` count instead. `--clips N` always
requests exactly N clips and overrides either mode. Every clip, including a
single-clip run, gets its own numbered directory. Existing exports are never
overwritten: a repeat run of the same video gets a timestamped directory.

`metadata.json`:

```json
{
  "source_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "source_title": "How data quality beats compute",
  "source_duration": 6138.0,
  "clip_start": 812.4,
  "clip_end": 842.1,
  "clip_duration": 29.7,
  "title": "Data quality beats compute",
  "description": "Why a single laptop gets you most of the way...",
  "hashtags": ["#shorts", "#ai", "#machinelearning"],
  "hook": "Most people think training a model needs a huge cluster of GPUs.",
  "reason": "A contrarian claim backed by a concrete measurement, fully self-contained.",
  "score": 9.1,
  "generated_at": "2026-11-18T09:14:02+00:00"
}
```

`clip_analysis.json` additionally contains the seven per-dimension scores, the
weights used, the boundary adjustments that were made, and the top candidates
that lost — so a selection can always be audited.

The working directory is separate from the output:

```
workspace/
└── dQw4w9WgXcQ/              one directory per video id
    ├── source/               downloaded video + source.json marker
    ├── audio/                16 kHz mono WAV  (deleted after success)
    ├── transcript/           transcript.json  (kept: enables resume)
    ├── analysis/             candidates.json, selection.json (kept: cache)
    ├── clips/                cut segments     (deleted after success)
    ├── subtitles/            per-clip .srt/.vtt/.ass
    └── clipper.log           full debug log of the run
```

Re-running the same URL reuses the download, audio, transcript and LLM analysis,
so a crash during rendering costs seconds, not another transcription. Use
`--no-cache` to force a fresh pass, `--keep-temp` to keep every intermediate.

## 12. Troubleshooting

**`FFmpeg was not found`**
Install it ([section 3](#3-ffmpeg-installation)) and reopen the terminal, or set
`FFMPEG_PATH` / `FFPROBE_PATH` in `.env`.

**`Cannot reach Ollama at http://localhost:11434`**
Run `ollama serve`. If it listens elsewhere, set `OLLAMA_BASE_URL`. On Windows
check the Ollama tray icon is running.

**Ollama says the selected model is not installed**
Run `ollama pull <model-name>` for the chosen model. `ollama list` shows what you have.

**`pip install faster-whisper` fails / no `ctranslate2` wheel**
You are on a too-new Python. Use 3.10–3.12 (`py -3.12 -m venv .venv`).

**`Download failed` / HTTP 403 / "Sign in to confirm you're not a bot"**
Update yt-dlp first: `pip install -U yt-dlp`. For age-restricted or
rate-limited videos, borrow browser cookies:
`YTDLP_COOKIES_FROM_BROWSER=chrome` (also `firefox`, `edge`).

**`This URL is a currently live stream`**
Wait until the stream ends, then use the `/live/` or `watch?v=` VOD URL.

**`The transcript is empty - no speech was detected`**
Music-only or silent video, or a wrong language guess. Try
`--whisper medium --language en`.

**Captions are not burned in / wrong font**
The font must exist on the machine. On Windows use `font_name: Arial` (or
`Impact`, `Bebas Neue`); on Linux install `fonts-dejavu`. Burn-in is skipped
with a warning rather than failing the render.

**Boxes instead of characters in burned captions**
Pick a font that covers the language (`Noto Sans`, `Arial Unicode MS`). The
`.srt`/`.vtt` files are unaffected.

**`The model did not return valid JSON` warnings**
Harmless: the response is repaired and retried, and the pipeline falls back to
heuristic scores if a batch keeps failing. A larger model (`qwen2.5:14b`) makes
it rare. You can also lower `ollama.temperature`.

**Transcription is very slow**
Expect roughly real-time or slower on CPU with `small`. Use a smaller model
(`base`), or a GPU ([section 13](#13-gpu-acceleration)).

**Out of memory while transcribing on GPU**
Set `WHISPER_COMPUTE_TYPE=int8_float16`, or use a smaller model.

**Everything is slow to encode**
The default `video.preset: veryfast` prioritizes speed. Choose
`video.preset: medium` in `config.yaml` for smaller files at slower encoding
speed; `video_codec: h264_nvenc` offloads to an NVIDIA GPU.

**Where are the logs?** `workspace/<video_id>/clipper.log`, always at DEBUG level.

## 13. GPU acceleration

Transcription uses CUDA automatically when it is usable. `device: auto` probes
CTranslate2 (and torch, if installed) and falls back to CPU silently, so the
same command works on any machine.

Requirements: NVIDIA driver, CUDA 12 runtime, and cuDNN 9 for CTranslate2 4.x.
Easiest route inside the venv:

```bash
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

If CTranslate2 cannot find Python-installed libraries on Linux, set their
library directories in `LD_LIBRARY_PATH` in the same shell that runs the app:

```bash
NVIDIA_LIBS="$(python -c 'import site,pathlib; roots=[pathlib.Path(p)/"nvidia" for p in site.getsitepackages()]; print(":".join(str(x) for r in roots for x in (r/"cublas"/"lib",r/"cudnn"/"lib") if x.is_dir()))')"
export LD_LIBRARY_PATH="$NVIDIA_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python main.py URL --device cuda
```

Run dependency installation and the app from the same activated Python
environment. The requirements pin `faster-whisper` to 1.0.3 and PyAV below
19 because that release passes PyAV's removed `metadata_errors` argument.
After dependency changes, activate the app environment and run
`python -m pip install -r requirements.txt` there.

Force it explicitly:

```bash
python main.py URL --device cuda --whisper large-v3
```

```yaml
whisper:
  device: cuda
  compute_type: float16      # int8_float16 uses less VRAM
```

Rough VRAM: `small` ~1 GB, `medium` ~2.5 GB, `large-v3` ~5 GB with `float16`.

Video encoding can use the GPU too:

```yaml
video:
  video_codec: h264_nvenc    # NVIDIA;  h264_qsv (Intel), h264_amf (AMD)
  preset: p4
```

Ollama uses your GPU on its own if it can — the app does not need to configure it.

## 14. CPU fallback

Everything runs on CPU with `--device cpu`. The defaults (`base` Whisper,
`qwen3:4b`, `libx264 veryfast`) are chosen for a normal consumer machine. For a faster
CPU run:

```bash
python main.py URL --whisper base --model qwen2.5:3b --max-candidates 20
python main.py URL --device cpu
```

```yaml
video:
  preset: veryfast
  crf: 23
whisper:
  compute_type: int8        # fastest on CPU
  beam_size: 1              # less accurate, quicker
```

## 15. How to change the LLM

Any Ollama model that can follow a JSON instruction works. The code never
assumes Qwen specifically.

```bash
ollama pull qwen2.5:14b
python main.py URL --model qwen2.5:14b
```

```
OLLAMA_MODEL=qwen2.5:14b
```

```yaml
ollama:
  model: qwen2.5:14b
  temperature: 0.2
  num_ctx: 8192
  candidates_per_request: 12  # use 6 for small-context models
```

To use a different machine on your network:

```bash
python main.py URL --ollama-url http://192.168.1.50:11434
```

Smaller models sometimes produce malformed JSON. `app/llm_analyzer.py` repairs
and retries, and keeps heuristic scores if a batch cannot be salvaged, so a weak
model degrades the choice rather than breaking the run.

How the scoring is weighted is configuration, not code:

```yaml
scoring:
  hook: 0.23
  interestingness: 0.21
  information_value: 0.08
  emotional_impact: 0.16
  shareability: 0.16
  clarity: 0.06
  standalone_context: 0.10
```

The prompts themselves live in `prompts/*.txt` — plain text, edit freely:

| File | Stage |
|---|---|
| `prompts/candidate_analysis.txt` | scores every candidate on the seven dimensions |
| `prompts/final_selection.txt` | compares finalists and picks the winner |
| `prompts/metadata_generation.txt` | title, post caption, hashtags |

## 16. How to change video formats

```bash
python main.py URL --format 9:16     # vertical only
python main.py URL --format 16:9     # landscape only
python main.py URL --format both     # default
```

```
VIDEO_FORMATS=16:9,9:16
```

```yaml
video:
  formats: ["16:9", "9:16"]
  landscape_resolution: 1920x1080
  vertical_resolution: 1080x1920     # e.g. 720x1280 for a lighter file
```

Aliases accepted: `landscape`, `youtube`, `horizontal`, `vertical`, `shorts`,
`reels`, `tiktok`, `both`.

The 9:16 render never squashes the picture:

1. sample frames from the chosen clip,
2. detect faces (OpenCV) and compute a weighted subject centre,
3. crop a 9:16 window around the subject, clamped inside the frame,
4. fall back to a centre crop when nothing is detected or OpenCV is missing,
5. scale to the target resolution with Lanczos,
6. keep captions inside the safe area via `subtitles.margin_ratio`.

A source that is already vertical or square is padded instead of cropped, and a
vertical source rendered to 16:9 gets a blurred pillarbox background rather than
black bars.

## 17. How clip selection works

The LLM never sees the whole transcript at once, and the first 30 seconds are
never chosen by default.

**Stage 1 — candidate windows (`app/candidate_generator.py`)**
Overlapping windows are built from transcript *segment boundaries* across the
entire video, every duration in the allowed range. Each gets objective
metadata: word count, words/second, silence ratio, whether it starts and ends on
a sentence. A deterministic pre-score then drops obvious junk — sponsor reads,
promo codes, "like and subscribe", greetings, intros, dead air, filler-heavy
delivery, repetition, mid-sentence starts — and boosts concrete figures,
questions and contrarian phrasing. The best ~36, spread across the runtime, go
forward.

**Stage 2 — Qwen scoring (`prompts/candidate_analysis.txt`)**
Candidates are sent in batches of 12 by default and scored 0–10 on `hook`,
`interestingness`, `information_value`, `emotional_impact`, `standalone_context`,
`shareability` and `clarity`. The prompt specifically judges stop-scroll
strength, retention through a payoff, standalone clarity and a real reason to
share. The weighted blend above puts extra weight on those viewer responses.
The resulting score is an editorial ranking, not a promise of virality.

**Stage 3 — final selection (`prompts/final_selection.txt`)**
The top finalists go back to the model with their transcripts and scores; it
picks the winner (or the N best, non-overlapping) and must justify the choice in
strict JSON. Every field is validated: unknown ids, bad numbers and malformed
JSON are rejected, and the pipeline falls back to the highest weighted score
rather than trusting bad output. Nothing the model emits is ever executed.

**Stage 4 — exact boundaries (`app/clip_selector.py`)**
The model's timestamps are advisory. Final in/out points are snapped to sentence
boundaries from the transcript, scored against the duration target, and clamped
to `[min, max]`: a 34 s candidate is trimmed at the last clean sentence end, and
a 27 s candidate is kept as-is rather than padded into nonsense. Small lead-in /
tail-out padding is then applied without breaking the limits.

## 18. Project structure

```
ai-video-clipper/
├── main.py                       CLI
├── requirements.txt
├── .env.example
├── config.yaml
├── README.md
├── app/
│   ├── config.py                 layered config + validation
│   ├── logger.py                 console + file logging
│   ├── errors.py                 typed, actionable exceptions
│   ├── utils.py                  URL/time/filename helpers
│   ├── media.py                  safe FFmpeg/FFprobe wrappers
│   ├── preflight.py              dependency checks before any work
│   ├── downloader.py             yt-dlp stage (+ download cache)
│   ├── transcription.py          faster-whisper stage (+ transcript cache)
│   ├── transcript.py             transcript model and queries
│   ├── candidate_generator.py    windows + objective pre-scoring
│   ├── llm_analyzer.py           Ollama client, JSON repair, prompt stages
│   ├── clip_selector.py          timestamp validation + boundary snapping
│   ├── subtitle_generator.py     SRT / VTT / styled ASS
│   ├── video_processor.py        cutting, smart crop, rendering
│   ├── metadata_generator.py     metadata.json / clip_analysis.json
│   └── pipeline.py               orchestration + caching
├── prompts/
│   ├── candidate_analysis.txt
│   ├── final_selection.txt
│   └── metadata_generation.txt
├── tests/                        148 unit + integration tests
├── output/
└── workspace/
```

Safety properties worth knowing: FFmpeg is always invoked with an argument
vector (`shell=False`); LLM output is parsed as data and never executed;
filenames derived from titles or model output are sanitised against path
traversal; filter arguments are escaped, and subtitle burn-in runs with the
working directory set to the subtitle file so Windows drive letters cannot break
the filter graph.

## 19. Tests

```bash
pip install pytest
python -m pytest tests -q
```

No network, FFmpeg, GPU or Ollama needed — external stages are stubbed.

```
148 passed
```

Covered: configuration loading and layering, URL validation, filename
sanitisation, timestamp formatting and validation, transcript parsing and
round-tripping, candidate generation and heuristic scoring, score weighting,
malformed/truncated/single-quoted LLM JSON, LLM fallbacks on failure, clip
duration fitting and boundary snapping, caption wrapping and ASS styling,
reframing filter graphs, metadata shape, and a full stubbed pipeline run
(output layout, caching, multi-clip, cleanup).

## 20. Local dashboard

The dashboard is a local Flask page that calls the same pipeline and shares its
configuration, cache, workspace, and output directories. It allows one active
job at a time and stores job status history in `workspace/dashboard_jobs.json`.
Install dependencies in the same Python environment used for the CLI, then run:

```bash
python -m pip install -r requirements.txt
python -m app.dashboard
```

Open <http://127.0.0.1:5000>. The dashboard does not publish or upload clips.
It binds to localhost by default; use `--config`, `--env`, `--host`, or `--port`
to select local settings or a different local port.

## 21. Extending the pipeline

Each stage is a class with a narrow interface, so these drop in without
reworking the pipeline:

| Idea | Where |
|---|---|
| multiple clips per video | already supported: `--clips 5` |
| batch processing | loop over URLs calling `Pipeline.run` |
| silence removal | extra filter in `VideoProcessor.render` |
| audio enhancement | add to the `-af` chain in `render` |
| background music | second input + `amix` in `render` |
| face tracking (per-frame) | replace `detect_subject_center` with a tracker |
| speaker detection | add diarisation to `Transcriber`, store on `Segment` |
| B-roll / thumbnails | new stage after `_produce_clip` |
| automatic publishing | consume `metadata.json` in a new module |
| web UI | import `Pipeline` and call `run()`; it returns a `JobResult` |

A new LLM stage only needs a prompt file in `prompts/` and a method on
`LLMAnalyzer`; `PromptLibrary` handles loading and `{{placeholder}}` filling.
