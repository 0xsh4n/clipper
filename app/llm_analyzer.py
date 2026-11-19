"""Ollama local LLM integration.

Two independent pieces live here:

* :class:`OllamaClient` - a dependency-free HTTP client (stdlib ``urllib``)
  with health checks, retries and strict JSON handling.
* :class:`LLMAnalyzer` - the prompt-driven stages: candidate scoring, final
  clip selection and publishing metadata generation.

The LLM only ever returns *data*. Nothing it produces is executed, and every
field is validated before use.
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .candidate_generator import SCORE_KEYS, Candidate, compute_overall, normalise_scores
from .config import AppConfig, OllamaConfig
from .errors import DependencyError, LLMError
from .logger import get_logger
from .utils import clamp, dedupe_preserve_order, format_clock

log = get_logger("llm")

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)\s*```", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


# --------------------------------------------------------------------------- #
# JSON handling
# --------------------------------------------------------------------------- #
def extract_json(text: str) -> Any:
    """Parse JSON out of a noisy LLM response.

    Tries, in order: direct parse, fenced code block, first balanced
    object/array, and a light repair pass (trailing commas, truncated tail).
    Raises :class:`LLMError` when nothing parses.
    """

    if text is None:
        raise LLMError("The model returned an empty response.")
    raw = str(text).strip()
    if not raw:
        raise LLMError("The model returned an empty response.")

    for candidate in _json_candidates(raw):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            repaired = repair_json(candidate)
            if repaired is None:
                continue
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                continue
    raise LLMError(
        "The model did not return valid JSON.",
        f"First 300 characters of the response: {raw[:300]!r}",
    )


def _json_candidates(raw: str) -> List[str]:
    out: List[str] = [raw]
    for match in _FENCE_RE.findall(raw):
        if match.strip():
            out.append(match.strip())
    for opener, closer in (("{", "}"), ("[", "]")):
        block = _balanced_block(raw, opener, closer)
        if block:
            out.append(block)
    # Longest first: a full object beats a nested fragment.
    return dedupe_preserve_order(sorted(out, key=len, reverse=True))


def _balanced_block(raw: str, opener: str, closer: str) -> Optional[str]:
    start = raw.find(opener)
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(raw)):
        char = raw[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return raw[start : index + 1]
    # Unterminated - return the tail so repair_json can close it.
    return raw[start:]


def repair_json(text: str) -> Optional[str]:
    """Best-effort repair of common LLM JSON mistakes."""

    if not text:
        return None
    repaired = text.strip()
    repaired = _FENCE_RE.sub(r"\1", repaired)
    # Strip prose before the first structural character.
    openers = [repaired.find(opener) for opener in ("{", "[")]
    starts = [index for index in openers if index >= 0]
    if starts and min(starts) > 0:
        repaired = repaired[min(starts):]
    repaired = repaired.replace("“", '"').replace("”", '"')
    repaired = repaired.replace("‘", "'").replace("’", "'")
    repaired = re.sub(r"//[^\n\r]*", "", repaired)
    repaired = _TRAILING_COMMA_RE.sub(r"\1", repaired)
    repaired = re.sub(r"\bNaN\b|\bInfinity\b|\b-Infinity\b", "null", repaired)
    repaired = re.sub(r"\bTrue\b", "true", repaired)
    repaired = re.sub(r"\bFalse\b", "false", repaired)
    repaired = re.sub(r"\bNone\b", "null", repaired)
    if "'" in repaired and '"' not in repaired:
        # Python-style dict output: ' is the only quote character in use.
        repaired = repaired.replace("'", '"')

    # Close unbalanced braces/brackets (truncated generation).
    depth_curly = 0
    depth_square = 0
    in_string = False
    escaped = False
    for char in repaired:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth_curly += 1
        elif char == "}":
            depth_curly -= 1
        elif char == "[":
            depth_square += 1
        elif char == "]":
            depth_square -= 1
    if in_string:
        repaired += '"'
    if depth_curly <= 0 and depth_square <= 0:
        # Balanced already: drop any prose that follows the JSON value.
        last = max(repaired.rfind("}"), repaired.rfind("]"))
        if last >= 0:
            repaired = repaired[: last + 1]
    repaired = _TRAILING_COMMA_RE.sub(r"\1", repaired.rstrip().rstrip(","))
    repaired += "]" * max(0, depth_square)
    repaired += "}" * max(0, depth_curly)
    return repaired


def coerce_number(value: Any, default: Optional[float] = None) -> Optional[float]:
    """Pull a float out of ``9.1``, ``"9.1"``, ``"9,1"`` or ``"13:32"``."""

    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        number = float(value)
        return number if number == number else default  # reject NaN
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        if re.fullmatch(r"\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?", text):
            parts = [float(p) for p in text.split(":")]
            total = 0.0
            for part in parts:
                total = total * 60 + part
            return total
        text = text.replace(",", ".")
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        if match:
            return float(match.group(0))
    return default


# --------------------------------------------------------------------------- #
# Ollama HTTP client
# --------------------------------------------------------------------------- #
@dataclass
class LLMResponse:
    text: str
    duration_ms: int = 0
    model: str = ""


class OllamaClient:
    """Minimal Ollama REST client (``/api/version``, ``/api/tags``, ``/api/generate``)."""

    def __init__(self, config: OllamaConfig) -> None:
        self.config = config
        self.base_url = str(config.base_url).rstrip("/")
        self.model = str(config.model)

    # -- health ----------------------------------------------------------- #
    def verify(self) -> Dict[str, str]:
        """Check Ollama is reachable and the model is pulled."""

        try:
            version_payload = self._request("GET", "/api/version", timeout=10)
        except LLMError as exc:
            raise DependencyError(
                f"Cannot reach Ollama at {self.base_url}.",
                "Start it with `ollama serve`, then check OLLAMA_BASE_URL.\n"
                f"     Underlying error: {exc.message}",
            ) from exc

        version = str(version_payload.get("version") or "")
        models = self.list_models()
        if not self._model_available(self.model, models):
            available = ", ".join(sorted(models)[:12]) or "none"
            raise DependencyError(
                f"Ollama model {self.model!r} is not installed.",
                f"Pull it with: ollama pull {self.model}\n"
                f"     Installed models: {available}",
            )
        return {"version": version, "model": self.model}

    def list_models(self) -> List[str]:
        payload = self._request("GET", "/api/tags", timeout=15)
        models: List[str] = []
        for entry in payload.get("models") or []:
            name = str(entry.get("name") or entry.get("model") or "").strip()
            if name:
                models.append(name)
        return models

    @staticmethod
    def _model_available(wanted: str, models: Sequence[str]) -> bool:
        wanted = wanted.strip()
        if not wanted:
            return False
        normalised = {m.strip() for m in models}
        if wanted in normalised:
            return True
        # "qwen2.5" should match "qwen2.5:latest" / "qwen2.5:7b".
        base = wanted.split(":", 1)[0]
        return any(m == wanted or m.split(":", 1)[0] == base for m in normalised)

    # -- generation ------------------------------------------------------- #
    def generate(
        self,
        prompt: str,
        *,
        json_mode: bool = True,
        think: Optional[bool] = None,
        temperature: Optional[float] = None,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        body: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": float(self.config.temperature if temperature is None else temperature),
                "num_ctx": int(self.config.num_ctx),
                "top_p": 0.9,
                "repeat_penalty": 1.05,
            },
        }
        if json_mode:
            body["format"] = "json"
        if think is not None:
            body["think"] = think
        if system:
            body["system"] = system
        if max_tokens:
            body["options"]["num_predict"] = int(max_tokens)

        payload = self._request("POST", "/api/generate", body=body, timeout=self.config.timeout)
        text = str(payload.get("response") or "")
        if not text.strip():
            raise LLMError(
                "Ollama returned an empty completion.",
                "Try a different model size, or lower ollama.num_ctx.",
            )
        return LLMResponse(
            text=text,
            duration_ms=int(float(payload.get("total_duration") or 0) / 1e6),
            model=str(payload.get("model") or self.model),
        )

    def generate_json(
        self,
        prompt: str,
        *,
        repair_prompt: Optional[str] = None,
        think: Optional[bool] = None,
        temperature: Optional[float] = None,
        system: Optional[str] = None,
    ) -> Any:
        """Generate and parse JSON, retrying with a repair instruction."""

        attempts = max(1, int(self.config.max_retries))
        last_error: Optional[Exception] = None
        current_prompt = prompt

        for attempt in range(1, attempts + 1):
            try:
                response = self.generate(
                    current_prompt,
                    json_mode=True,
                    think=think,
                    temperature=temperature if attempt == 1 else 0.0,
                    system=system,
                )
                return extract_json(response.text)
            except LLMError as exc:
                last_error = exc
                log.warning("LLM JSON attempt %d/%d failed: %s", attempt, attempts, exc.message)
                if "timed out" in exc.message.lower() or "timeout" in exc.message.lower():
                    raise
                if attempt < attempts:
                    current_prompt = (
                        (repair_prompt or prompt)
                        + "\n\nYour previous answer was not valid JSON. "
                        "Reply with ONLY one valid JSON object. No prose, no markdown fences."
                    )
                    time.sleep(min(2.0 * attempt, 5.0))
        raise LLMError(
            "The model failed to produce valid JSON after retries.",
            str(last_error) if last_error else "",
        )

    # -- transport -------------------------------------------------------- #
    def _request(
        self,
        method: str,
        path: str,
        *,
        body: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        attempts = max(1, int(self.config.max_retries))
        last_error: Optional[Exception] = None

        for attempt in range(1, attempts + 1):
            request = urllib.request.Request(url, data=data, headers=headers, method=method)
            try:
                with urllib.request.urlopen(request, timeout=timeout or self.config.timeout) as response:
                    payload = response.read().decode("utf-8", errors="replace")
                try:
                    parsed = json.loads(payload or "{}")
                except json.JSONDecodeError as exc:
                    raise LLMError(f"Ollama returned malformed JSON from {path}.") from exc
                if not isinstance(parsed, dict):
                    raise LLMError(f"Ollama returned an unexpected payload from {path}.")
                return parsed
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:400]
                if exc.code == 404 and "model" in detail.lower():
                    raise DependencyError(
                        f"Ollama does not have the model {self.model!r}.",
                        f"Pull it with: ollama pull {self.model}",
                    ) from exc
                last_error = LLMError(f"Ollama HTTP {exc.code} on {path}: {detail}")
                if exc.code < 500:
                    raise last_error from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                reason = getattr(exc, "reason", exc)
                if isinstance(reason, TimeoutError) or "timed out" in str(reason).lower():
                    raise LLMError(
                        f"Ollama request timed out after {timeout or self.config.timeout}s.",
                        "The server is reachable but generation is too slow. Try a smaller candidate batch or a lower-context model.",
                    ) from exc
                last_error = LLMError(f"Could not reach Ollama at {url}: {exc}")
            if attempt < attempts:
                time.sleep(min(2.0 * attempt, 6.0))

        assert last_error is not None
        raise last_error


# --------------------------------------------------------------------------- #
# prompt-driven analysis
# --------------------------------------------------------------------------- #
class PromptLibrary:
    """Loads prompt templates and fills ``{{placeholders}}``."""

    def __init__(self, prompts_dir: Path) -> None:
        self.prompts_dir = Path(prompts_dir)
        self._cache: Dict[str, str] = {}

    def render(self, name: str, **values: Any) -> str:
        template = self._load(name)
        for key, value in values.items():
            template = template.replace("{{" + key + "}}", str(value))
        return template

    def _load(self, name: str) -> str:
        if name in self._cache:
            return self._cache[name]
        path = self.prompts_dir / f"{name}.txt"
        if not path.is_file():
            raise LLMError(
                f"Prompt template not found: {path}",
                "The prompts/ directory must sit next to main.py.",
            )
        text = path.read_text(encoding="utf-8")
        self._cache[name] = text
        return text


class LLMAnalyzer:
    """Stage 2/3: score candidates, choose the final clip, write metadata."""

    def __init__(self, config: AppConfig, client: Optional[OllamaClient] = None) -> None:
        self.config = config
        self.client = client or OllamaClient(config.ollama)
        self.prompts = PromptLibrary(config.prompts_dir)

    # -- stage 2: scoring ------------------------------------------------- #
    def score_candidates(self, candidates: Sequence[Candidate], context: Dict[str, Any]) -> List[Candidate]:
        batch_size = max(1, int(self.config.ollama.candidates_per_request))
        batches = [list(candidates[i : i + batch_size]) for i in range(0, len(candidates), batch_size)]
        log.info(
            "Analyzing %d candidates with %s (%d batches)",
            len(candidates),
            self.config.ollama.model,
            len(batches),
        )

        for index, batch in enumerate(batches, start=1):
            try:
                self._score_batch(batch, context)
            except LLMError as exc:
                log.warning(
                    "Batch %d/%d could not be scored by the LLM (%s) - keeping heuristic scores",
                    index,
                    len(batches),
                    exc.message,
                )
                for candidate in batch:
                    self._apply_heuristic_fallback(candidate)
            log.info("Scored batch %d/%d", index, len(batches))

        scored = sorted(candidates, key=lambda c: -c.overall)
        for candidate in scored[:5]:
            log.debug(
                "candidate %s overall=%.2f hook=%.1f title=%r",
                candidate.label,
                candidate.overall,
                candidate.scores.get("hook", 0.0),
                candidate.title,
            )
        return list(scored)

    def _score_batch(self, batch: Sequence[Candidate], context: Dict[str, Any]) -> None:
        payload = [
            {
                "id": candidate.id,
                "start": round(candidate.start, 2),
                "end": round(candidate.end, 2),
                "duration": round(candidate.duration, 2),
                "word_count": candidate.word_count,
                "words_per_second": round(candidate.speech_density, 2),
                "silence_ratio": round(candidate.silence_ratio, 2),
                "starts_mid_sentence": not candidate.starts_sentence,
                "ends_mid_sentence": not candidate.ends_sentence,
                "transcript": candidate.text,
            }
            for candidate in batch
        ]
        prompt = self.prompts.render(
            "candidate_analysis",
            video_title=context.get("title", ""),
            video_duration=context.get("duration_label", ""),
            language=context.get("language", ""),
            target_duration=self.config.clip.target_duration,
            candidate_count=len(batch),
            candidates=json.dumps(payload, ensure_ascii=False, indent=2),
        )
        data = self.client.generate_json(prompt, think=False)
        entries = self._as_entries(data, keys=("scores", "candidates", "results", "analysis"))
        by_id = {candidate.id: candidate for candidate in batch}
        matched = 0

        for entry in entries:
            if not isinstance(entry, dict):
                continue
            raw_id = coerce_number(entry.get("id", entry.get("candidate_id")), None)
            if raw_id is None:
                continue
            candidate = by_id.get(int(raw_id))
            if candidate is None:
                continue
            scores = normalise_scores(
                {key: entry.get(key) for key in SCORE_KEYS if entry.get(key) is not None}
            )
            if not scores:
                continue
            candidate.scores = scores
            candidate.overall = compute_overall(scores, self.config.scoring)
            candidate.title = _clean_text(entry.get("title"), 120)
            candidate.hook = _clean_text(entry.get("hook_line") or entry.get("hook_text"), 200)
            candidate.reason = _clean_text(entry.get("reason") or entry.get("notes"), 600)
            candidate.scored_by_llm = True
            matched += 1

        for candidate in batch:
            if not candidate.scored_by_llm:
                self._apply_heuristic_fallback(candidate)
        if matched == 0:
            raise LLMError("The model returned no usable candidate scores.")

    def _apply_heuristic_fallback(self, candidate: Candidate) -> None:
        """Keep the pipeline moving when the LLM skips a candidate."""

        if candidate.scored_by_llm:
            return
        base = clamp(candidate.heuristic_score, 0.0, 10.0)
        candidate.scores = {key: base for key in SCORE_KEYS}
        candidate.overall = compute_overall(candidate.scores, self.config.scoring)
        if not candidate.reason:
            candidate.reason = "Scored by heuristics only (the model did not return a score)."

    # -- stage 3: final selection ----------------------------------------- #
    def select_final(
        self,
        candidates: Sequence[Candidate],
        context: Dict[str, Any],
        *,
        wanted: int = 1,
        shortlist_size: int = 8,
    ) -> List[Dict[str, Any]]:
        """Ask the model to choose between the best candidates.

        Returns a list of validated dicts: ``{id, title, hook, reason, score}``.
        """

        ranked = sorted(candidates, key=lambda c: -c.overall)[: max(wanted, shortlist_size)]
        if not ranked:
            return []

        payload = [
            {
                "id": candidate.id,
                "start": round(candidate.start, 2),
                "end": round(candidate.end, 2),
                "duration": round(candidate.duration, 2),
                "timecode": candidate.label,
                "scores": {k: round(v, 1) for k, v in candidate.scores.items()},
                "overall": round(candidate.overall, 2),
                "starts_mid_sentence": not candidate.starts_sentence,
                "ends_mid_sentence": not candidate.ends_sentence,
                "transcript": candidate.text,
            }
            for candidate in ranked
        ]
        prompt = self.prompts.render(
            "final_selection",
            video_title=context.get("title", ""),
            video_duration=context.get("duration_label", ""),
            target_duration=self.config.clip.target_duration,
            min_duration=self.config.clip.min_duration,
            max_duration=self.config.clip.max_duration,
            wanted=wanted,
            candidate_count=len(payload),
            candidates=json.dumps(payload, ensure_ascii=False, indent=2),
        )

        try:
            data = self.client.generate_json(prompt, think=True)
        except LLMError as exc:
            log.warning("Final selection failed (%s) - falling back to the top scored candidate", exc.message)
            return [self._fallback_selection(candidate) for candidate in ranked[:wanted]]

        valid_ids = {candidate.id for candidate in ranked}
        selections = self._parse_selections(data, valid_ids)
        if not selections:
            log.warning("The model's final answer had no valid candidate id - using the top scored candidate")
            return [self._fallback_selection(candidate) for candidate in ranked[:wanted]]

        # Keep the model's ranking, drop duplicates, pad from our own ranking.
        out: List[Dict[str, Any]] = []
        used: set[int] = set()
        for selection in selections:
            if selection["id"] in used:
                continue
            used.add(selection["id"])
            out.append(selection)
            if len(out) >= wanted:
                break
        for candidate in ranked:
            if len(out) >= wanted:
                break
            if candidate.id in used:
                continue
            used.add(candidate.id)
            out.append(self._fallback_selection(candidate))
        return out

    def _parse_selections(self, data: Any, valid_ids: set[int]) -> List[Dict[str, Any]]:
        entries = self._as_entries(
            data, keys=("selections", "clips", "ranked", "results", "selected_clips")
        )
        if not entries and isinstance(data, dict) and ("id" in data or "candidate_id" in data):
            entries = [data]

        out: List[Dict[str, Any]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            raw_id = coerce_number(entry.get("id", entry.get("candidate_id")), None)
            if raw_id is None:
                continue
            candidate_id = int(raw_id)
            if candidate_id not in valid_ids:
                log.debug("Ignoring selection for unknown candidate id %s", candidate_id)
                continue
            out.append(
                {
                    "id": candidate_id,
                    "title": _clean_text(entry.get("title"), 120),
                    "hook": _clean_text(entry.get("hook") or entry.get("hook_line"), 200),
                    "reason": _clean_text(entry.get("reason") or entry.get("why"), 800),
                    "score": clamp(coerce_number(entry.get("score"), 0.0) or 0.0, 0.0, 10.0),
                    # Advisory only: boundaries are recomputed from the transcript.
                    "llm_start": coerce_number(entry.get("start"), None),
                    "llm_end": coerce_number(entry.get("end"), None),
                }
            )
        return out

    @staticmethod
    def _fallback_selection(candidate: Candidate) -> Dict[str, Any]:
        return {
            "id": candidate.id,
            "title": candidate.title,
            "hook": candidate.hook,
            "reason": candidate.reason or "Highest weighted score across all scored candidates.",
            "score": round(candidate.overall, 2),
            "llm_start": None,
            "llm_end": None,
            "fallback": True,
        }

    # -- metadata --------------------------------------------------------- #
    def generate_publishing_metadata(
        self,
        clip_text: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Title, description and hashtags for the finished clip."""

        prompt = self.prompts.render(
            "metadata_generation",
            video_title=context.get("title", ""),
            channel=context.get("uploader", ""),
            clip_duration=round(float(context.get("duration", 0.0)), 1),
            hook=context.get("hook", ""),
            reason=context.get("reason", ""),
            transcript=clip_text,
        )
        try:
            data = self.client.generate_json(prompt, think=False)
        except LLMError as exc:
            log.warning("Metadata generation failed (%s) - using transcript-derived fallbacks", exc.message)
            return self._fallback_metadata(clip_text, context)

        if not isinstance(data, dict):
            return self._fallback_metadata(clip_text, context)

        hashtags = _clean_hashtags(data.get("hashtags"))
        title = _clean_text(data.get("title"), 100)
        caption = _clean_text(data.get("caption") or data.get("description"), 400)
        description = caption
        fallback = self._fallback_metadata(clip_text, context)
        title = title or fallback["title"]
        description = description or fallback["description"]
        caption = caption or description
        hashtags = hashtags or fallback["hashtags"]
        return {
            "title": title,
            "caption": caption,
            "description": description,
            "hashtags": hashtags,
            "alt_titles": [t for t in (_clean_text(x, 100) for x in (data.get("alt_titles") or [])) if t][:5],
        }

    @staticmethod
    def _fallback_metadata(clip_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        hook = str(context.get("hook") or "").strip()
        first_sentence = re.split(r"(?<=[.!?])\s+", clip_text.strip())[0] if clip_text.strip() else ""
        title = (hook or first_sentence or str(context.get("title") or "Clip"))[:95].strip()
        description = (clip_text.strip()[:400] or title)
        stopwords = {
            "about", "after", "again", "also", "because", "been", "being", "could", "does",
            "doing", "from", "have", "here", "into", "just", "like", "make", "more", "most",
            "much", "only", "other", "over", "really", "said", "some", "such", "than", "that",
            "them", "then", "there", "these", "they", "thing", "this", "those", "through", "very",
            "want", "what", "when", "where", "which", "while", "will", "with", "would", "your",
        }
        terms = [
            word for word, _ in Counter(
                word for word in re.findall(r"[^\W_]{4,}", clip_text.lower(), flags=re.UNICODE)
                if word not in stopwords
            ).most_common(8)
        ]
        hashtags = _clean_hashtags(["#shorts", *(f"#{word}" for word in terms[:4])])
        return {
            "title": title,
            "caption": description,
            "description": description,
            "hashtags": hashtags or ["#shorts"],
            "alt_titles": [],
        }

    # -- shared ----------------------------------------------------------- #
    @staticmethod
    def _as_entries(data: Any, keys: Sequence[str]) -> List[Any]:
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in keys:
                value = data.get(key)
                if isinstance(value, list):
                    return value
                if isinstance(value, dict):
                    return [value]
            # Some models answer {"0": {...}, "1": {...}}.
            if data and all(str(k).isdigit() for k in data):
                return list(data.values())
        return []


def _clean_text(value: Any, limit: int) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        value = " ".join(str(v) for v in value)
    text = re.sub(r"\s+", " ", str(value)).strip().strip('"').strip()
    return text[:limit].strip()


def _clean_hashtags(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = re.split(r"[\s,]+", value)
    elif isinstance(value, (list, tuple)):
        items = [str(v) for v in value]
    else:
        return []
    out: List[str] = []
    for item in items:
        tag = re.sub(r"[^0-9A-Za-z_]+", "", str(item))
        if not tag:
            continue
        out.append("#" + tag[:40])
    return dedupe_preserve_order(out)[:12]
