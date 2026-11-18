"""LLM stages with a stubbed Ollama client (no network, no model needed)."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from app.candidate_generator import Candidate
from app.config import OllamaConfig, load_config
from app.errors import DependencyError, LLMError
from app.llm_analyzer import LLMAnalyzer, OllamaClient


class FakeClient:
    """Returns queued payloads (or raises) instead of calling Ollama."""

    def __init__(self, responses: List[Any]) -> None:
        self.responses = list(responses)
        self.prompts: List[str] = []

    def generate_json(self, prompt: str, **_: Any) -> Any:
        self.prompts.append(prompt)
        if not self.responses:
            raise LLMError("no more canned responses")
        payload = self.responses.pop(0)
        if isinstance(payload, Exception):
            raise payload
        return payload


def make_candidates(count: int = 3) -> List[Candidate]:
    out = []
    for index in range(count):
        out.append(
            Candidate(
                id=index,
                start=100.0 * (index + 1),
                end=100.0 * (index + 1) + 30.0,
                text=f"Candidate number {index} says something interesting about data quality.",
                word_count=75,
                speech_density=2.5,
                silence_ratio=0.1,
                starts_sentence=True,
                ends_sentence=True,
                heuristic_score=5.0 + index,
            )
        )
    return out


def full_scores(**overrides) -> Dict[str, Any]:
    payload = {
        "hook": 8,
        "interestingness": 8,
        "information_value": 7,
        "emotional_impact": 6,
        "shareability": 8,
        "clarity": 9,
        "standalone_context": 7,
    }
    payload.update(overrides)
    return payload


def test_score_candidates_applies_llm_scores(config):
    candidates = make_candidates(2)
    client = FakeClient(
        [
            {
                "scores": [
                    {"id": 0, **full_scores(hook=9), "title": "Strong one", "hook_line": "Quote", "reason": "Because"},
                    {"id": 1, **full_scores(hook=2)},
                ]
            }
        ]
    )
    analyzer = LLMAnalyzer(config, client=client)
    scored = analyzer.score_candidates(candidates, {"title": "Video", "duration_label": "01:00:00"})

    assert [c.id for c in scored] == [0, 1]       # sorted by overall, best first
    assert scored[0].scored_by_llm is True
    assert scored[0].title == "Strong one"
    assert scored[0].overall > scored[1].overall
    assert "Candidate number 0" in client.prompts[0]  # the transcript reached the prompt


def test_score_candidates_falls_back_to_heuristics_on_bad_json(config):
    candidates = make_candidates(2)
    analyzer = LLMAnalyzer(config, client=FakeClient([LLMError("invalid json")]))
    scored = analyzer.score_candidates(candidates, {})

    assert all(not c.scored_by_llm for c in scored)
    assert all(c.overall > 0 for c in scored)      # heuristic score was used
    assert scored[0].heuristic_score >= scored[-1].heuristic_score


def test_score_candidates_ignores_unknown_ids(config):
    candidates = make_candidates(1)
    analyzer = LLMAnalyzer(config, client=FakeClient([{"scores": [{"id": 99, **full_scores()}]}]))
    scored = analyzer.score_candidates(candidates, {})
    assert scored[0].scored_by_llm is False        # nothing matched, heuristics kept


def test_score_candidates_accepts_a_bare_list(config):
    candidates = make_candidates(1)
    analyzer = LLMAnalyzer(config, client=FakeClient([[{"id": 0, **full_scores()}]]))
    scored = analyzer.score_candidates(candidates, {})
    assert scored[0].scored_by_llm is True


def test_select_final_returns_validated_selection(config):
    candidates = make_candidates(3)
    for index, candidate in enumerate(candidates):
        candidate.scores = full_scores()
        candidate.overall = 7.0 + index
    client = FakeClient(
        [
            {
                "selected": True,
                "selections": [
                    {
                        "id": 1,
                        "start": 205.0,
                        "end": 233.0,
                        "title": "The surprising truth",
                        "hook": "Most people think...",
                        "reason": "Contrarian and self-contained.",
                        "score": 9.1,
                    }
                ],
            }
        ]
    )
    analyzer = LLMAnalyzer(config, client=client)
    selections = analyzer.select_final(candidates, {"title": "Video"}, wanted=1)

    assert len(selections) == 1
    assert selections[0]["id"] == 1
    assert selections[0]["score"] == 9.1
    assert selections[0]["llm_start"] == 205.0


def test_select_final_rejects_hallucinated_ids(config):
    candidates = make_candidates(2)
    for candidate in candidates:
        candidate.overall = 7.5
    analyzer = LLMAnalyzer(
        config, client=FakeClient([{"selections": [{"id": 4242, "score": 10}]}])
    )
    selections = analyzer.select_final(candidates, {}, wanted=1)
    assert selections[0]["id"] in {0, 1}
    assert selections[0].get("fallback") is True


def test_select_final_pads_to_requested_count(config):
    candidates = make_candidates(3)
    for candidate in candidates:
        candidate.overall = 6.0
    analyzer = LLMAnalyzer(config, client=FakeClient([{"selections": [{"id": 0, "score": 8}]}]))
    selections = analyzer.select_final(candidates, {}, wanted=3)
    assert len(selections) == 3
    assert len({s["id"] for s in selections}) == 3


def test_select_final_survives_total_llm_failure(config):
    candidates = make_candidates(2)
    candidates[1].overall = 9.0
    analyzer = LLMAnalyzer(config, client=FakeClient([LLMError("down")]))
    selections = analyzer.select_final(candidates, {}, wanted=1)
    assert selections[0]["id"] == 1


def test_metadata_generation_cleans_hashtags(config):
    analyzer = LLMAnalyzer(
        config,
        client=FakeClient(
            [
                {
                    "title": '"Data quality beats compute"',
                    "description": "Why a laptop is enough.",
                    "hashtags": ["#ai", "machinelearning", "#ai", "bad tag!"],
                    "alt_titles": ["One", "Two"],
                }
            ]
        ),
    )
    metadata = analyzer.generate_publishing_metadata("Transcript text.", {"title": "Video"})

    assert metadata["title"] == "Data quality beats compute"
    assert metadata["hashtags"][:3] == ["#ai", "#machinelearning", "#badtag"]
    assert metadata["alt_titles"] == ["One", "Two"]


def test_metadata_generation_falls_back(config):
    analyzer = LLMAnalyzer(config, client=FakeClient([LLMError("offline")]))
    metadata = analyzer.generate_publishing_metadata(
        "The first sentence is the fallback title. The rest is description.",
        {"title": "Video", "hook": "A quoted hook"},
    )
    assert metadata["title"] == "A quoted hook"
    assert metadata["description"]
    assert metadata["hashtags"]


def test_prompt_templates_render_without_leftover_placeholders(config):
    candidates = make_candidates(1)
    client = FakeClient([{"scores": [{"id": 0, **full_scores()}]}])
    analyzer = LLMAnalyzer(config, client=client)
    analyzer.score_candidates(candidates, {"title": "T", "duration_label": "00:10:00", "language": "en"})

    prompt = client.prompts[0]
    assert "{{" not in prompt and "}}" not in prompt
    assert "T" in prompt


def test_model_availability_matching():
    assert OllamaClient._model_available("qwen2.5", ["qwen2.5:latest"]) is True
    assert OllamaClient._model_available("qwen2.5:7b", ["qwen2.5:7b"]) is True
    assert OllamaClient._model_available("qwen2.5", ["llama3.1:8b"]) is False
    assert OllamaClient._model_available("", ["qwen2.5"]) is False


def test_verify_reports_an_unreachable_server():
    client = OllamaClient(OllamaConfig(base_url="http://127.0.0.1:1", model="qwen2.5", max_retries=1, timeout=1))
    with pytest.raises(DependencyError) as excinfo:
        client.verify()
    assert "ollama serve" in str(excinfo.value).lower()
