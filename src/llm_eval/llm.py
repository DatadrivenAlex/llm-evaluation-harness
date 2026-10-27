"""Offline model stand-ins. No network calls."""

from __future__ import annotations

import json
from pathlib import Path

from llm_eval.metrics import split_sentences, tokenize
from llm_eval.schema import EvalCase


class OfflineLLM:
    """Resolve an answer without leaving the machine.

    Lookup order:

    1. ``responses`` keyed by case id
    2. ``responses`` keyed by the question text
    3. ``case.recorded_answer``, when enabled
    4. The context sentence with the highest token overlap to the question
    """

    def __init__(self, responses: dict[str, str] | None = None, *, use_recorded: bool = True):
        self.responses = dict(responses or {})
        self.use_recorded = use_recorded

    def complete(self, case: EvalCase) -> str:
        if case.id in self.responses:
            return self.responses[case.id]
        if case.question in self.responses:
            return self.responses[case.question]
        if self.use_recorded and case.recorded_answer is not None:
            return case.recorded_answer
        return extractive_answer(case)


def extractive_answer(case: EvalCase) -> str:
    """Deterministic extractive stub used when no transcript was recorded."""

    sentences: list[str] = []
    for chunk in case.context:
        sentences.extend(split_sentences(chunk))
    if not sentences:
        return ""
    question_tokens = set(tokenize(case.question))
    best = sentences[0]
    best_score = -1
    for sentence in sentences:
        score = len(question_tokens & set(tokenize(sentence)))
        if score > best_score:
            best = sentence
            best_score = score
    return best


def load_responses(path: str | Path) -> dict[str, str]:
    """Load a JSON object mapping case id or question text to an answer."""

    response_path = Path(path)
    try:
        data = json.loads(response_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{response_path}: invalid JSON: {exc.msg}") from exc
    except OSError as exc:
        raise ValueError(f"responses file not found: {response_path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{response_path}: expected a JSON object")
    loaded: dict[str, str] = {}
    for key, value in data.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError(f"{response_path}: keys and values must be strings")
        loaded[key] = value
    return loaded
