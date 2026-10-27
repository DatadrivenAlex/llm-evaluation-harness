"""Offline model: recorded answers, response files, and the extractive stub."""

import pytest

from llm_eval.llm import OfflineLLM, extractive_answer, load_responses
from llm_eval.schema import EvalCase, load_dataset

from paths import SAMPLE_JSONL


def _case(**overrides) -> EvalCase:
    payload = {
        "id": "storage-region",
        "question": "Where are backups stored?",
        "expected_answer": "Backups stay in the EU.",
        "expected_facts": (),
        "context": (
            "Primary storage is in the eu-central region.",
            "Backups stay in the EU.",
        ),
        "recorded_answer": "Backups stay in the EU.",
        "tags": (),
    }
    payload.update(overrides)
    return EvalCase(**payload)


def test_recorded_answer_is_used_by_default():
    case = load_dataset(SAMPLE_JSONL)[0]
    assert OfflineLLM().complete(case) == case.recorded_answer


def test_responses_override_recorded_answer_by_id():
    case = _case(recorded_answer="Backups stay in the EU.")
    llm = OfflineLLM({"storage-region": "Backups live in the Zurich vault."})
    assert llm.complete(case) == "Backups live in the Zurich vault."


def test_responses_can_be_keyed_by_question():
    case = _case()
    llm = OfflineLLM({case.question: "Backups stay in the EU region."})
    assert llm.complete(case) == "Backups stay in the EU region."


def test_extractive_stub_selects_the_overlapping_sentence():
    case = _case(recorded_answer=None)
    assert extractive_answer(case) == "Backups stay in the EU."
    assert OfflineLLM(use_recorded=False).complete(case) == "Backups stay in the EU."


def test_extractive_stub_returns_empty_without_context():
    case = _case(context=(), recorded_answer=None)
    assert extractive_answer(case) == ""


def test_load_responses_rejects_a_list(tmp_path):
    path = tmp_path / "responses.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        load_responses(path)


def test_load_responses_round_trip(tmp_path):
    path = tmp_path / "responses.json"
    path.write_text('{"trash": "14 days"}', encoding="utf-8")
    assert load_responses(path) == {"trash": "14 days"}
