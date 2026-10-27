"""Golden dataset schema and the JSONL, JSON, and YAML loaders."""

import pytest

from llm_eval.schema import DatasetError, load_dataset, parse_case

from paths import GOLDEN_FAIL, SAMPLE_JSONL, SAMPLE_YAML


def test_load_jsonl_sample_has_the_faq_cases():
    cases = load_dataset(SAMPLE_JSONL)
    assert [case.id for case in cases] == [
        "password-reset",
        "free-plan",
        "sso",
        "export",
        "support-hours",
        "trash",
        "rate-limit",
        "storage-region",
    ]
    storage = cases[-1]
    assert storage.expected_answer is None
    assert len(storage.expected_facts) == 2
    assert storage.recorded_answer


def test_load_yaml_matches_jsonl():
    jsonl_cases = load_dataset(SAMPLE_JSONL)
    yaml_cases = load_dataset(SAMPLE_YAML)
    assert yaml_cases == jsonl_cases


def test_golden_fail_dataset_is_the_single_bad_case():
    cases = load_dataset(GOLDEN_FAIL)
    assert len(cases) == 1
    assert cases[0].id == "trash-hallucination"
    assert "90" in cases[0].recorded_answer
    assert "Zurich" in cases[0].recorded_answer


def test_jsonl_invalid_line_reports_the_line_number(tmp_path):
    path = tmp_path / "broken.jsonl"
    path.write_text('{"question": "ok", "expected_answer": "yes"}\n{not-json}\n', encoding="utf-8")
    with pytest.raises(DatasetError, match=r":2: invalid JSON"):
        load_dataset(path)


def test_schema_error_includes_the_jsonl_line(tmp_path):
    path = tmp_path / "missing.jsonl"
    path.write_text(
        '{"question": "Q?", "expected_answer": "A"}\n{"question": "Only this"}\n',
        encoding="utf-8",
    )
    with pytest.raises(DatasetError, match=r":2:.*expected_answer or expected_facts"):
        load_dataset(path)


def test_missing_question_is_rejected():
    with pytest.raises(DatasetError, match="question"):
        parse_case({"expected_answer": "14 days"}, 1)


def test_missing_expected_oracle_is_rejected():
    with pytest.raises(DatasetError, match="expected_answer or expected_facts"):
        parse_case({"question": "How long?"}, 4)


def test_duplicate_ids_are_rejected(tmp_path):
    path = tmp_path / "dup.jsonl"
    line = '{"id": "same", "question": "Q?", "expected_answer": "A"}\n'
    path.write_text(line + line, encoding="utf-8")
    with pytest.raises(DatasetError, match="duplicate case id 'same'"):
        load_dataset(path)


def test_context_string_is_coerced_to_one_chunk():
    case = parse_case(
        {"question": "Q?", "expected_answer": "A", "context": "Only one chunk."},
        1,
    )
    assert case.context == ("Only one chunk.",)
    assert case.id == "case-1"


def test_blank_id_falls_back_to_the_index():
    case = parse_case({"id": None, "question": "Q?", "expected_facts": ["A fact."]}, 3)
    assert case.id == "case-3"
    assert case.expected_answer is None


def test_missing_file_raises_dataset_error(tmp_path):
    with pytest.raises(DatasetError, match="dataset not found"):
        load_dataset(tmp_path / "nope.jsonl")


def test_unsupported_suffix_is_rejected(tmp_path):
    path = tmp_path / "cases.txt"
    path.write_text("question\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="unsupported format"):
        load_dataset(path)


def test_json_document_with_cases_key(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(
        '{"cases": [{"id": "one", "question": "Q?", "expected_answer": "A"}]}',
        encoding="utf-8",
    )
    cases = load_dataset(path)
    assert cases[0].id == "one"


def test_yaml_without_cases_key_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("question: Q?\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="cases"):
        load_dataset(path)
