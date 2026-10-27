"""Report files a reviewer or a CI artifact store can open."""

import json

from llm_eval.llm import OfflineLLM
from llm_eval.runner import evaluate
from llm_eval.report import write_report
from llm_eval.schema import load_dataset

from paths import GOLDEN_FAIL, SAMPLE_JSONL


def test_sample_report_writes_json_and_markdown(tmp_path):
    report = evaluate(load_dataset(SAMPLE_JSONL), OfflineLLM(), dataset="datasets/rag_faq.jsonl")
    paths = write_report(report, tmp_path)

    payload = json.loads(paths["report_json"].read_text(encoding="utf-8"))
    assert payload["summary"]["cases"] == 8
    assert payload["summary"]["pass_rate"] == 1.0
    assert payload["gate"]["passed"] is True
    assert payload["failures"] == []
    assert payload["cases"][1]["id"] == "free-plan"
    assert "metrics" in payload["cases"][1]

    markdown = paths["report_md"].read_text(encoding="utf-8")
    assert "Gate: PASS" in markdown
    assert "free-plan" in markdown
    assert "token_f1" in markdown

    failures_md = paths["failures_md"].read_text(encoding="utf-8")
    assert "No failing cases." in failures_md
    assert json.loads(paths["failures_json"].read_text(encoding="utf-8")) == []


def test_failure_export_names_the_golden_case(tmp_path):
    report = evaluate(load_dataset(GOLDEN_FAIL), OfflineLLM(), dataset="datasets/golden_fail.jsonl")
    paths = write_report(report, tmp_path / "nested" / "out")

    failures = json.loads(paths["failures_json"].read_text(encoding="utf-8"))
    assert failures[0]["id"] == "trash-hallucination"
    assert "Zurich" in failures[0]["metrics"]["unsupported_entities"]
    assert "90" in failures[0]["metrics"]["unsupported_entities"]

    markdown = paths["report_md"].read_text(encoding="utf-8")
    assert "Gate: FAIL" in markdown
    assert "pass_rate" in markdown
    failure_md = paths["failures_md"].read_text(encoding="utf-8")
    assert "trash-hallucination — FAIL" in failure_md
    assert "Zurich" in failure_md
