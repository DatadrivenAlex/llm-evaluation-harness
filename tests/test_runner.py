"""Aggregation, the shipped datasets, and the quality gate."""

import pytest

from llm_eval.gates import Summary, Thresholds, judge
from llm_eval.llm import OfflineLLM
from llm_eval.runner import evaluate
from llm_eval.schema import load_dataset

from paths import GOLDEN_FAIL, SAMPLE_JSONL


def test_sample_dataset_passes_the_default_gate():
    report = evaluate(load_dataset(SAMPLE_JSONL), OfflineLLM(), dataset=str(SAMPLE_JSONL))
    assert report.summary.cases == 8
    assert report.summary.passed == 8
    assert report.summary.pass_rate == 1.0
    assert report.summary.hallucination_rate == 0.0
    assert report.gate.passed is True
    assert report.failures == ()
    assert report.version


def test_free_plan_paraphrase_is_not_exact_but_still_passes():
    report = evaluate(load_dataset(SAMPLE_JSONL), OfflineLLM())
    free_plan = next(result for result in report.results if result.case_id == "free-plan")
    assert free_plan.exact_match == 0.0
    assert free_plan.token_f1 >= 0.5
    assert free_plan.faithfulness >= 0.6
    assert free_plan.hallucination is False
    assert free_plan.passed is True
    assert report.summary.exact_match == pytest.approx(7 / 8)


def test_storage_case_passes_from_facts_without_expected_answer():
    report = evaluate(load_dataset(SAMPLE_JSONL), OfflineLLM())
    storage = next(result for result in report.results if result.case_id == "storage-region")
    assert storage.expected_answer is None
    assert storage.exact_match == 1.0
    assert storage.passed is True


def test_failing_golden_case_fails_the_build_when_threshold_is_not_met():
    """The shipped counterexample must miss the default gate.

    pytest stays green because it asserts the failure. The CLI turns the same
    result into a non-zero exit, which is what fails CI.
    """

    report = evaluate(load_dataset(GOLDEN_FAIL), OfflineLLM(), dataset="golden_fail.jsonl")
    case = report.results[0]
    assert case.hallucination is True
    assert case.unsupported_entities == ("90", "Zurich")
    assert case.faithfulness == 1.0
    assert case.token_f1 == pytest.approx(0.60869565217)
    assert case.passed is False
    assert report.summary.pass_rate == 0.0
    assert report.gate.passed is False
    assert any(reason.startswith("pass_rate") for reason in report.gate.reasons)
    assert any(reason.startswith("hallucination_rate") for reason in report.gate.reasons)


def test_relaxed_thresholds_do_not_hide_the_case_failure():
    relaxed = Thresholds(
        min_pass_rate=0.0,
        min_token_f1=0.0,
        min_faithfulness=0.0,
        max_hallucination_rate=1.0,
    )
    report = evaluate(load_dataset(GOLDEN_FAIL), OfflineLLM(), relaxed)
    assert report.results[0].passed is False
    assert report.gate.passed is True
    assert report.gate.reasons == ()


def test_require_exact_match_fails_the_paraphrase_at_a_perfect_pass_rate():
    limits = Thresholds(require_exact_match=True, min_pass_rate=1.0)
    report = evaluate(load_dataset(SAMPLE_JSONL), OfflineLLM(), limits)
    free_plan = next(result for result in report.results if result.case_id == "free-plan")
    assert free_plan.passed is False
    assert report.gate.passed is False


def test_empty_dataset_fails_the_gate():
    report = evaluate([], OfflineLLM())
    assert report.summary.cases == 0
    assert report.gate.passed is False
    assert "dataset has no cases" in report.gate.reasons


def test_gate_boundary_is_inclusive_for_pass_rate_and_hallucination():
    summary = Summary(
        cases=5,
        passed=4,
        pass_rate=0.8,
        exact_match=1.0,
        token_f1=1.0,
        soft_overlap=1.0,
        faithfulness=1.0,
        hallucination_rate=0.2,
    )
    assert judge(summary, Thresholds()).passed is True
    assert judge(summary, Thresholds(min_pass_rate=0.81)).passed is False
    assert judge(summary, Thresholds(max_hallucination_rate=0.19)).passed is False


def test_threshold_outside_unit_interval_is_rejected():
    with pytest.raises(ValueError, match="min_pass_rate"):
        Thresholds(min_pass_rate=1.1)
