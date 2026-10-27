"""Run a dataset through an offline model and score every case."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from llm_eval.gates import GateResult, Summary, Thresholds, judge
from llm_eval.llm import OfflineLLM
from llm_eval.metrics import score_case
from llm_eval.schema import EvalCase


@dataclass(frozen=True)
class CaseResult:
    """Prediction plus score for one golden case."""

    case_id: str
    question: str
    answer: str
    expected_answer: str | None
    expected_facts: tuple[str, ...]
    context: tuple[str, ...]
    tags: tuple[str, ...]
    exact_match: float
    token_f1: float
    soft_overlap: float
    faithfulness: float
    hallucination: bool
    unsupported_entities: tuple[str, ...]
    passed: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class EvaluationReport:
    """Full result of one offline evaluation run."""

    dataset: str
    generated_at: str
    version: str
    thresholds: Thresholds
    summary: Summary
    gate: GateResult
    results: tuple[CaseResult, ...]

    @property
    def failures(self) -> tuple[CaseResult, ...]:
        return tuple(result for result in self.results if not result.passed)


def evaluate(
    cases: list[EvalCase] | tuple[EvalCase, ...],
    llm: OfflineLLM | None = None,
    thresholds: Thresholds | None = None,
    *,
    dataset: str = "",
) -> EvaluationReport:
    """Score every case. The model defaults to the recorded-answer stub."""

    from llm_eval import __version__

    model = llm if llm is not None else OfflineLLM()
    limits = thresholds if thresholds is not None else Thresholds()
    results = tuple(_score_one(case, model, limits) for case in cases)
    summary = _summarize(results)
    return EvaluationReport(
        dataset=dataset,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        version=__version__,
        thresholds=limits,
        summary=summary,
        gate=judge(summary, limits),
        results=results,
    )


def _score_one(case: EvalCase, llm: OfflineLLM, thresholds: Thresholds) -> CaseResult:
    answer = llm.complete(case)
    scored = score_case(
        answer,
        expected_answer=case.expected_answer,
        expected_facts=case.expected_facts,
        context=case.context,
        question=case.question,
        min_token_f1=thresholds.min_token_f1,
        min_faithfulness=thresholds.min_faithfulness,
        require_exact_match=thresholds.require_exact_match,
    )
    return CaseResult(
        case_id=case.id,
        question=case.question,
        answer=answer,
        expected_answer=case.expected_answer,
        expected_facts=case.expected_facts,
        context=case.context,
        tags=case.tags,
        exact_match=scored.exact_match,
        token_f1=scored.token_f1,
        soft_overlap=scored.soft_overlap,
        faithfulness=scored.faithfulness,
        hallucination=scored.hallucination,
        unsupported_entities=scored.unsupported_entities,
        passed=scored.passed,
        reasons=scored.reasons,
    )


def _summarize(results: tuple[CaseResult, ...]) -> Summary:
    count = len(results)
    if count == 0:
        return Summary(
            cases=0,
            passed=0,
            pass_rate=0.0,
            exact_match=0.0,
            token_f1=0.0,
            soft_overlap=0.0,
            faithfulness=0.0,
            hallucination_rate=0.0,
        )
    passed = sum(1 for result in results if result.passed)
    return Summary(
        cases=count,
        passed=passed,
        pass_rate=passed / count,
        exact_match=_mean([result.exact_match for result in results]),
        token_f1=_mean([result.token_f1 for result in results]),
        soft_overlap=_mean([result.soft_overlap for result in results]),
        faithfulness=_mean([result.faithfulness for result in results]),
        hallucination_rate=_mean([1.0 if result.hallucination else 0.0 for result in results]),
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)
