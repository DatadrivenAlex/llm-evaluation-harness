"""JSON and Markdown exports for humans and CI artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from llm_eval.gates import GateResult, Summary, Thresholds
from llm_eval.runner import CaseResult, EvaluationReport


def report_to_dict(report: EvaluationReport) -> dict[str, object]:
    """Serialize a report to plain JSON-compatible objects."""

    return {
        "dataset": report.dataset,
        "generated_at": report.generated_at,
        "version": report.version,
        "thresholds": _thresholds_dict(report.thresholds),
        "summary": _summary_dict(report.summary),
        "gate": _gate_dict(report.gate),
        "cases": [_case_dict(result) for result in report.results],
        "failures": [_case_dict(result) for result in report.failures],
    }


def write_report(report: EvaluationReport, out_dir: str | Path) -> dict[str, Path]:
    """Write the full report and a failures-only export.

    Files:

    * ``report.json`` / ``report.md`` — every case and the gate
    * ``failures.json`` / ``failures.md`` — cases that missed a case threshold
    """

    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    payload = report_to_dict(report)
    paths = {
        "report_json": destination / "report.json",
        "report_md": destination / "report.md",
        "failures_json": destination / "failures.json",
        "failures_md": destination / "failures.md",
    }
    paths["report_json"].write_text(_dumps(payload), encoding="utf-8")
    paths["report_md"].write_text(render_markdown(report), encoding="utf-8")
    paths["failures_json"].write_text(_dumps(payload["failures"]), encoding="utf-8")
    paths["failures_md"].write_text(render_failures_markdown(report), encoding="utf-8")
    return paths


def render_markdown(report: EvaluationReport) -> str:
    """Full report a reviewer can read without opening the JSON."""

    summary = report.summary
    lines = [
        "# Evaluation report",
        "",
        f"- Generated: {report.generated_at}",
        f"- Harness: llm-evaluation-harness {report.version}",
        f"- Dataset: {report.dataset or '(inline)'}",
        f"- Cases: {summary.cases}",
        f"- Passed: {summary.passed}",
        f"- Pass rate: {summary.pass_rate:.1%}",
        f"- Gate: {_gate_label(report.gate)}",
        "",
        "## Thresholds",
        "",
        _thresholds_markdown(report.thresholds),
        "",
        "## Mean scores",
        "",
        "| Metric | Score |",
        "| --- | ---: |",
        f"| exact_match | {summary.exact_match:.3f} |",
        f"| token_f1 | {summary.token_f1:.3f} |",
        f"| soft_overlap | {summary.soft_overlap:.3f} |",
        f"| faithfulness | {summary.faithfulness:.3f} |",
        f"| hallucination_rate | {summary.hallucination_rate:.3f} |",
        "",
    ]
    if report.gate.reasons:
        lines.extend(["## Gate reasons", ""])
        lines.extend(f"- {reason}" for reason in report.gate.reasons)
        lines.append("")
    lines.extend(["## Failing cases", ""])
    lines.append(_failure_body(report))
    lines.extend(["## Cases", ""])
    if not report.results:
        lines.append("No cases.")
        lines.append("")
    for result in report.results:
        lines.extend(_case_markdown(result))
    return "\n".join(lines).rstrip() + "\n"


def render_failures_markdown(report: EvaluationReport) -> str:
    """Markdown limited to cases that failed their own thresholds."""

    lines = [
        "# Failing cases",
        "",
        f"- Dataset: {report.dataset or '(inline)'}",
        f"- Failures: {len(report.failures)} of {report.summary.cases}",
        f"- Gate: {_gate_label(report.gate)}",
        "",
        _failure_body(report).rstrip(),
        "",
    ]
    return "\n".join(lines)


def format_summary(report: EvaluationReport) -> str:
    """Short stdout block for the CLI."""

    summary = report.summary
    status = _gate_label(report.gate)
    lines = [
        f"Dataset: {report.dataset or '(inline)'}",
        (
            f"Cases: {summary.cases}  Passed: {summary.passed}  "
            f"Pass rate: {summary.pass_rate:.1%}"
        ),
        (
            f"exact_match {summary.exact_match:.3f} | "
            f"token_f1 {summary.token_f1:.3f} | "
            f"soft_overlap {summary.soft_overlap:.3f} | "
            f"faithfulness {summary.faithfulness:.3f} | "
            f"hallucination_rate {summary.hallucination_rate:.3f}"
        ),
        f"Gate: {status}",
    ]
    if report.failures:
        failed_ids = ", ".join(result.case_id for result in report.failures)
        lines.append(f"Failing cases: {failed_ids}")
    for reason in report.gate.reasons:
        lines.append(f"- {reason}")
    return "\n".join(lines)


def _failure_body(report: EvaluationReport) -> str:
    if not report.failures:
        return "No failing cases.\n"
    blocks: list[str] = []
    for result in report.failures:
        blocks.extend(_case_markdown(result))
    return "\n".join(blocks)


def _case_markdown(result: CaseResult) -> list[str]:
    label = "PASS" if result.passed else "FAIL"
    entities = ", ".join(result.unsupported_entities) if result.unsupported_entities else "none"
    if result.expected_answer:
        expected_line = f"- Expected: {_inline(result.expected_answer)}"
    else:
        expected_line = "- Expected facts: " + _inline(" ".join(result.expected_facts))
    lines = [
        f"### {result.case_id} — {label}",
        "",
        f"- Question: {_inline(result.question)}",
        f"- Answer: {_inline(result.answer)}",
        expected_line,
        f"- exact_match: {result.exact_match:.3f}",
        f"- token_f1: {result.token_f1:.3f}",
        f"- soft_overlap: {result.soft_overlap:.3f}",
        f"- faithfulness: {result.faithfulness:.3f}",
        f"- hallucination: {'yes' if result.hallucination else 'no'}",
        f"- unsupported_entities: {entities}",
    ]
    if result.reasons:
        lines.append("- Reasons: " + "; ".join(result.reasons))
    lines.append("")
    return lines


def _thresholds_markdown(thresholds: Thresholds) -> str:
    return "\n".join(
        [
            f"- min_pass_rate: {thresholds.min_pass_rate:.3f}",
            f"- min_token_f1: {thresholds.min_token_f1:.3f}",
            f"- min_faithfulness: {thresholds.min_faithfulness:.3f}",
            f"- max_hallucination_rate: {thresholds.max_hallucination_rate:.3f}",
            f"- require_exact_match: {str(thresholds.require_exact_match).lower()}",
        ]
    )


def _thresholds_dict(thresholds: Thresholds) -> dict[str, object]:
    return {
        "min_pass_rate": thresholds.min_pass_rate,
        "min_token_f1": thresholds.min_token_f1,
        "min_faithfulness": thresholds.min_faithfulness,
        "max_hallucination_rate": thresholds.max_hallucination_rate,
        "require_exact_match": thresholds.require_exact_match,
    }


def _summary_dict(summary: Summary) -> dict[str, float | int]:
    return {
        "cases": summary.cases,
        "passed": summary.passed,
        "pass_rate": summary.pass_rate,
        "exact_match": summary.exact_match,
        "token_f1": summary.token_f1,
        "soft_overlap": summary.soft_overlap,
        "faithfulness": summary.faithfulness,
        "hallucination_rate": summary.hallucination_rate,
    }


def _gate_dict(gate: GateResult) -> dict[str, object]:
    return {"passed": gate.passed, "reasons": list(gate.reasons)}


def _case_dict(result: CaseResult) -> dict[str, object]:
    return {
        "id": result.case_id,
        "question": result.question,
        "answer": result.answer,
        "expected_answer": result.expected_answer,
        "expected_facts": list(result.expected_facts),
        "context": list(result.context),
        "tags": list(result.tags),
        "passed": result.passed,
        "reasons": list(result.reasons),
        "metrics": {
            "exact_match": result.exact_match,
            "token_f1": result.token_f1,
            "soft_overlap": result.soft_overlap,
            "faithfulness": result.faithfulness,
            "hallucination": result.hallucination,
            "unsupported_entities": list(result.unsupported_entities),
        },
    }


def _gate_label(gate: GateResult) -> str:
    return "PASS" if gate.passed else "FAIL"


def _inline(text: str) -> str:
    return " ".join(text.split())


def _dumps(payload: object) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
