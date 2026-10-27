"""Optional demo API. Install with ``pip install -e ".[api]"``.

The core library does not import FastAPI. This module is the only place that does.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from llm_eval import __version__
from llm_eval.gates import Thresholds
from llm_eval.llm import OfflineLLM
from llm_eval.report import report_to_dict
from llm_eval.runner import evaluate
from llm_eval.schema import DatasetError, parse_case

app = FastAPI(
    title="LLM Evaluation Harness",
    version=__version__,
    summary="Offline RAG evaluation demo.",
)


class CaseIn(BaseModel):
    id: str | None = None
    question: str
    expected_answer: str | None = None
    expected_facts: list[str] = Field(default_factory=list)
    context: list[str] = Field(default_factory=list)
    recorded_answer: str | None = None
    tags: list[str] = Field(default_factory=list)


class EvaluateIn(BaseModel):
    cases: list[CaseIn]
    min_pass_rate: float = 0.8
    min_token_f1: float = 0.5
    min_faithfulness: float = 0.6
    max_hallucination_rate: float = 0.2
    require_exact_match: bool = False
    use_recorded: bool = True


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "llm-evaluation-harness", "version": __version__}


@app.post("/v1/evaluate", response_model=None)
def evaluate_cases(body: EvaluateIn) -> dict[str, Any]:
    """Score inline cases with the offline stub and return the report JSON.

    The HTTP status stays 200 when the quality gate fails so a demo client can
    read ``gate.passed``. The CLI is what returns a non-zero exit code.
    """

    try:
        thresholds = Thresholds(
            min_pass_rate=body.min_pass_rate,
            min_token_f1=body.min_token_f1,
            min_faithfulness=body.min_faithfulness,
            max_hallucination_rate=body.max_hallucination_rate,
            require_exact_match=body.require_exact_match,
        )
        cases = [
            parse_case(case.model_dump(), index, source="request")
            for index, case in enumerate(body.cases, start=1)
        ]
    except (DatasetError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    report = evaluate(
        cases,
        OfflineLLM(use_recorded=body.use_recorded),
        thresholds,
        dataset="request",
    )
    return report_to_dict(report)
