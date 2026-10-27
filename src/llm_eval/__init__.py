"""Offline LLM and RAG evaluation harness."""

__version__ = "0.1.0"

from llm_eval.gates import GateResult, Thresholds
from llm_eval.llm import OfflineLLM
from llm_eval.runner import EvaluationReport, evaluate
from llm_eval.schema import EvalCase, load_dataset

__all__ = [
    "EvalCase",
    "EvaluationReport",
    "GateResult",
    "OfflineLLM",
    "Thresholds",
    "evaluate",
    "load_dataset",
]
