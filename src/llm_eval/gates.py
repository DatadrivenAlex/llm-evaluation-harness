"""Dataset-level quality gate."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Thresholds:
    """Cutoffs for a case and for the dataset gate.

    A case passes when token F1 and faithfulness meet their minimums, exact
    match holds if it was required, and the answer introduces no unsupported
    entity. The dataset gate then checks pass rate and hallucination rate.
    """

    min_pass_rate: float = 0.8
    min_token_f1: float = 0.5
    min_faithfulness: float = 0.6
    max_hallucination_rate: float = 0.2
    require_exact_match: bool = False

    def __post_init__(self) -> None:
        for name in (
            "min_pass_rate",
            "min_token_f1",
            "min_faithfulness",
            "max_hallucination_rate",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"{name} must be a number between 0 and 1")
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1, got {value}")


@dataclass(frozen=True)
class GateResult:
    """Outcome of applying :class:`Thresholds` to an aggregate summary."""

    passed: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class Summary:
    """Means across every case. Rates are fractions in ``[0, 1]``."""

    cases: int
    passed: int
    pass_rate: float
    exact_match: float
    token_f1: float
    soft_overlap: float
    faithfulness: float
    hallucination_rate: float


def judge(summary: Summary, thresholds: Thresholds) -> GateResult:
    """Return a failing gate when a dataset threshold is missed."""

    reasons: list[str] = []
    if summary.cases == 0:
        reasons.append("dataset has no cases")
    if summary.pass_rate < thresholds.min_pass_rate:
        reasons.append(
            f"pass_rate {summary.pass_rate:.4f} < min_pass_rate {thresholds.min_pass_rate:.4f}"
        )
    if summary.hallucination_rate > thresholds.max_hallucination_rate:
        reasons.append(
            "hallucination_rate "
            f"{summary.hallucination_rate:.4f} > max_hallucination_rate "
            f"{thresholds.max_hallucination_rate:.4f}"
        )
    return GateResult(passed=not reasons, reasons=tuple(reasons))
