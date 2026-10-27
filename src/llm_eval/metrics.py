"""Lexical metrics for offline RAG answer scoring.

These are deterministic heuristics. They are not a model judge and they are
not RAGAS. Faithfulness is token overlap between answer sentences and the
supplied context. The hallucination flag looks for numbers, acronyms, and
capitalized words in the answer that never appear in the context or question.
Negation and paraphrases that share no tokens are outside what this can see.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

_TOKEN = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_NUMBER = re.compile(r"(?<![\w])[$€£]?\d+(?:[.,]\d+)?%?")
_ACRONYM = re.compile(r"\b[A-Z]{2,}\b")
_CAPITALIZED = re.compile(r"\b[A-Z][a-z]{2,}\b")

STOPWORDS = frozenset(
    """
    a an the of to in on for and or is are was were be been being with from by at
    as it this that these those our your we you they them their i my not no nor
    but if then than so into over under about within before after also can may per
    """.split()
)


@dataclass(frozen=True)
class CaseScore:
    """Metrics and the pass decision for a single case."""

    exact_match: float
    token_f1: float
    soft_overlap: float
    faithfulness: float
    hallucination: bool
    unsupported_entities: tuple[str, ...]
    passed: bool
    reasons: tuple[str, ...]


def tokenize(text: str) -> list[str]:
    """Lowercase word and number tokens. Punctuation is discarded."""

    return [match.group(0).lower() for match in _TOKEN.finditer(text)]


def normalize(text: str) -> str:
    """Canonical form used by exact match."""

    return " ".join(tokenize(text))


def split_sentences(text: str) -> list[str]:
    """Split on end punctuation. Abbreviations are not specially handled."""

    stripped = text.strip()
    if not stripped:
        return []
    return [part.strip() for part in _SENTENCE_SPLIT.split(stripped) if part.strip()]


def exact_match_score(
    prediction: str,
    expected_answer: str | None,
    expected_facts: list[str] | tuple[str, ...],
) -> float:
    """Return 1 when the prediction matches the oracle after normalization.

    When ``expected_answer`` is set, the normalized strings must be equal.
    Otherwise every normalized fact must occur inside the normalized prediction.
    """

    predicted = normalize(prediction)
    if expected_answer:
        return 1.0 if predicted == normalize(expected_answer) else 0.0
    facts = [normalize(fact) for fact in expected_facts]
    if not facts or any(not fact for fact in facts):
        return 0.0
    return 1.0 if all(fact in predicted for fact in facts) else 0.0


def token_f1(prediction: str, reference: str) -> float:
    """SQuAD-style multiset token F1. Token order does not matter."""

    predicted = tokenize(prediction)
    expected = tokenize(reference)
    if not predicted and not expected:
        return 1.0
    if not predicted or not expected:
        return 0.0
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def soft_overlap(prediction: str, reference: str) -> float:
    """Jaccard similarity over the unique tokens of prediction and reference."""

    predicted = set(tokenize(prediction))
    expected = set(tokenize(reference))
    if not predicted and not expected:
        return 1.0
    if not predicted or not expected:
        return 0.0
    return len(predicted & expected) / len(predicted | expected)


def reference_text(expected_answer: str | None, expected_facts: list[str] | tuple[str, ...]) -> str:
    """Text the overlap metrics compare against."""

    if expected_answer:
        return expected_answer
    return " ".join(expected_facts)


def faithfulness(answer: str, context: list[str] | tuple[str, ...], *, support_threshold: float = 0.6) -> float:
    """Fraction of answer sentences whose content tokens are grounded in context.

    A sentence is supported when at least ``support_threshold`` of its content
    tokens (stopwords and bare numbers removed) appear in the context. Numbers
    are left to the hallucination check. An empty answer scores 0.
    """

    sentences = split_sentences(answer)
    if not sentences:
        return 0.0
    context_tokens = set(_content_tokens(tokenize(" ".join(context))))
    supported = 0
    considered = 0
    for sentence in sentences:
        content = _content_tokens(tokenize(sentence))
        if not content:
            continue
        considered += 1
        if not context_tokens:
            continue
        hits = sum(1 for token in content if token in context_tokens)
        if hits / len(content) >= support_threshold:
            supported += 1
    if considered == 0:
        return 1.0
    return supported / considered


def unsupported_entities(answer: str, context: list[str] | tuple[str, ...], question: str = "") -> tuple[str, ...]:
    """Numbers, acronyms, and capitalized words absent from context and question.

    Sentence-initial capitals count. Stopwords such as "The" do not. A number
    is supported only when the same numeric value appears in the support text,
    so 3 does not excuse 30.
    """

    support = " ".join(context) + " " + question
    support_tokens = set(tokenize(support))
    support_numbers = _number_keys(support)
    found: list[str] = []
    seen: set[str] = set()

    def add(surface: str) -> None:
        key = surface.strip()
        marker = key.lower()
        if not key or marker in seen:
            return
        seen.add(marker)
        found.append(key)

    for match in _NUMBER.finditer(answer):
        if _number_key(match.group(0)) not in support_numbers:
            add(match.group(0))
    for match in _ACRONYM.finditer(answer):
        if match.group(0).lower() not in support_tokens:
            add(match.group(0))
    for match in _CAPITALIZED.finditer(answer):
        word = match.group(0)
        if word.lower() in STOPWORDS:
            continue
        if word.lower() not in support_tokens:
            add(word)
    return tuple(found)


def score_case(
    prediction: str,
    *,
    expected_answer: str | None,
    expected_facts: list[str] | tuple[str, ...],
    context: list[str] | tuple[str, ...],
    question: str = "",
    min_token_f1: float = 0.5,
    min_faithfulness: float = 0.6,
    require_exact_match: bool = False,
) -> CaseScore:
    """Score one prediction. A hallucinated entity fails the case."""

    reference = reference_text(expected_answer, expected_facts)
    exact = exact_match_score(prediction, expected_answer, expected_facts)
    f1 = token_f1(prediction, reference)
    overlap = soft_overlap(prediction, reference)
    faith = faithfulness(prediction, context)
    entities = unsupported_entities(prediction, context, question)
    reasons: list[str] = []
    if require_exact_match and exact < 1.0:
        reasons.append("exact_match 0.000 < 1.000")
    if f1 < min_token_f1:
        reasons.append(f"token_f1 {f1:.3f} < {min_token_f1:.3f}")
    if faith < min_faithfulness:
        reasons.append(f"faithfulness {faith:.3f} < {min_faithfulness:.3f}")
    if entities:
        reasons.append("hallucination: " + ", ".join(entities))
    return CaseScore(
        exact_match=exact,
        token_f1=f1,
        soft_overlap=overlap,
        faithfulness=faith,
        hallucination=bool(entities),
        unsupported_entities=entities,
        passed=not reasons,
        reasons=tuple(reasons),
    )


def _content_tokens(tokens: list[str]) -> list[str]:
    return [token for token in tokens if token not in STOPWORDS and not token.isdigit()]


def _number_keys(text: str) -> set[str]:
    return {_number_key(match.group(0)) for match in _NUMBER.finditer(text)}


def _number_key(raw: str) -> str:
    return raw.lower().lstrip("$€£").rstrip("%").replace(",", "")
