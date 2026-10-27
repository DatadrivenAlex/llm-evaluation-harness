"""Dataset schema and loaders for JSONL, JSON, and YAML golden sets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml


class DatasetError(ValueError):
    """A golden dataset is missing, malformed, or fails schema checks."""


@dataclass(frozen=True)
class EvalCase:
    """One question, its oracle, and the context a RAG answer may use.

    ``expected_answer`` is a full reference string. ``expected_facts`` are
    shorter claims that must be covered when a single reference string is
    omitted. ``recorded_answer`` is an optional offline model transcript.
    """

    id: str
    question: str
    expected_answer: str | None
    expected_facts: tuple[str, ...]
    context: tuple[str, ...]
    recorded_answer: str | None = None
    tags: tuple[str, ...] = ()


def load_dataset(path: str | Path) -> list[EvalCase]:
    """Load cases from a ``.jsonl``, ``.json``, ``.yaml``, or ``.yml`` file."""

    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise DatasetError(f"dataset not found: {dataset_path}")

    suffix = dataset_path.suffix.lower()
    text = dataset_path.read_text(encoding="utf-8")
    source = str(dataset_path)
    if suffix == ".jsonl":
        numbered = _parse_jsonl(text, source)
        cases = [
            parse_case(raw, index, source=f"{source}:{lineno}")
            for index, (lineno, raw) in enumerate(numbered, start=1)
        ]
    elif suffix == ".json":
        raw_cases = _cases_from_document(_load_json(text, source), source)
        cases = [
            parse_case(raw, index, source=source) for index, raw in enumerate(raw_cases, start=1)
        ]
    elif suffix in {".yaml", ".yml"}:
        raw_cases = _cases_from_document(_load_yaml(text, source), source)
        cases = [
            parse_case(raw, index, source=source) for index, raw in enumerate(raw_cases, start=1)
        ]
    else:
        raise DatasetError(
            f"{source}: unsupported format {suffix!r}. Use .jsonl, .json, .yaml, or .yml."
        )

    _reject_duplicate_ids(cases, source)
    return cases


def parse_case(raw: object, index: int, *, source: str = "dataset") -> EvalCase:
    """Validate one case object. ``index`` is 1-based for error messages."""

    where = f"{source}: case {index}"
    if not isinstance(raw, dict):
        raise DatasetError(f"{where} must be an object")

    question = raw.get("question")
    if not isinstance(question, str) or not question.strip():
        raise DatasetError(f"{where} is missing a non-empty question")

    expected_answer = raw.get("expected_answer")
    if expected_answer is not None:
        if not isinstance(expected_answer, str):
            raise DatasetError(f"{where} field expected_answer must be a string")
        expected_answer = expected_answer.strip() or None

    facts = _string_list(raw.get("expected_facts"), field="expected_facts", where=where)
    if expected_answer is None and not facts:
        raise DatasetError(f"{where} needs expected_answer or expected_facts")

    context_value = raw.get("context", raw.get("contexts", []))
    context = _string_list(context_value, field="context", where=where)
    tags = _string_list(raw.get("tags", []), field="tags", where=where)

    recorded = raw.get("recorded_answer")
    if recorded is not None and not isinstance(recorded, str):
        raise DatasetError(f"{where} field recorded_answer must be a string")

    raw_id = raw.get("id")
    case_id = f"case-{index}" if raw_id is None or raw_id == "" else raw_id
    if not isinstance(case_id, str) or not case_id.strip():
        raise DatasetError(f"{where} field id must be a non-empty string")

    return EvalCase(
        id=case_id.strip(),
        question=question.strip(),
        expected_answer=expected_answer,
        expected_facts=tuple(facts),
        context=tuple(context),
        recorded_answer=recorded,
        tags=tuple(tags),
    )


def _parse_jsonl(text: str, source: str) -> list[tuple[int, object]]:
    cases: list[tuple[int, object]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            cases.append((lineno, json.loads(stripped)))
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{source}:{lineno}: invalid JSON: {exc.msg}") from exc
    return cases


def _load_json(text: str, source: str) -> object:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise DatasetError(f"{source}: invalid JSON: {exc.msg}") from exc


def _load_yaml(text: str, source: str) -> object:
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise DatasetError(f"{source}: invalid YAML: {exc}") from exc


def _cases_from_document(data: object, source: str) -> list[object]:
    if isinstance(data, dict):
        if "cases" not in data:
            raise DatasetError(f"{source}: expected a 'cases' list")
        data = data["cases"]
    if not isinstance(data, list):
        raise DatasetError(f"{source}: dataset must be a list of cases")
    return data


def _string_list(value: object, *, field: str, where: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        items: list[str] = []
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise DatasetError(f"{where} field {field} must be a list of non-empty strings")
            items.append(item.strip())
        return items
    raise DatasetError(f"{where} field {field} must be a string or a list of strings")


def _reject_duplicate_ids(cases: list[EvalCase], source: str) -> None:
    seen: set[str] = set()
    for case in cases:
        if case.id in seen:
            raise DatasetError(f"{source}: duplicate case id {case.id!r}")
        seen.add(case.id)
