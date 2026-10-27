"""Command line interface.

Examples::

    python -m llm_eval run --dataset datasets/rag_faq.jsonl --out reports/
    python -m llm_eval run --dataset datasets/golden_fail.jsonl --out reports/golden-fail/
    python -m llm_eval serve --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from llm_eval.gates import Thresholds
from llm_eval.llm import OfflineLLM, load_responses
from llm_eval.report import format_summary, write_report
from llm_eval.runner import evaluate
from llm_eval.schema import DatasetError, load_dataset


def main(argv: list[str] | None = None) -> int:
    """Run the CLI. Returns a process exit code."""

    load_dotenv()
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return _run(args)
    if args.command == "serve":
        return _serve(args)
    parser.print_help(sys.stderr)
    return 2


def load_dotenv(path: Path | None = None) -> None:
    """Load KEY=VALUE lines from ``.env`` without overriding the environment."""

    env_path = path if path is not None else Path(".env")
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-eval",
        description="Offline LLM and RAG evaluation harness.",
    )
    subparsers = parser.add_subparsers(dest="command")

    run = subparsers.add_parser("run", help="Score a golden dataset and write a report")
    run.add_argument("--dataset", required=True, help="Path to a JSONL, JSON, or YAML dataset")
    run.add_argument("--out", required=True, help="Directory for JSON and Markdown reports")
    run.add_argument(
        "--responses",
        help="JSON object mapping case id or question to an answer. Overrides recorded answers.",
    )
    run.add_argument("--min-pass-rate", type=float, default=None)
    run.add_argument("--min-token-f1", type=float, default=None)
    run.add_argument("--min-faithfulness", type=float, default=None)
    run.add_argument("--max-hallucination-rate", type=float, default=None)
    run.add_argument(
        "--require-exact-match",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Fail a case unless normalized exact match is 1",
    )
    run.add_argument(
        "--ignore-recorded",
        action="store_true",
        help="Skip recorded_answer fields and use the extractive stub",
    )

    serve = subparsers.add_parser("serve", help="Run the optional FastAPI demo")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    return parser


def _run(args: argparse.Namespace) -> int:
    try:
        thresholds = _thresholds_from_args(args)
        cases = load_dataset(args.dataset)
        responses = load_responses(args.responses) if args.responses else None
    except (DatasetError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    report = evaluate(
        cases,
        OfflineLLM(responses, use_recorded=not args.ignore_recorded),
        thresholds,
        dataset=str(args.dataset),
    )
    paths = write_report(report, args.out)
    print(format_summary(report))
    print(f"Report: {paths['report_md']}")
    return 0 if report.gate.passed else 1


def _serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print(
            "error: the API extra is not installed. Run: pip install -e '.[api]'",
            file=sys.stderr,
        )
        return 2
    uvicorn.run("llm_eval.api:app", host=args.host, port=args.port)
    return 0


def _thresholds_from_args(args: argparse.Namespace) -> Thresholds:
    require = args.require_exact_match
    if require is None:
        require = _env_bool("LLM_EVAL_REQUIRE_EXACT_MATCH", False)
    return Thresholds(
        min_pass_rate=_env_float("LLM_EVAL_MIN_PASS_RATE", args.min_pass_rate, 0.8),
        min_token_f1=_env_float("LLM_EVAL_MIN_TOKEN_F1", args.min_token_f1, 0.5),
        min_faithfulness=_env_float("LLM_EVAL_MIN_FAITHFULNESS", args.min_faithfulness, 0.6),
        max_hallucination_rate=_env_float(
            "LLM_EVAL_MAX_HALLUCINATION_RATE",
            args.max_hallucination_rate,
            0.2,
        ),
        require_exact_match=require,
    )


def _env_float(name: str, cli_value: float | None, default: float) -> float:
    if cli_value is not None:
        return cli_value
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} must be true or false, got {raw!r}")


if __name__ == "__main__":
    raise SystemExit(main())
