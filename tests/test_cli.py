"""CLI contract: offline run, non-zero gate, and .env overrides."""

import json
import os
import subprocess
import sys

from paths import GOLDEN_FAIL, SAMPLE_JSONL


def _run(args, *, cwd, env=None):
    return subprocess.run(
        [sys.executable, "-m", "llm_eval", *args],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _clean_env():
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("LLM_EVAL_"):
            del env[key]
    return env


def test_module_help_exits_zero():
    result = _run(["--help"], cwd=SAMPLE_JSONL.parent)
    assert result.returncode == 0
    assert "run" in result.stdout
    assert "serve" in result.stdout


def test_cli_without_a_command_exits_two():
    result = _run([], cwd=SAMPLE_JSONL.parent)
    assert result.returncode == 2


def test_cli_sample_dataset_writes_a_passing_report(tmp_path):
    out = tmp_path / "reports"
    result = _run(
        ["run", "--dataset", str(SAMPLE_JSONL), "--out", str(out)],
        cwd=tmp_path,
        env=_clean_env(),
    )
    assert result.returncode == 0, result.stderr
    assert "Gate: PASS" in result.stdout
    assert "Pass rate: 100.0%" in result.stdout
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["gate"]["passed"] is True
    assert "Gate: PASS" in (out / "report.md").read_text(encoding="utf-8")


def test_cli_failing_golden_case_exits_nonzero(tmp_path):
    out = tmp_path / "reports"
    result = _run(
        ["run", "--dataset", str(GOLDEN_FAIL), "--out", str(out)],
        cwd=tmp_path,
        env=_clean_env(),
    )
    assert result.returncode == 1
    assert "Gate: FAIL" in result.stdout
    assert "trash-hallucination" in result.stdout
    assert (out / "failures.json").is_file()
    assert (out / "failures.md").is_file()


def test_cli_missing_dataset_exits_two(tmp_path):
    result = _run(
        ["run", "--dataset", str(tmp_path / "missing.jsonl"), "--out", str(tmp_path / "out")],
        cwd=tmp_path,
        env=_clean_env(),
    )
    assert result.returncode == 2
    assert "dataset not found" in result.stderr


def test_cli_rejects_a_threshold_above_one(tmp_path):
    result = _run(
        [
            "run",
            "--dataset",
            str(SAMPLE_JSONL),
            "--out",
            str(tmp_path / "out"),
            "--min-pass-rate",
            "2",
        ],
        cwd=tmp_path,
        env=_clean_env(),
    )
    assert result.returncode == 2
    assert "min_pass_rate" in result.stderr


def test_cli_dotenv_can_relax_the_golden_failure(tmp_path):
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "LLM_EVAL_MIN_PASS_RATE=0",
                "LLM_EVAL_MIN_TOKEN_F1=0",
                "LLM_EVAL_MIN_FAITHFULNESS=0",
                "LLM_EVAL_MAX_HALLUCINATION_RATE=1",
            ]
        ),
        encoding="utf-8",
    )
    result = _run(
        ["run", "--dataset", str(GOLDEN_FAIL), "--out", str(tmp_path / "out")],
        cwd=tmp_path,
        env=_clean_env(),
    )
    assert result.returncode == 0, result.stderr
    assert "Gate: PASS" in result.stdout


def test_cli_flag_overrides_dotenv(tmp_path):
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "LLM_EVAL_MIN_PASS_RATE=0",
                "LLM_EVAL_MIN_TOKEN_F1=0",
                "LLM_EVAL_MIN_FAITHFULNESS=0",
                "LLM_EVAL_MAX_HALLUCINATION_RATE=1",
            ]
        ),
        encoding="utf-8",
    )
    result = _run(
        [
            "run",
            "--dataset",
            str(GOLDEN_FAIL),
            "--out",
            str(tmp_path / "out"),
            "--min-pass-rate",
            "1",
        ],
        cwd=tmp_path,
        env=_clean_env(),
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "Gate: FAIL" in result.stdout


def test_importing_the_package_does_not_import_fastapi():
    code = "import sys, llm_eval; raise SystemExit(1 if 'fastapi' in sys.modules else 0)"
    result = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
