"""Paths shared by tests. Kept out of conftest so test modules can import them."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"
SAMPLE_JSONL = DATASETS / "rag_faq.jsonl"
SAMPLE_YAML = DATASETS / "rag_faq.yaml"
GOLDEN_FAIL = DATASETS / "golden_fail.jsonl"
