# Evaluation report

- Generated: 2026-10-27T12:29:35+00:00
- Harness: llm-evaluation-harness 0.1.0
- Dataset: datasets/golden_fail.jsonl
- Cases: 1
- Passed: 0
- Pass rate: 0.0%
- Gate: FAIL

## Thresholds

- min_pass_rate: 0.800
- min_token_f1: 0.500
- min_faithfulness: 0.600
- max_hallucination_rate: 0.200
- require_exact_match: false

## Mean scores

| Metric | Score |
| --- | ---: |
| exact_match | 0.000 |
| token_f1 | 0.609 |
| soft_overlap | 0.438 |
| faithfulness | 1.000 |
| hallucination_rate | 1.000 |

## Gate reasons

- pass_rate 0.0000 < min_pass_rate 0.8000
- hallucination_rate 1.0000 > max_hallucination_rate 0.2000

## Failing cases

### trash-hallucination — FAIL

- Question: How long are deleted notes recoverable?
- Answer: Deleted notes stay in Trash for 90 days and are restored from the Zurich vault.
- Expected: Deleted notes stay in Trash for 14 days.
- exact_match: 0.000
- token_f1: 0.609
- soft_overlap: 0.438
- faithfulness: 1.000
- hallucination: yes
- unsupported_entities: 90, Zurich
- Reasons: hallucination: 90, Zurich

## Cases

### trash-hallucination — FAIL

- Question: How long are deleted notes recoverable?
- Answer: Deleted notes stay in Trash for 90 days and are restored from the Zurich vault.
- Expected: Deleted notes stay in Trash for 14 days.
- exact_match: 0.000
- token_f1: 0.609
- soft_overlap: 0.438
- faithfulness: 1.000
- hallucination: yes
- unsupported_entities: 90, Zurich
- Reasons: hallucination: 90, Zurich
