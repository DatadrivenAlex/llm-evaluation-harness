# Evaluation report

- Generated: 2026-10-27T12:29:35+00:00
- Harness: llm-evaluation-harness 0.1.0
- Dataset: datasets/rag_faq.jsonl
- Cases: 8
- Passed: 8
- Pass rate: 100.0%
- Gate: PASS

## Thresholds

- min_pass_rate: 0.800
- min_token_f1: 0.500
- min_faithfulness: 0.600
- max_hallucination_rate: 0.200
- require_exact_match: false

## Mean scores

| Metric | Score |
| --- | ---: |
| exact_match | 0.875 |
| token_f1 | 0.984 |
| soft_overlap | 0.971 |
| faithfulness | 1.000 |
| hallucination_rate | 0.000 |

## Failing cases

No failing cases.

## Cases

### password-reset — PASS

- Question: How do I reset my password?
- Answer: Password reset links expire after 30 minutes. Request a new link from the sign-in page.
- Expected: Password reset links expire after 30 minutes. Request a new link from the sign-in page.
- exact_match: 1.000
- token_f1: 1.000
- soft_overlap: 1.000
- faithfulness: 1.000
- hallucination: no
- unsupported_entities: none

### free-plan — PASS

- Question: What does the Free plan include?
- Answer: The Free plan comes with 3 notebooks and 50 MB of attachments.
- Expected: The Free plan includes 3 notebooks and 50 MB of attachments.
- exact_match: 0.000
- token_f1: 0.870
- soft_overlap: 0.769
- faithfulness: 1.000
- hallucination: no
- unsupported_entities: none

### sso — PASS

- Question: Which plans include SSO?
- Answer: SAML SSO is available on the Team plan. OIDC is available on the Enterprise plan.
- Expected: SAML SSO is available on the Team plan. OIDC is available on the Enterprise plan.
- exact_match: 1.000
- token_f1: 1.000
- soft_overlap: 1.000
- faithfulness: 1.000
- hallucination: no
- unsupported_entities: none

### export — PASS

- Question: How do I export my notes?
- Answer: Open Settings, choose Export, and download a ZIP of markdown files. Exports are ready within 1 hour.
- Expected: Open Settings, choose Export, and download a ZIP of markdown files. Exports are ready within 1 hour.
- exact_match: 1.000
- token_f1: 1.000
- soft_overlap: 1.000
- faithfulness: 1.000
- hallucination: no
- unsupported_entities: none

### support-hours — PASS

- Question: When is email support available?
- Answer: Email support runs Monday to Friday, 09:00 to 17:00 UTC. Enterprise customers also get weekend chat.
- Expected: Email support runs Monday to Friday, 09:00 to 17:00 UTC. Enterprise customers also get weekend chat.
- exact_match: 1.000
- token_f1: 1.000
- soft_overlap: 1.000
- faithfulness: 1.000
- hallucination: no
- unsupported_entities: none

### trash — PASS

- Question: How long are deleted notes recoverable?
- Answer: Deleted notes stay in Trash for 14 days.
- Expected: Deleted notes stay in Trash for 14 days.
- exact_match: 1.000
- token_f1: 1.000
- soft_overlap: 1.000
- faithfulness: 1.000
- hallucination: no
- unsupported_entities: none

### rate-limit — PASS

- Question: What is the API rate limit?
- Answer: The public API allows 60 requests per minute per token. Burst traffic may receive HTTP 429.
- Expected: The public API allows 60 requests per minute per token. Burst traffic may receive HTTP 429.
- exact_match: 1.000
- token_f1: 1.000
- soft_overlap: 1.000
- faithfulness: 1.000
- hallucination: no
- unsupported_entities: none

### storage-region — PASS

- Question: Where is my data stored?
- Answer: Primary storage is in the eu-central region. Backups stay in the EU.
- Expected facts: Primary storage is in the eu-central region. Backups stay in the EU.
- exact_match: 1.000
- token_f1: 1.000
- soft_overlap: 1.000
- faithfulness: 1.000
- hallucination: no
- unsupported_entities: none
