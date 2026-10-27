"""Optional FastAPI demo. The core package does not import this module."""

from fastapi.testclient import TestClient

from llm_eval.api import app


client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "llm-evaluation-harness"
    assert body["version"]


def test_evaluate_endpoint_scores_a_grounded_answer():
    response = client.post(
        "/v1/evaluate",
        json={
            "cases": [
                {
                    "id": "trash",
                    "question": "How long are deleted notes recoverable?",
                    "expected_answer": "Deleted notes stay in Trash for 14 days.",
                    "context": ["Deleted notes stay in Trash for 14 days."],
                    "recorded_answer": "Deleted notes stay in Trash for 14 days.",
                }
            ]
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["dataset"] == "request"
    assert body["gate"]["passed"] is True
    assert body["summary"]["pass_rate"] == 1.0
    assert body["cases"][0]["metrics"]["exact_match"] == 1.0


def test_evaluate_endpoint_reports_a_failed_gate_with_http_200():
    response = client.post(
        "/v1/evaluate",
        json={
            "cases": [
                {
                    "question": "How long are deleted notes recoverable?",
                    "expected_answer": "Deleted notes stay in Trash for 14 days.",
                    "context": ["Deleted notes stay in Trash for 14 days."],
                    "recorded_answer": "Notes stay in the Zurich vault for 90 days.",
                }
            ]
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["gate"]["passed"] is False
    assert body["cases"][0]["metrics"]["hallucination"] is True
    assert body["cases"][0]["id"] == "case-1"


def test_evaluate_rejects_a_case_without_an_oracle():
    response = client.post("/v1/evaluate", json={"cases": [{"question": "Hi?"}]})
    assert response.status_code == 422
    assert "expected_answer" in response.json()["detail"]


def test_evaluate_rejects_a_threshold_above_one():
    response = client.post(
        "/v1/evaluate",
        json={
            "cases": [
                {
                    "question": "Q?",
                    "expected_answer": "A",
                    "recorded_answer": "A",
                }
            ],
            "min_pass_rate": 2,
        },
    )
    assert response.status_code == 422
