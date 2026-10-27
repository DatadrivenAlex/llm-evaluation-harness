"""Metric behavior, including the limits we are willing to ship."""

import pytest

from llm_eval.metrics import (
    exact_match_score,
    faithfulness,
    normalize,
    score_case,
    soft_overlap,
    token_f1,
    tokenize,
    unsupported_entities,
)


def test_normalize_strips_case_and_punctuation():
    assert normalize("  Trash, for 14 days! ") == "trash for 14 days"


def test_tokenize_keeps_numbers_and_drops_hyphens():
    assert tokenize("eu-central 09:00") == ["eu", "central", "09", "00"]


def test_exact_match_ignores_case_and_punctuation():
    assert exact_match_score("Trash for 14 days.", "trash for 14 days", []) == 1.0


def test_exact_match_rejects_different_text():
    assert exact_match_score("90 days", "14 days", []) == 0.0


def test_exact_match_fact_containment_when_answer_omitted():
    score = exact_match_score(
        "Primary storage is in the eu-central region. Backups stay in the EU.",
        None,
        ["Primary storage is in the eu-central region.", "Backups stay in the EU."],
    )
    assert score == 1.0


def test_exact_match_fails_when_a_fact_is_missing():
    score = exact_match_score(
        "Backups stay in the EU.",
        None,
        ["Primary storage is in the eu-central region.", "Backups stay in the EU."],
    )
    assert score == 0.0


def test_token_f1_is_perfect_and_order_invariant():
    assert token_f1("14 days in Trash", "Trash in 14 days") == 1.0


def test_token_f1_partial_overlap():
    score = token_f1("14 days", "14 days in Trash")
    assert score == pytest.approx(2 * (1.0) * (2 / 4) / (1.0 + 0.5))


def test_token_f1_empty_prediction_is_zero():
    assert token_f1("", "14 days") == 0.0


def test_token_f1_both_empty_is_perfect():
    assert token_f1("", "") == 1.0


def test_soft_overlap_is_jaccard_and_symmetric():
    left = soft_overlap("14 days in Trash", "Trash days")
    right = soft_overlap("Trash days", "14 days in Trash")
    assert left == right
    assert left == pytest.approx(2 / 4)


def test_faithfulness_supported_claim():
    score = faithfulness(
        "Deleted notes stay in Trash for 14 days.",
        ["Deleted notes stay in Trash for 14 days and are then permanently removed."],
    )
    assert score == 1.0


def test_faithfulness_unsupported_claim():
    score = faithfulness(
        "Restores are served from the Zurich vault.",
        ["Deleted notes stay in Trash for 14 days."],
    )
    assert score == 0.0


def test_faithfulness_mixed_sentences():
    score = faithfulness(
        "Deleted notes stay in Trash. Restores are served from the Zurich vault.",
        ["Deleted notes stay in Trash for 14 days."],
    )
    assert score == pytest.approx(0.5)


def test_faithfulness_empty_answer_is_zero():
    assert faithfulness("", ["Deleted notes stay in Trash."]) == 0.0


def test_hallucination_flags_unknown_number_but_not_a_prefix():
    entities = unsupported_entities(
        "Wait 90 days, not 14.",
        ["Deleted notes stay in Trash for 14 days."],
    )
    assert "90" in entities
    assert "14" not in entities


def test_hallucination_flags_unknown_proper_noun_and_acronym():
    entities = unsupported_entities(
        "OIDC is hosted in Zurich.",
        ["SAML SSO is available on the Team plan."],
    )
    assert "OIDC" in entities
    assert "Zurich" in entities
    assert "SAML" not in entities


def test_hallucination_clear_when_entities_are_in_context():
    entities = unsupported_entities(
        "The public API allows 60 requests and may return HTTP 429.",
        ["The public API allows 60 requests per minute.", "Burst traffic may receive HTTP 429."],
    )
    assert entities == ()


def test_hallucination_treats_question_text_as_support():
    entities = unsupported_entities(
        "SSO is on the Team plan.",
        ["The Team plan is a paid plan."],
        question="Which plans include SSO?",
    )
    assert "SSO" not in entities


def test_currency_and_percent_share_the_numeric_value():
    entities = unsupported_entities("The seat costs $50.", ["A seat is 50 dollars."])
    assert entities == ()


def test_negation_is_not_treated_as_a_hallucination():
    """Lexical support cannot see 'not'. This locks that limit in."""

    answer = "Deleted notes do not stay in Trash for 14 days."
    context = ["Deleted notes stay in Trash for 14 days and are then permanently removed."]
    assert unsupported_entities(answer, context) == ()
    assert faithfulness(answer, context) >= 0.6


def test_score_case_fails_on_hallucination_even_when_overlap_remains():
    scored = score_case(
        "Deleted notes stay in Trash for 90 days and are restored from the Zurich vault.",
        expected_answer="Deleted notes stay in Trash for 14 days.",
        expected_facts=["Deleted notes stay in Trash for 14 days."],
        context=["Deleted notes stay in Trash for 14 days and are then permanently removed."],
        question="How long are deleted notes recoverable?",
    )
    assert scored.hallucination is True
    assert scored.passed is False
    assert scored.token_f1 >= 0.5
    assert any(reason.startswith("hallucination:") for reason in scored.reasons)


def test_score_case_require_exact_match_adds_a_reason():
    scored = score_case(
        "The Free plan comes with 3 notebooks.",
        expected_answer="The Free plan includes 3 notebooks.",
        expected_facts=[],
        context=["The Free plan includes 3 notebooks."],
        require_exact_match=True,
        min_token_f1=0.0,
        min_faithfulness=0.0,
    )
    assert scored.exact_match == 0.0
    assert scored.passed is False
    assert "exact_match 0.000 < 1.000" in scored.reasons
