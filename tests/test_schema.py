import schema

EVIDENCE = {"E1": {"source": "rules.pdf", "page": 7, "score": 0.6, "text": "Producers shall ..."}}


def base(**over):
    d = {
        "summary": "s", "hazard_level": "high", "hazard_reason": "r", "recoverable_materials": ["copper"],
        "risks": ["lead"], "do": ["a"], "dont": ["b"], "covered_by_rules": True, "not_covered_note": "",
        "confidence": "high",
        "disposal_options": [{"channel": "Take-back", "details": "d", "evidence_ids": ["E1"]}],
        "citations": [{"evidence_id": "E1", "supports": "take-back duty"}],
    }
    d.update(over)
    return d


def test_valid_answer_passes_with_page():
    out = schema.normalize_result(base(), EVIDENCE, grounded=True)
    assert out["covered_by_rules"] and out["citations"][0]["page"] == 7
    assert out["disposal_options"][0]["cited"]


def test_invented_citation_ids_are_dropped_and_claim_downgraded():
    raw = base(citations=[{"evidence_id": "E9", "supports": "x"}],
               disposal_options=[{"channel": "c", "details": "d", "evidence_ids": ["E9"]}])
    out = schema.normalize_result(raw, EVIDENCE, grounded=True)
    assert out["citations"] == []
    assert out["covered_by_rules"] is False and out["confidence"] == "low"
    assert out["disposal_options"][0]["cited"] is False


def test_not_grounded_forces_not_covered_even_if_model_insists():
    out = schema.normalize_result(base(), EVIDENCE, grounded=False)
    assert out["covered_by_rules"] is False and out["confidence"] == "low"
    assert out["not_covered_note"]


def test_bad_enums_and_types_are_sanitised():
    out = schema.normalize_result({"hazard_level": "catastrophic", "risks": "not a list", "confidence": 5},
                                  EVIDENCE, grounded=True)
    assert out["hazard_level"] == "medium" and out["risks"] == [] and out["confidence"] == "low"


def test_parse_json_tolerates_fences_and_chatter():
    assert schema.parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert schema.parse_json('Sure! Here you go: {"a": 2} hope it helps') == {"a": 2}


def test_parse_json_rejects_garbage():
    import pytest
    with pytest.raises(ValueError):
        schema.parse_json("no json here")
