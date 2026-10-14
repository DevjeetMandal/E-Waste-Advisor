"""
Structured output contract between the LLM and the UI, plus validation.

The point of normalize_result() is integrity: the model may only cite evidence
IDs that were actually retrieved, and it cannot claim the rules "cover" a
device unless at least one valid citation backs the claim.
"""

import json
import re

HAZARD_LEVELS = ("low", "medium", "high")
CONFIDENCE_LEVELS = ("high", "medium", "low")

RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "hazard_level": {"type": "string", "enum": list(HAZARD_LEVELS)},
        "hazard_reason": {"type": "string"},
        "recoverable_materials": {"type": "array", "items": {"type": "string"}},
        "risks": {"type": "array", "items": {"type": "string"}},
        "do": {"type": "array", "items": {"type": "string"}},
        "dont": {"type": "array", "items": {"type": "string"}},
        "disposal_options": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "channel": {"type": "string"},
                    "details": {"type": "string"},
                    "evidence_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["channel", "details", "evidence_ids"],
            },
        },
        "covered_by_rules": {"type": "boolean"},
        "not_covered_note": {"type": "string"},
        "confidence": {"type": "string", "enum": list(CONFIDENCE_LEVELS)},
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"evidence_id": {"type": "string"}, "supports": {"type": "string"}},
                "required": ["evidence_id", "supports"],
            },
        },
    },
    "required": [
        "summary", "hazard_level", "hazard_reason", "recoverable_materials", "risks", "do", "dont",
        "disposal_options", "covered_by_rules", "not_covered_note", "confidence", "citations",
    ],
}


def json_schema_format():
    return {
        "type": "json_schema",
        "json_schema": {"name": "ewaste_guidance", "strict": False, "schema": RESULT_SCHEMA},
    }


def schema_as_text():
    return json.dumps(RESULT_SCHEMA, indent=1)


def parse_json(text):
    """Tolerant JSON extraction: handles code fences and chatter around the object."""
    if not text:
        raise ValueError("empty response")
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        start, end = t.find("{"), t.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("no JSON object found")
        return json.loads(t[start : end + 1])


def _s(v, limit=500):
    return str(v).strip()[:limit] if v is not None else ""


def _list(v, limit=8, item_limit=300):
    if not isinstance(v, list):
        return []
    return [_s(x, item_limit) for x in v if _s(x)][:limit]


def normalize_result(raw, evidence_by_id, grounded):
    """raw: parsed model JSON. evidence_by_id: {"E1": {source,page,score,text}, ...}."""
    raw = raw if isinstance(raw, dict) else {}

    def valid_ids(ids):
        return [i for i in (ids or []) if isinstance(i, str) and i in evidence_by_id]

    options = []
    for o in raw.get("disposal_options") or []:
        if not isinstance(o, dict):
            continue
        ids = valid_ids(o.get("evidence_ids"))
        options.append({
            "channel": _s(o.get("channel"), 120),
            "details": _s(o.get("details"), 500),
            "evidence_ids": ids,
            "cited": bool(ids),
        })
    options = [o for o in options if o["channel"]][:6]

    citations, seen = [], set()
    for c in raw.get("citations") or []:
        if not isinstance(c, dict):
            continue
        eid = c.get("evidence_id")
        if eid in evidence_by_id and eid not in seen:
            seen.add(eid)
            ev = evidence_by_id[eid]
            citations.append({
                "evidence_id": eid,
                "source": ev["source"],
                "page": ev.get("page"),
                "score": ev.get("score"),
                "supports": _s(c.get("supports"), 300),
                "excerpt": ev["text"][:700],
            })
    for o in options:  # option-level citations count as citations too
        for eid in o["evidence_ids"]:
            if eid not in seen:
                seen.add(eid)
                ev = evidence_by_id[eid]
                citations.append({
                    "evidence_id": eid, "source": ev["source"], "page": ev.get("page"),
                    "score": ev.get("score"), "supports": "Cited for: " + o["channel"],
                    "excerpt": ev["text"][:700],
                })

    hazard = raw.get("hazard_level") if raw.get("hazard_level") in HAZARD_LEVELS else "medium"
    confidence = raw.get("confidence") if raw.get("confidence") in CONFIDENCE_LEVELS else "low"
    covered = bool(raw.get("covered_by_rules"))
    note = _s(raw.get("not_covered_note"))

    # Integrity gates
    if not grounded:
        covered, confidence = False, "low"
        note = note or "The reference documents did not clearly cover this device."
    elif covered and not citations:
        covered, confidence = False, "low"
        note = (note + " " if note else "") + "The model did not cite a supporting source, so this is not treated as rule-backed."

    return {
        "summary": _s(raw.get("summary"), 600),
        "hazard_level": hazard,
        "hazard_reason": _s(raw.get("hazard_reason"), 400),
        "recoverable_materials": _list(raw.get("recoverable_materials"), 10, 60),
        "risks": _list(raw.get("risks")),
        "do": _list(raw.get("do")),
        "dont": _list(raw.get("dont")),
        "disposal_options": options,
        "covered_by_rules": covered,
        "not_covered_note": note,
        "confidence": confidence,
        "citations": citations,
    }


def fallback_result(message):
    """Used when the model output cannot be parsed at all."""
    return {
        "summary": message, "hazard_level": "medium", "hazard_reason": "", "recoverable_materials": [],
        "risks": [], "do": [], "dont": [], "disposal_options": [], "covered_by_rules": False,
        "not_covered_note": message, "confidence": "low", "citations": [],
    }
