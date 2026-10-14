import json

import pytest

import config
from agent import Agent
from llm import DemoLLM, LLMBadRequest, LLMResponse, ToolCall


class Scripted:
    """Plays back planner + answer behaviours so we can test misbehaving models."""
    model = "scripted"

    def __init__(self, planner="skip", answers=None, bad_tools=False):
        self.planner, self.answers, self.bad_tools = planner, list(answers or []), bad_tools
        self.answer_calls = 0

    def chat(self, messages, tools=None, response_format=None):
        if tools:
            if self.bad_tools:
                raise LLMBadRequest("tool_use_failed")
            return LLMResponse(content="DONE", model=self.model)  # model "forgets" to call tools
        self.answer_calls += 1
        out = self.answers.pop(0)
        return LLMResponse(content=out if isinstance(out, str) else json.dumps(out), model=self.model)


GOOD = {
    "summary": "ok", "hazard_level": "medium", "hazard_reason": "", "recoverable_materials": ["copper"],
    "risks": [], "do": [], "dont": [], "covered_by_rules": True, "not_covered_note": "", "confidence": "high",
    "disposal_options": [{"channel": "Take-back", "details": "d", "evidence_ids": ["E1"]}],
    "citations": [{"evidence_id": "E1", "supports": "s"}],
}


@pytest.fixture(autouse=True)
def low_threshold(monkeypatch):
    monkeypatch.setattr(config, "MIN_SCORE", 0.05)


def test_demo_agent_uses_all_three_tools_with_location(index):
    res = Agent(DemoLLM(), index).run("old laptop with swollen battery", state_name="Odisha", city="Bhubaneswar")
    tools = [t["tool"] for t in res.trace]
    assert tools == ["classify_device", "search_rules", "find_recyclers"]
    assert not any(t["forced"] for t in res.trace)
    assert res.recyclers["matches"][0]["name"] == "Cosmic Net"
    assert res.evidence and res.evidence[0]["id"] == "E1"
    assert any(f["level"] == "urgent" for f in res.safety)


def test_safeguards_force_tools_when_model_skips_them(index):
    llm = Scripted(answers=[GOOD])
    res = Agent(llm, index).run("refrigerators and air conditioners", state_name="Delhi")
    forced = {t["tool"] for t in res.trace if t["forced"]}
    assert forced == {"classify_device", "search_rules", "find_recyclers"}
    assert res.result["covered_by_rules"] is True


def test_no_recycler_lookup_without_location(index):
    res = Agent(Scripted(answers=[GOOD]), index).run("old refrigerator")
    assert "find_recyclers" not in [t["tool"] for t in res.trace]
    assert res.recyclers is None


def test_fabricated_citation_id_does_not_survive(index):
    bad = dict(GOOD, citations=[{"evidence_id": "E42", "supports": "x"}],
               disposal_options=[{"channel": "c", "details": "d", "evidence_ids": ["E42"]}])
    res = Agent(Scripted(answers=[bad]), index).run("old laptop")
    assert res.result["citations"] == [] and res.result["covered_by_rules"] is False


def test_grounding_gate_overrides_confident_model(index, monkeypatch):
    monkeypatch.setattr(config, "MIN_SCORE_NO_DEVICE", 0.99)
    res = Agent(Scripted(answers=[GOOD]), index).run("what is the airspeed of a swallow")  # no device named
    assert res.grounded is False
    assert res.result["covered_by_rules"] is False and res.result["confidence"] == "low"


def test_malformed_json_is_retried_with_next_format(index):
    llm = Scripted(answers=["this is not json at all", GOOD])
    res = Agent(llm, index).run("old laptop")
    assert llm.answer_calls == 2 and res.result["summary"] == "ok"


def test_total_failure_returns_safe_fallback_not_exception(index):
    llm = Scripted(answers=["junk", "junk", "junk"])
    res = Agent(llm, index).run("old laptop")
    assert res.result["covered_by_rules"] is False and "could not be processed" in res.result["summary"]


def test_tool_calling_failure_degrades_to_fixed_pipeline(index):
    res = Agent(Scripted(answers=[GOOD], bad_tools=True), index).run("old laptop")
    assert any("tool-calling unavailable" in t["summary"] for t in res.trace)
    assert {"classify_device", "search_rules"} <= {t["tool"] for t in res.trace}


def test_garbage_tool_arguments_do_not_crash(index):
    class BadArgs(Scripted):
        def chat(self, messages, tools=None, response_format=None):
            if tools and not any(m.get("role") == "tool" for m in messages):
                return LLMResponse(tool_calls=[ToolCall("x", "search_rules", "{not json"),
                                               ToolCall("y", "nonexistent_tool", "{}")])
            return super().chat(messages, tools, response_format)

    res = Agent(BadArgs(answers=[GOOD]), index).run("old laptop")
    assert any(t["tool"] == "nonexistent_tool" for t in res.trace)
    assert res.result["summary"] == "ok"
