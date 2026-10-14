"""Exercises GroqLLM against a fake client that raises the SDK's real exception types."""
import types

import groq
import httpx
import pytest

import config
from llm import GroqLLM, LLMBadRequest, LLMError


def _exc(cls, status):
    req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return cls("boom", response=httpx.Response(status, request=req), body=None)


def _ok(content="hi", tool_calls=None):
    msg = types.SimpleNamespace(content=content, tool_calls=tool_calls)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


class FakeClient:
    def __init__(self, behaviours):
        self.behaviours, self.seen = list(behaviours), []
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.seen.append(kw)
        b = self.behaviours.pop(0)
        if isinstance(b, Exception):
            raise b
        return b


@pytest.fixture(autouse=True)
def models(monkeypatch):
    monkeypatch.setattr(config, "GROQ_MODELS", ["retired-model", "good-model"])


def test_retired_model_404_falls_through_to_next_model():
    c = FakeClient([_exc(groq.NotFoundError, 404), _ok("answer")])
    r = GroqLLM(c).chat([{"role": "user", "content": "x"}])
    assert r.content == "answer" and r.model == "good-model"
    assert [k["model"] for k in c.seen] == ["retired-model", "good-model"]


def test_all_models_gone_gives_actionable_message():
    c = FakeClient([_exc(groq.NotFoundError, 404), _exc(groq.NotFoundError, 404)])
    with pytest.raises(LLMError, match="GROQ_MODEL"):
        GroqLLM(c).chat([{"role": "user", "content": "x"}])


def test_rate_limit_and_auth_are_user_readable():
    with pytest.raises(LLMError, match="rate-limited"):
        GroqLLM(FakeClient([_exc(groq.RateLimitError, 429)])).chat([])
    with pytest.raises(LLMError, match="API key"):
        GroqLLM(FakeClient([_exc(groq.AuthenticationError, 401)])).chat([])


def test_bad_request_is_distinguishable_for_fallbacks():
    with pytest.raises(LLMBadRequest):
        GroqLLM(FakeClient([_exc(groq.BadRequestError, 400)])).chat([])


def test_tool_calls_are_parsed_and_params_forwarded():
    tc = types.SimpleNamespace(id="t1", function=types.SimpleNamespace(name="search_rules", arguments='{"query":"q"}'))
    c = FakeClient([_ok(content=None, tool_calls=[tc])])
    r = GroqLLM(c).chat([], tools=[{"type": "function"}], response_format={"type": "json_object"})
    assert r.tool_calls[0].name == "search_rules" and r.tool_calls[0].arguments == '{"query":"q"}'
    kw = c.seen[0]
    assert kw["tool_choice"] == "auto" and kw["response_format"] == {"type": "json_object"}
    assert kw["max_completion_tokens"] == config.LLM_MAX_TOKENS
