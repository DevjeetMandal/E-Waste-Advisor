"""
Thin LLM layer.

GroqLLM talks to Groq and, if a model has been retired (404), silently moves
to the next one in GROQ_MODEL. DemoLLM is a deterministic stand-in used only by
the test suite / offline UI preview - its output is clearly marked as demo.
"""

import json
import re
from dataclasses import dataclass, field

import config


class LLMError(Exception):
    """Something the user can be told about (rate limit, bad key, no network)."""


class LLMBadRequest(LLMError):
    """The API rejected the request shape (e.g. tool call failed, unsupported format)."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON string


@dataclass
class LLMResponse:
    content: str = ""
    tool_calls: list = field(default_factory=list)
    model: str = ""


class GroqLLM:
    def __init__(self, client=None):
        if client is None:
            from groq import Groq

            client = Groq()
        self.client = client

    def chat(self, messages, tools=None, response_format=None):
        import groq

        last_404 = None
        for model in config.GROQ_MODELS:
            kwargs = dict(
                model=model,
                messages=messages,
                temperature=0.2,
                max_completion_tokens=config.LLM_MAX_TOKENS,
            )
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            if response_format:
                kwargs["response_format"] = response_format
            try:
                resp = self.client.chat.completions.create(**kwargs)
            except groq.NotFoundError as e:
                last_404 = e
                continue
            except groq.RateLimitError as e:
                raise LLMError("The free Groq tier is rate-limited right now. Wait a minute and retry.") from e
            except groq.AuthenticationError as e:
                raise LLMError("Groq rejected the API key. Check GROQ_API_KEY in your .env.") from e
            except groq.APIConnectionError as e:
                raise LLMError("Could not reach Groq. Check your internet connection.") from e
            except groq.BadRequestError as e:
                raise LLMBadRequest(str(e)) from e
            except groq.APIStatusError as e:
                raise LLMError(f"Groq returned an error ({e.status_code}).") from e

            msg = resp.choices[0].message
            calls = [
                ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments or "{}")
                for tc in (getattr(msg, "tool_calls", None) or [])
            ]
            return LLMResponse(content=msg.content or "", tool_calls=calls, model=model)

        raise LLMError(
            "None of the configured Groq models are available "
            f"({', '.join(config.GROQ_MODELS)}). Update GROQ_MODEL in .env - see console.groq.com/docs/models."
        ) from last_404


class DemoLLM:
    """Deterministic fake. Never use for real answers."""

    model = "demo"

    def chat(self, messages, tools=None, response_format=None):
        if tools:
            already = any(m.get("role") == "tool" for m in messages)
            if already:
                return LLMResponse(content="DONE", model=self.model)
            user = next(m["content"] for m in messages if m["role"] == "user")
            desc = re.search(r"Device description: (.*)", user).group(1)
            calls = [
                ToolCall("c1", "classify_device", json.dumps({"description": desc})),
                ToolCall("c2", "search_rules", json.dumps({"query": desc})),
            ]
            loc = re.search(r"state=([^,\n]*), city=([^\n]*)", user)
            if loc and (loc.group(1).strip() or loc.group(2).strip()):
                calls.append(ToolCall("c3", "find_recyclers", json.dumps(
                    {"state": loc.group(1).strip(), "city": loc.group(2).strip()})))
            return LLMResponse(tool_calls=calls, model=self.model)

        prompt = messages[-1]["content"]
        has_e1 = "[E1]" in prompt
        grounded = "Retrieval status: GROUNDED" in prompt
        out = {
            "summary": "DEMO MODE - this is a canned response, not a real model answer.",
            "hazard_level": "medium",
            "hazard_reason": "Demo placeholder.",
            "recoverable_materials": ["copper", "aluminium"],
            "risks": ["Demo placeholder risk."],
            "do": ["Use an authorised collection point."],
            "dont": ["Do not sell to informal scrap dealers."],
            "disposal_options": [{
                "channel": "Authorised recycler / take-back",
                "details": "Demo placeholder.",
                "evidence_ids": ["E1"] if has_e1 else [],
            }],
            "covered_by_rules": bool(has_e1 and grounded),
            "not_covered_note": "" if grounded else "Demo: evidence too weak.",
            "confidence": "medium" if grounded else "low",
            "citations": [{"evidence_id": "E1", "supports": "Demo citation."}] if has_e1 else [],
        }
        return LLMResponse(content=json.dumps(out), model=self.model)


def get_llm():
    if config.LLM_BACKEND == "fake":
        return DemoLLM()
    return GroqLLM()
