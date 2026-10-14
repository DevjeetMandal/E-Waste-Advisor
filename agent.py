"""
The agent.

Phase 1 - PLAN: the model is given tools (classify_device, search_rules,
find_recyclers) and decides which to call and with what queries. Tool results
are executed by code, never by the model.

Phase 2 - ANSWER: a separate, tool-free call receives the gathered evidence as
numbered excerpts [E1], [E2]... and must return JSON matching schema.py. Code
then validates it: unknown citations are dropped, and the model cannot claim
the rules cover a device without a real citation.

Guarantees that do NOT depend on the model behaving:
  * classification and at least one rule search always happen (forced if skipped)
  * a similarity gate (config.MIN_SCORE) marks the answer "not covered" when
    retrieval is weak
  * recycler names come from the CSV, never from the model
  * safety hints are rule-based (safety.py)
"""

import hashlib
import json
import re
from dataclasses import dataclass

import config
import recyclers as recycler_db
import schema
from classify import classify_with_evidence
from llm import LLMBadRequest
from safety import safety_flags

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "classify_device",
            "description": "Map a free-text device description to a broad e-waste group using transparent keyword rules.",
            "parameters": {
                "type": "object",
                "properties": {"description": {"type": "string"}},
                "required": ["description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_rules",
            "description": (
                "Semantic search over India's official e-waste documents (E-Waste Management Rules 2022, "
                "CPCB category list). Use focused queries; you may call this several times, e.g. once for "
                "the device and once for a hazardous component like its battery."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "k": {"type": "integer", "minimum": 1, "maximum": 6},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_recyclers",
            "description": "Look up CPCB-listed authorised dismantlers/recyclers. Only call if the user gave a state or city.",
            "parameters": {
                "type": "object",
                "properties": {"state": {"type": "string"}, "city": {"type": "string"}},
            },
        },
    },
]

PLANNER_SYSTEM = (
    "You are the planning step of an e-waste guidance assistant for users in India. Gather evidence "
    "with the tools. Always call classify_device and search_rules (several focused searches are better "
    "than one vague one, e.g. the device itself and any hazardous part such as a battery). Call "
    "find_recyclers only if the user supplied a state or city. Do not write the final answer. "
    "When you have enough evidence, reply with the single word DONE."
)

ANSWER_SYSTEM = (
    "You write disposal guidance for an Indian household user. Respond with ONE JSON object matching the "
    "schema below and nothing else.\n"
    "Rules:\n"
    "- Statements about legal obligations, authorised channels or take-back must be supported by the "
    "numbered evidence excerpts. Put the supporting IDs (e.g. \"E2\") in evidence_ids / citations. "
    "Never invent an ID.\n"
    "- recoverable_materials and risks may use widely known general knowledge about electronics.\n"
    "- Do not name specific companies or addresses; local recyclers are added separately.\n"
    "- If retrieval status is NOT GROUNDED or the evidence does not address this device, set "
    "covered_by_rules=false, confidence=low, explain in not_covered_note, and give only cautious general advice.\n"
    "- Be concise and practical. summary is 1-2 sentences.\n\n"
    "JSON schema:\n" + schema.schema_as_text()
)



def assess_grounding(classification, evidence):
    """Decide whether retrieval is strong enough to let the answer claim rule coverage.

    Returns (grounded, reason, best_cosine). Two ways to pass:
      1. best cosine similarity clears the bar (stricter if the query named no device), or
      2. a retrieved passage literally names the device the classifier found - this rescues
         item-list tables that dense embeddings represent poorly.
    Short keywords (< 5 letters: "ac", "tv", "iron"...) are ignored for (2); they are too
    ambiguous in legal text.
    """
    best = max((h["score"] for h in evidence), default=0.0)
    named = [k for k in classification["matched"] if len(k) >= 5]
    bar = config.MIN_SCORE if classification["matched"] else config.MIN_SCORE_NO_DEVICE
    if best >= bar:
        return True, "similarity", best
    for h in evidence:
        text = h["text"].lower()
        for kw in named:
            if re.search(r"(?<![a-z0-9])" + re.escape(kw) + r"(?:s|es)?(?![a-z0-9])", text):
                return True, f"'{kw}' is named in a retrieved passage", best
    return False, "weak retrieval", best


@dataclass
class AgentResult:
    description: str
    classification: dict
    result: dict
    evidence: list
    recyclers: dict | None
    trace: list
    safety: list
    grounded: bool
    best_score: float
    grounded_reason: str = ""
    model: str = ""
    language: str = "English"


def _trim(text, n):
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


class Agent:
    def __init__(self, llm, index, recycler_rows=None):
        self.llm = llm
        self.index = index
        self.recycler_rows = recycler_rows

    # ---- tool implementations ------------------------------------------------
    def _t_classify(self, args, state):
        out = classify_with_evidence(args.get("description") or state["description"])
        state["classification"] = out
        return out, f"{out['category']} (matched: {', '.join(out['matched']) or 'nothing - default group'})"

    def _t_search(self, args, state):
        query = (args.get("query") or state["description"]).strip()
        k = max(1, min(int(args.get("k") or config.TOP_K), 6))
        hits = self.index.search(query, k=k)
        for h in hits:
            key = hashlib.md5(f"{h['source']}|{h['page']}|{h['text']}".encode()).hexdigest()
            prev = state["evidence"].get(key)
            if prev is None or h["fused"] > prev["fused"]:
                state["evidence"][key] = h
        state["searched"] = True
        best = max((h["score"] for h in hits), default=0.0)
        payload = [
            {"source": h["source"], "page": h["page"], "score": h["score"], "found_by": h["via"],
             "text": _trim(h["text"], 500)}
            for h in hits
        ]
        return {"results": payload}, f"'{_trim(query, 60)}' → {len(hits)} chunks, best similarity {best:.2f}"

    def _t_recyclers(self, args, state):
        res = recycler_db.find_recyclers(args.get("state"), args.get("city"), rows=self.recycler_rows)
        state["recyclers"] = res
        state["recycler_called"] = True
        n = len(res["matches"])
        where = res.get("state") or "no state"
        return (
            {"matches": [{"name": m["name"], "city": m.get("city") or m.get("district")} for m in res["matches"]],
             "note": res["note"]},
            f"{n} listed recyclers for {where}",
        )

    def _execute(self, name, args, state, forced=False):
        fn = {"classify_device": self._t_classify, "search_rules": self._t_search,
              "find_recyclers": self._t_recyclers}.get(name)
        if fn is None:
            result, summary = {"error": f"unknown tool {name}"}, f"unknown tool '{name}'"
        else:
            try:
                result, summary = fn(args, state)
            except Exception as e:  # tool errors go back to the model, never crash the request
                result, summary = {"error": str(e)}, f"error: {e}"
        state["trace"].append({"tool": name, "args": args, "summary": summary, "forced": forced})
        return result

    # ---- main ------------------------------------------------------------------
    def run(self, description, state_name="", city="", language="English"):
        st = {
            "description": description, "classification": None, "evidence": {}, "recyclers": None,
            "trace": [], "searched": False, "recycler_called": False,
        }
        have_location = bool((state_name or "").strip() or (city or "").strip())

        messages = [
            {"role": "system", "content": PLANNER_SYSTEM},
            {"role": "user", "content": (
                f"Device description: {description}\n"
                f"User location: state={state_name or ''}, city={city or ''}")},
        ]

        model_used = ""
        try:
            for _ in range(config.MAX_AGENT_STEPS):
                resp = self.llm.chat(messages, tools=TOOLS)
                model_used = resp.model or model_used
                if not resp.tool_calls:
                    break
                messages.append({
                    "role": "assistant", "content": resp.content or None,
                    "tool_calls": [{"id": c.id, "type": "function",
                                    "function": {"name": c.name, "arguments": c.arguments}}
                                   for c in resp.tool_calls],
                })
                for c in resp.tool_calls:
                    try:
                        args = json.loads(c.arguments or "{}")
                        if not isinstance(args, dict):
                            args = {}
                    except json.JSONDecodeError:
                        args = {}
                    result = self._execute(c.name, args, st)
                    messages.append({"role": "tool", "tool_call_id": c.id,
                                     "content": json.dumps(result)[:6000]})
        except LLMBadRequest as e:
            st["trace"].append({"tool": "planner", "args": {}, "forced": True,
                                "summary": f"tool-calling unavailable ({_trim(str(e), 80)}); using the fixed pipeline"})

        # Deterministic safety net
        if st["classification"] is None:
            self._execute("classify_device", {"description": description}, st, forced=True)
        if not st["searched"]:
            self._execute("search_rules", {"query": description}, st, forced=True)
        if have_location and not st["recycler_called"]:
            self._execute("find_recyclers", {"state": state_name, "city": city}, st, forced=True)

        # Evidence ranking -> E1..En
        ranked = sorted(st["evidence"].values(), key=lambda h: h["fused"], reverse=True)[:6]
        evidence_by_id = {f"E{i}": h for i, h in enumerate(ranked, start=1)}
        grounded, grounded_reason, best_score = assess_grounding(st["classification"], ranked)

        result, model_used = self._answer(description, st["classification"], evidence_by_id,
                                          grounded, grounded_reason, best_score, language, model_used)

        evidence_list = [{"id": k, **v} for k, v in evidence_by_id.items()]
        return AgentResult(
            description=description, classification=st["classification"], result=result,
            evidence=evidence_list, recyclers=st["recyclers"], trace=st["trace"],
            safety=safety_flags(description), grounded=grounded, best_score=best_score,
            grounded_reason=grounded_reason, model=model_used, language=language,
        )

    def _answer(self, description, classification, evidence_by_id, grounded, grounded_reason, best_score,
                language, model_used):
        ev_text = "\n\n".join(
            f"[{eid}] (source: {h['source']}, page {h['page'] if h['page'] else 'n/a'}, similarity {h['score']:.2f}, found by {h['via']})\n{_trim(h['text'], 900)}"
            for eid, h in evidence_by_id.items()
        ) or "(no evidence retrieved)"
        status = (f"GROUNDED (best similarity {best_score:.2f}; {grounded_reason})" if grounded
                  else f"NOT GROUNDED (best similarity {best_score:.2f} is below the confidence threshold)")
        lang_line = (
            "Write all human-readable string values in English." if language == "English"
            else f"Write all human-readable string values in {language}. Keep enum values "
                 "(hazard_level, confidence) and evidence IDs in English."
        )
        prompt = (
            f"Device description: {description}\n"
            f"Category (rule-based classifier): {classification['category']} "
            f"(matched: {', '.join(classification['matched']) or 'none'})\n"
            f"Retrieval status: {status}\n{lang_line}\n\nEvidence:\n{ev_text}"
        )
        messages = [{"role": "system", "content": ANSWER_SYSTEM}, {"role": "user", "content": prompt}]

        raw, last_err = None, ""
        for fmt in (schema.json_schema_format(), {"type": "json_object"}, None):
            try:
                resp = self.llm.chat(messages, response_format=fmt)
                model_used = resp.model or model_used
                raw = schema.parse_json(resp.content)
                break
            except LLMBadRequest as e:
                last_err = str(e)
                continue
            except ValueError as e:
                last_err = f"unparseable model output ({e})"
                continue
        if raw is None:
            return schema.fallback_result(
                "The AI response could not be processed. Please retry. " + _trim(last_err, 120)), model_used
        return schema.normalize_result(raw, evidence_by_id, grounded), model_used
