"""
Evaluation harness. Produces numbers for the "Test & Refine" part of your write-up
and writes them to eval_report.md.

    python eval.py              # classifier, retrieval, and the real grounding gate (free, offline)
    python eval.py --e2e 8      # also run the full agent on 8 queries (uses Groq calls)

The retrieval `expect_any` keywords in data/eval_cases.json are starter guesses:
edit them to wording that really appears in your PDFs. A "miss" means no expected
keyword in the top-k chunks - always eyeball it with `python diagnose.py "<query>"`
before concluding retrieval failed.
"""

import argparse
import json
import statistics
from collections import Counter
from datetime import date

import config
import rag_utils
from agent import assess_grounding
from classify import classify_device, classify_with_evidence


def load_cases():
    with open(f"{config.DATA_DIR}/eval_cases.json", encoding="utf-8") as f:
        return json.load(f)


def best_cosine(hits):
    return max((h["score"] for h in hits), default=0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e2e", type=int, default=0, help="run the full agent on N in-domain queries")
    ap.add_argument("--k", type=int, default=config.TOP_K)
    args = ap.parse_args()

    cases = load_cases()
    index = rag_utils.get_index()
    k = args.k

    L = [f"# Evaluation report ({date.today().isoformat()})", "",
         f"Index: {index.meta.get('embedder')} · {index.meta.get('chunks')} chunks"
         + (" · **PLACEHOLDER knowledge base**" if index.meta.get("placeholder") else ""), ""]
    L.append("Chunks per source: " + ", ".join(f"{s} = {n}" for s, n in Counter(c["source"] for c in index.chunks).items()))

    # ---- classifier -----------------------------------------------------------------
    wrong = [(t, e, classify_device(t)) for t, e in cases["classifier"] if classify_device(t) != e]
    n = len(cases["classifier"])
    L += ["", "## Classifier", f"{n - len(wrong)}/{n} correct ({100 * (n - len(wrong)) / n:.0f}%)"]
    L += [f"- `{t}` → got **{g}**, expected {e}" for t, e, g in wrong]

    # ---- retrieval keyword hit-rate ---------------------------------------------------
    misses = []
    for c in cases["retrieval"]:
        blob = " ".join(h["text"].lower() for h in index.search(c["query"], k=k))
        if not any(kw.lower() in blob for kw in c["expect_any"]):
            misses.append(c["query"])
    m = len(cases["retrieval"])
    L += ["", f"## Retrieval hit@{k} (hybrid)", f"{m - len(misses)}/{m} queries had an expected keyword in the top-{k} chunks"]
    L += [f"- missed: `{q}`  →  inspect with `python diagnose.py \"{q}\"`" for q in misses]

    # ---- the real grounding gate -----------------------------------------------------------
    def gate(q):
        hits = index.search(q, k=k)
        cls = classify_with_evidence(q)
        ok, why, best = assess_grounding(cls, hits)
        return ok, why, best, bool(cls["matched"])

    in_rows = [(c["query"], *gate(c["query"])) for c in cases["retrieval"]]
    ood_rows = [(q, *gate(q)) for q in cases["out_of_domain"]]
    refused = [r for r in in_rows if not r[1]]
    leaked = [r for r in ood_rows if r[1]]

    L += ["", "## Grounding gate (the real one: similarity bar + device-named rescue)",
          f"- MIN_SCORE = {config.MIN_SCORE} (device named) · MIN_SCORE_NO_DEVICE = {config.MIN_SCORE_NO_DEVICE}",
          f"- in-domain queries wrongly refused: **{len(refused)}/{len(in_rows)}**",
          f"- out-of-domain queries wrongly accepted: **{len(leaked)}/{len(ood_rows)}**"]
    for q, ok, why, best, _ in refused:
        L.append(f"  - refused in-domain: `{q}` (best cosine {best:.2f})")
    for q, ok, why, best, _ in leaked:
        L.append(f"  - accepted off-topic: `{q}` (best cosine {best:.2f}; {why})")

    ind = [r[3] for r in in_rows]
    ood = [r[3] for r in ood_rows]
    L += ["", "## Raw cosine distributions (for tuning)",
          f"- in-domain: min {min(ind):.2f} · median {statistics.median(ind):.2f} · max {max(ind):.2f}",
          f"- out-of-domain: min {min(ood):.2f} · median {statistics.median(ood):.2f} · max {max(ood):.2f}"]
    if min(ind) > max(ood):
        L.append(f"- clean separation: around **{(min(ind) + max(ood)) / 2:.2f}** splits the groups")
    else:
        L.append("- the groups overlap on raw similarity alone; that is why the gate also uses "
                 "device naming and a stricter bar for queries with no device. Prefer refusing over a confident wrong answer.")

    # ---- optional end-to-end -----------------------------------------------------------------
    if args.e2e:
        from agent import Agent
        from llm import get_llm
        agent = Agent(get_llm(), index)
        qs = [c["query"] for c in cases["retrieval"]][: args.e2e]
        cited = backed = 0
        res = None
        for q in qs:
            res = agent.run(q)
            cited += bool(res.result["citations"])
            backed += res.result["covered_by_rules"]
        L += ["", f"## End-to-end ({len(qs)} in-domain queries, model {res.model})",
              f"- answers with at least one valid citation: {cited}/{len(qs)}",
              f"- answers marked rule-backed: {backed}/{len(qs)}"]

    report = "\n".join(L)
    print(report)
    with open("eval_report.md", "w", encoding="utf-8") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()
