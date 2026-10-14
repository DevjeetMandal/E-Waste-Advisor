"""
Shows what retrieval actually returns, so thresholds and misses can be judged by eye.
Run from the project folder.

    python diagnose.py                       # best chunk + gate decision for every eval query
    python diagnose.py "old CRT television"  # top chunks (full text) for one query
    python diagnose.py --find refrigerator   # is this word in the index at all?
"""

import json
import sys

import config
import rag_utils
from agent import assess_grounding
from classify import classify_with_evidence


def show(hits, full=False):
    for h in hits:
        page = f"p.{h['page']}" if h["page"] else "p.?"
        body = h["text"] if full else h["text"][:100]
        print(f"  cos {h['score']:.2f}  via {h['via']:<8} {h['source']} {page}  {body!r}")


def report(index, q):
    hits = index.search(q, k=config.TOP_K)
    ok, why, best = assess_grounding(classify_with_evidence(q), hits)
    print(f"{q}\n  -> {'GROUNDED' if ok else 'REFUSED'} ({why}; best cosine {best:.2f})")
    show(hits[:2])


def main():
    index = rag_utils.get_index()
    args = sys.argv[1:]

    if args[:1] == ["--find"] and len(args) > 1:
        word = args[1].lower()
        found = [(c["source"], c["page"]) for c in index.chunks if word in c["text"].lower()]
        print(f"'{word}' appears in {len(found)} of {len(index.chunks)} chunks")
        print(found[:15])
        return

    if args:
        show(index.search(" ".join(args), k=config.TOP_K), full=True)
        return

    with open(f"{config.DATA_DIR}/eval_cases.json", encoding="utf-8") as f:
        cases = json.load(f)
    print("=== IN-DOMAIN (want GROUNDED) ===")
    for c in cases["retrieval"]:
        report(index, c["query"])
    print("\n=== OUT-OF-DOMAIN (want REFUSED) ===")
    for q in cases["out_of_domain"]:
        report(index, q)


if __name__ == "__main__":
    main()
