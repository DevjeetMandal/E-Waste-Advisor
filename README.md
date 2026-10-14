# E-Waste Advisor (v2)

An agentic RAG assistant that tells people in India how to dispose of old electronics:
hazard level, what is worth recovering, where to take it, and **which official document
and page says so**.

## What's new in v2
| Area | v1 | v2 |
|---|---|---|
| Pipeline | fixed: classify -> retrieve -> one LLM call | **tool-calling agent** (classify, search rules, find recyclers) with safeguards that force the essential steps |
| Output | free text | **validated JSON** rendered as cards: hazard badge, materials, risks, do/don't, disposal options |
| Citations | filename only | **file + page number**, checked in code against what was actually retrieved |
| Retrieval | dense only | **hybrid**: dense + BM25 keyword search, rank-fused (finds items named in schedule tables) |
| Weak retrieval | still answered | **two-tier grounding gate**: similarity bar, stricter when no device is named; or the device is literally named in a retrieved passage |
| Input | text | text **or photo** (pretrained YOLO) |
| Local help | none | **CPCB-listed recyclers** by state/city (Odisha, Delhi, West Bengal, Jharkhand, Chhattisgarh, Assam so far) |
| Safety | LLM only | deterministic safety box (swollen battery, broken CFL, data wiping) - never model-generated |
| Languages | English | English / Hindi / Odia (machine-translated) |
| Classifier | substring matching (bug: "ac" matched "backpack") | whole-word matching |
| Reliability | crashed when Groq retired a model | automatic **model fallback list** + friendly errors |
| Testing | none | 57 tests + `eval.py` + `diagnose.py` |

## Source documents (important)
Put these in `data/raw/`:
- **India's E-Waste (Management) Rules, 2022** (Gazette PDF). The item schedules (Schedule I etc.) are inside it.
- Do **not** use CPCB's "Red/Orange/Green/White category" list of *industries*. It is not an e-waste item list;
  it fills the index with numeric tables about soap and rubber factories. (Recycler data comes from
  `data/recyclers.csv`, not from a PDF.)

`ingest.py` cleans what it reads: it strips the garbled Devanagari from bilingual Gazette pages, removes
running page headers, and drops number-heavy table chunks. It prints chunks per source so you can see
what actually went into the index.

## Setup
```bash
python -m venv venv && venv\Scripts\activate        # Windows
pip install -r requirements.txt
pip install -r requirements-vision.txt              # optional: photo input
copy .env.example .env                              # add your Groq key
python ingest.py                                    # builds the index from data/raw/*.pdf
python app.py                                       # http://127.0.0.1:5000
```
`ingest.py` now stores **page numbers**. Re-run it after pulling v2 (the old index has none, and the
app will tell you if the index doesn't match).

## Calibrate before you trust it
```bash
python eval.py          # classifier accuracy, retrieval hit rate, threshold calibration
python eval.py --e2e 8  # also runs the full agent on 8 queries (uses Groq calls)
```
`eval.py` measures the *real* gate: in-domain queries wrongly refused, off-topic queries wrongly accepted.
Edit `data/eval_cases.json` so the retrieval keywords match wording that really appears in your PDFs, then tune
`EWASTE_MIN_SCORE` / `EWASTE_MIN_SCORE_NO_DEVICE` in `.env`. The defaults (0.25 / 0.40) are guesses.
`python diagnose.py` shows, per query, the gate decision and the top chunks; `python diagnose.py --find word`
tells you whether a word is in the index at all.
`eval.py` writes `eval_report.md` - good material for the Test & Refine section.

## Tests
```bash
pip install -r requirements-dev.txt
python -m pytest -q
```
Tests run fully offline (fake LLM, hash embeddings). They cover citation validation, forced tool steps,
the grounding gate, malformed model output, the Groq 404 model-retirement case, DB migration and the Flask routes.

## Extending the recycler list
`data/recyclers.csv` (columns: name, address, state, district, city). Source: CPCB list of authorised
dismantlers/recyclers as on 08-06-2023. Add rows for more states; no code change needed.
Authorisations change - always verify with the State Pollution Control Board.

## Known limitations (state these in your write-up)
- Hazard level, materials and risks are AI-assessed from general knowledge, not from the rules.
- Photo detection covers a few COCO classes only (laptop, phone, TV/monitor, keyboard, mouse, remote,
  microwave, oven, toaster, refrigerator, hair dryer). It says nothing about condition.
- Recycler data is dated and partial.
- Hindi/Odia output is machine-translated and unreviewed.
- Free-tier Groq limits apply.
