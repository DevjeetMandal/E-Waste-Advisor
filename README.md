# E-Waste Recycling Guide

An AI advisor that takes a plain-language description of an old or broken
electronic device and explains what materials it contains, the risks of
disposing of it incorrectly, and the correct disposal channel — grounded
in India's e-waste management guidance via retrieval-augmented generation.

## How it works

1. **Classify** — `classify.py` maps your device description to a broad
   e-waste category using a transparent keyword rule set.
2. **Retrieve** — `rag_utils.py` embeds your query with a local
   sentence-transformers model and finds the most relevant chunks from
   the knowledge base built by `ingest.py`.
3. **Generate** — `app.py` sends your device details + the retrieved
   chunks to a free Groq-hosted Llama model, which explains the guidance
   and cites which source it drew from.
4. **Log** — every query and response is saved to a local SQLite
   database (`data/history.db`) for a simple history view.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# edit .env and add a free Groq API key from https://console.groq.com/keys
```

### Add the real knowledge base (important before final submission)

This repo ships with `sample_data/sample_rules.txt` — a short,
**paraphrased placeholder** so the pipeline can be tested immediately.
It is NOT the official legal text.

Before you submit:
1. Download the official India E-Waste (Management) Rules, 2022 PDF and
   CPCB's e-waste category list.
2. Place them in `data/raw/`.
3. Re-run `python ingest.py` — it will automatically prefer the real
   PDFs over the placeholder file.

### Build the index and run

```bash
python ingest.py    # builds data/chunks.json and data/embeddings.npy
python app.py        # starts the app at http://127.0.0.1:5000
```

Visit `/history` to see a log of past queries — useful for screenshots.

## Notes for the project write-up

- **AI elements used**: Retrieval-Augmented Generation (RAG), prompt
  engineering, rule-based classification, conversational interface.
- **Responsible AI**:
  - *Transparency*: every answer cites the source document it drew from,
    and the classifier logic is fully inspectable (no black-box model).
  - *Fairness*: guidance is grounded in retrieved reference text rather
    than the model's unconstrained assumptions.
  - *Ethics*: the system prompt instructs the model to say so honestly
    if the retrieved context doesn't clearly cover a device, rather than
    guessing — and steers users toward authorized recyclers over
    informal/unsafe channels.
  - *Privacy*: no personal data is collected beyond the device
    description itself.
