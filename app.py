import os

from dotenv import load_dotenv
from flask import Flask, render_template, request
from groq import Groq

from classify import classify_device
from rag_utils import retrieve, index_exists
from db import init_db, log_query, get_recent

load_dotenv()

app = Flask(__name__)
init_db()

GROQ_MODEL = "openai/gpt-oss-120b"  
groq_client = Groq()  


SYSTEM_PROMPT = """You are an e-waste disposal advisor for users in India. You are given:
- a description of an old or broken electronic device
- its likely e-waste category
- reference excerpts retrieved from India's e-waste management guidance documents

Using ONLY the reference excerpts provided (do not invent facts beyond them),
explain in plain, friendly language:
1. What materials in this device are worth recovering
2. What health or environmental risks come from disposing of it incorrectly
   (e.g. through informal scrap dealers or regular household trash)
3. The correct way to dispose of it (authorized collection point, producer
   take-back, or registered recycler/dismantler)

At the end, add a line: "Source: <the source filename(s) you drew on>".
If the reference excerpts don't clearly cover this device, say so honestly
rather than guessing, and give general safe-disposal advice instead.
Keep the whole answer under 200 words.
"""


def build_user_prompt(device_description, category, retrieved_chunks):
    context = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in retrieved_chunks
    )
    return (
        f"Device description: {device_description}\n"
        f"Likely category: {category}\n\n"
        f"Reference excerpts:\n{context}"
    )


@app.route("/", methods=["GET"])
def index():
    if not index_exists():
        return (
            "No retrieval index found. Run `python ingest.py` first, "
            "then restart the app.",
            500,
        )
    return render_template("index.html")


@app.route("/advise", methods=["POST"])
def advise():
    device_description = request.form.get("device_description", "").strip()
    if not device_description:
        return render_template("index.html", error="Please describe the device.")

    category = classify_device(device_description)
    retrieved_chunks = retrieve(device_description, k=4)

    user_prompt = build_user_prompt(device_description, category, retrieved_chunks)

    completion = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=400,
    )
    answer = completion.choices[0].message.content

    sources_used = ", ".join(sorted({c["source"] for c in retrieved_chunks}))
    log_query(device_description, category, answer, sources_used)

    return render_template(
        "result.html",
        device_description=device_description,
        category=category,
        answer=answer,
        sources_used=sources_used,
    )


@app.route("/history", methods=["GET"])
def history():
    return render_template("history.html", rows=get_recent(20))


if __name__ == "__main__":
    app.run(debug=True)
