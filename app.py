"""
E-Waste Advisor v2 - Flask app.

Run:
    pip install -r requirements.txt          (+ requirements-vision.txt for photo input)
    cp .env.example .env                      (add GROQ_API_KEY)
    python ingest.py
    python app.py
"""

import io
import json
import logging

from flask import Flask, jsonify, render_template, request

import config
import db
import rag_utils
import vision
from agent import Agent
from llm import LLMError, get_llm
from recyclers import covered_states

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = (config.MAX_UPLOAD_MB + 1) * 1024 * 1024
log = logging.getLogger("ewaste")

INDIAN_STATES = [
    "", "Andhra Pradesh", "Assam", "Bihar", "Chhattisgarh", "Delhi", "Goa", "Gujarat", "Haryana",
    "Himachal Pradesh", "Jammu & Kashmir", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh",
    "Maharashtra", "Odisha", "Punjab", "Rajasthan", "Tamil Nadu", "Telangana", "Uttar Pradesh",
    "Uttarakhand", "West Bengal",
]

db.init_db()


def demo_mode():
    return config.LLM_BACKEND == "fake" or config.EMBED_BACKEND == "hash"


def _form_ctx(**extra):
    ctx = dict(
        states=INDIAN_STATES, languages=config.LANGUAGES, vision_ok=vision.vision_available(),
        demo=demo_mode(), covered=covered_states(), max_mb=config.MAX_UPLOAD_MB,
    )
    ctx.update(extra)
    return ctx


def _run(description, photo_bytes, state, city, language):
    """Shared by the HTML route and the JSON API. Returns (agent_result, detection, error)."""
    detection = None
    if photo_bytes:
        if not vision.vision_available():
            return None, None, "Photo input is not installed (pip install -r requirements-vision.txt)."
        try:
            found = vision.detect_devices(photo_bytes)
        except vision.VisionError as e:
            return None, None, str(e)
        detection = found[0] if found else None
        if detection and not description:
            description = detection["device"]
        elif detection and description:
            description = f"{description} (photo suggests: {detection['device']})"
        elif not description:
            return None, None, ("I couldn't recognise a known device in that photo (batteries, bulbs and "
                                "chargers aren't detectable). Please describe the item in text.")
    if not description:
        return None, None, "Please describe the device or upload a photo."
    if language not in config.LANGUAGES:
        language = "English"
    try:
        index = rag_utils.get_index()
        agent = Agent(get_llm(), index)
        res = agent.run(description, state_name=state, city=city, language=language)
    except rag_utils.IndexError_ as e:
        return None, detection, str(e)
    except LLMError as e:
        return None, detection, str(e)
    db.log_query(res, used_photo=bool(photo_bytes))
    return res, detection, None


@app.route("/", methods=["GET"])
def index():
    if not rag_utils.index_exists():
        return render_template("index.html", **_form_ctx(error="No retrieval index found. Run `python ingest.py` first."))
    return render_template("index.html", **_form_ctx())


@app.route("/advise", methods=["POST"])
def advise():
    description = request.form.get("device_description", "").strip()
    photo = request.files.get("photo")
    photo_bytes = photo.read() if photo and photo.filename else None
    res, detection, err = _run(
        description, photo_bytes, request.form.get("state", ""), request.form.get("city", "").strip(),
        request.form.get("language", "English"),
    )
    if err:
        return render_template("index.html", **_form_ctx(error=err, prefill=description)), 400
    return render_template("result.html", r=res, detection=detection, demo=demo_mode())


@app.route("/api/advise", methods=["POST"])
def api_advise():
    body = request.get_json(silent=True) or {}
    res, detection, err = _run(
        (body.get("device_description") or "").strip(), None, body.get("state", ""),
        (body.get("city") or "").strip(), body.get("language", "English"),
    )
    if err:
        return jsonify({"error": err}), 400
    return jsonify({
        "category": res.classification, "result": res.result, "evidence": res.evidence,
        "recyclers": res.recyclers, "trace": res.trace, "safety": res.safety,
        "grounded": res.grounded, "best_score": res.best_score, "model": res.model,
    })


@app.route("/history", methods=["GET"])
def history():
    rows = db.get_recent(20)
    for r in rows:
        r["result"] = json.loads(r["result_json"]) if r.get("result_json") else None
    return render_template("history.html", rows=rows, stats=db.get_stats(), demo=demo_mode())


@app.route("/about", methods=["GET"])
def about():
    return render_template("about.html", demo=demo_mode(), min_score=config.MIN_SCORE,
                           covered=covered_states())


if __name__ == "__main__":
    app.run(debug=True)
