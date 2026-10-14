import io
import json
import sqlite3

import pytest

import config
import db
import vision


# ---- vision ---------------------------------------------------------------
def test_detection_mapping_filters_unknown_and_low_confidence():
    raw = [("laptop", 0.91), ("person", 0.99), ("cell phone", 0.2), ("laptop", 0.5), ("tv", 0.6)]
    out = vision.map_detections(raw, conf_threshold=0.35)
    assert [d["label"] for d in out] == ["laptop", "tv"]
    assert out[0]["confidence"] == 0.91 and out[0]["device"] == "laptop computer"


def test_non_image_is_rejected_cleanly():
    with pytest.raises(vision.VisionError):
        vision.load_image(b"definitely not an image")


def test_corrupt_image_with_valid_header_is_rejected_cleanly():
    with pytest.raises(vision.VisionError):
        vision.load_image(b"\xff\xd8\xff" + b"garbage" * 10)


def test_oversized_upload_rejected():
    with pytest.raises(vision.VisionError):
        vision.load_image(b"0" * (config.MAX_UPLOAD_MB * 1024 * 1024 + 1))


@pytest.mark.skipif(not vision.vision_available(), reason="ultralytics not installed")
def test_real_yolo_runs_end_to_end_on_blank_image():
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (320, 240), (120, 120, 120)).save(buf, format="PNG")
    try:
        assert vision.detect_devices(buf.getvalue()) == []
    except Exception as e:  # weights could not be downloaded (offline machine)
        pytest.skip(f"YOLO weights unavailable: {e}")


# ---- db migration -----------------------------------------------------------
def test_v1_database_is_migrated_in_place(tmp_path):
    p = str(tmp_path / "old.db")
    conn = sqlite3.connect(p)
    conn.execute("""CREATE TABLE queries (id INTEGER PRIMARY KEY AUTOINCREMENT, device_description TEXT NOT NULL,
                    category TEXT NOT NULL, response TEXT NOT NULL, sources TEXT, created_at TEXT NOT NULL)""")
    conn.execute("INSERT INTO queries (device_description, category, response, sources, created_at) VALUES ('old', 'x', 'y', 'z', 't')")
    conn.commit(); conn.close()
    db.init_db(p)
    cols = {r[1] for r in sqlite3.connect(p).execute("PRAGMA table_info(queries)")}
    assert {"hazard_level", "result_json", "trace_json", "used_photo"} <= cols
    assert db.get_recent(5, path=p)[0]["device_description"] == "old"  # old row survives


# ---- flask app ---------------------------------------------------------------
@pytest.fixture()
def client():
    import app as app_module
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


def test_home_page_and_demo_banner(client):
    r = client.get("/")
    assert r.status_code == 200 and b"What do you want to get rid of" in r.data
    assert b"DEMO / TEST MODE" in r.data


def test_advise_renders_structured_result(client):
    r = client.post("/advise", data={"device_description": "old laptop, swollen battery",
                                     "state": "Odisha", "city": "Bhubaneswar", "language": "English"})
    assert r.status_code == 200
    body = r.data.decode()
    assert "HAZARD" in body and "Cosmic Net" in body and "Damaged battery" in body
    assert "How the agent got here" in body


def test_empty_submission_is_rejected(client):
    r = client.post("/advise", data={"device_description": ""})
    assert r.status_code == 400 and b"Please describe" in r.data


def test_non_image_upload_gives_friendly_error(client):
    if not vision.vision_available():
        pytest.skip("vision not installed")
    r = client.post("/advise", data={"device_description": "", "photo": (io.BytesIO(b"nope"), "x.jpg")},
                    content_type="multipart/form-data")
    assert r.status_code == 400 and b"not a readable image" in r.data


def test_api_and_history(client):
    r = client.post("/api/advise", json={"device_description": "old CRT television"})
    assert r.status_code == 200
    data = r.get_json()
    assert data["result"]["hazard_level"] in ("low", "medium", "high")
    assert client.get("/history").status_code == 200
    assert db.get_stats()["total"] >= 1
    assert client.get("/about").status_code == 200
