"""
Photo input: detect e-waste devices in an uploaded image with a pretrained
YOLO model (COCO classes, no training needed).

Limits worth stating honestly:
  * COCO only knows a handful of electronics. Batteries, bulbs, chargers, cables,
    power banks etc. are NOT detectable - the UI asks for a text description then.
  * This identifies *what the object is*, nothing about its condition.
  * Images are processed in memory and never written to disk.
"""

import io

import config

# COCO label -> description fed into the rest of the pipeline
COCO_TO_DEVICE = {
    "laptop": "laptop computer",
    "cell phone": "mobile phone",
    "tv": "television or computer monitor",
    "keyboard": "computer keyboard",
    "mouse": "computer mouse",
    "remote": "remote control",
    "microwave": "microwave oven",
    "oven": "oven",
    "toaster": "toaster",
    "refrigerator": "refrigerator",
    "hair drier": "hair dryer",
}

_model = None


class VisionError(Exception):
    pass


def vision_available():
    try:
        import ultralytics  # noqa: F401
        import PIL  # noqa: F401
        return True
    except ImportError:
        return False


def _get_model():
    global _model
    if _model is None:
        from ultralytics import YOLO

        _model = YOLO(config.YOLO_WEIGHTS)  # downloads weights on first use
    return _model


def map_detections(raw, conf_threshold=None):
    """raw: iterable of (coco_label, confidence). Returns known electronics, best first."""
    conf_threshold = config.VISION_CONF if conf_threshold is None else conf_threshold
    best = {}
    for label, conf in raw:
        if label in COCO_TO_DEVICE and conf >= conf_threshold:
            if label not in best or conf > best[label]:
                best[label] = conf
    out = [
        {"label": lab, "device": COCO_TO_DEVICE[lab], "confidence": round(float(c), 3)}
        for lab, c in best.items()
    ]
    return sorted(out, key=lambda d: d["confidence"], reverse=True)


def _looks_like_image(data):
    """Check magic bytes first. Importing ultralytics patches PIL.Image.open so that an
    unreadable file triggers a runtime `pip install pi-heif`; rejecting bad files up
    front avoids that and gives a clear error."""
    return (
        data[:3] == b"\xff\xd8\xff"                       # JPEG
        or data[:8] == b"\x89PNG\r\n\x1a\n"                # PNG
        or (data[:4] == b"RIFF" and data[8:12] == b"WEBP")  # WEBP
        or data[:6] in (b"GIF87a", b"GIF89a")               # GIF
        or data[:2] == b"BM"                                # BMP
    )


def load_image(data):
    from PIL import Image

    if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise VisionError(f"Image is larger than {config.MAX_UPLOAD_MB} MB.")
    if not _looks_like_image(data):
        raise VisionError("That file is not a readable image. Please use JPG, PNG or WEBP.")
    try:
        img = Image.open(io.BytesIO(data))
        if img.width * img.height > config.MAX_IMAGE_PIXELS:
            raise VisionError("Image resolution is too large.")
        return img.convert("RGB")
    except VisionError:
        raise
    except Exception as e:
        raise VisionError("That image could not be decoded.") from e


def detect_devices(data):
    """Returns a list of {label, device, confidence}; empty if no known device is found."""
    img = load_image(data)
    model = _get_model()
    result = model.predict(img, conf=config.VISION_CONF, verbose=False)[0]
    names = result.names
    raw = [(names[int(c)], float(p)) for c, p in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist())]
    return map_detections(raw)
