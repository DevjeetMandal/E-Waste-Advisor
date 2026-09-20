CATEGORY_KEYWORDS = {
    "IT and Telecommunication Equipment": [
        "laptop", "computer", "pc", "desktop", "mobile", "phone", "smartphone",
        "tablet", "printer", "router", "modem", "monitor", "keyboard", "mouse",
        "hard disk", "ssd", "motherboard",
    ],
    "Consumer Electronics": [
        "tv", "television", "refrigerator", "fridge", "washing machine",
        "ac", "air conditioner", "microwave", "oven", "speaker", "music system",
    ],
    "Batteries and Power Storage": [
        "battery", "batteries", "power bank", "li-ion", "lithium", "cell",
    ],
    "Lighting Equipment": [
        "bulb", "tube light", "led", "cfl", "fluorescent", "lamp",
    ],
    "Electrical and Electronic Tools": [
        "drill", "tool", "tester", "soldering iron", "multimeter",
    ],
}

DEFAULT_CATEGORY = "Other Electronic/Electrical Equipment"


def classify_device(description):
    """Return the best-matching category name for a free-text description."""
    text = description.lower()
    best_category = DEFAULT_CATEGORY
    best_hits = 0

    for category, keywords in CATEGORY_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in text)
        if hits > best_hits:
            best_hits = hits
            best_category = category

    return best_category
