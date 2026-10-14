"""
Transparent keyword classifier mapping a device description to a broad group.

v1 used substring matching, so "ac" matched "backpack" and "led" matched
"controlled". v2 matches whole words/phrases only, and when two groups tie
(e.g. "battery from my phone") the one mentioned first wins.

These are broad illustrative groups - the CPCB schedule itself is item-level,
which is why the retrieval step, not this classifier, grounds the final answer.
"""

import re

CATEGORY_KEYWORDS = {
    "IT and Telecommunication Equipment": [
        "laptop", "computer", "pc", "desktop", "mobile", "phone", "smartphone", "tablet",
        "printer", "scanner", "router", "modem", "monitor", "keyboard", "mouse", "hard disk",
        "hard drive", "ssd", "motherboard", "charger", "adapter", "pen drive", "ups",
        "set top box", "telephone",
    ],
    "Consumer Electronics": [
        "tv", "television", "speaker", "speakers", "music system", "headphone", "headphones",
        "earphone", "earphones", "earbuds", "camera", "radio", "dvd player", "home theatre",
        "smartwatch", "remote",
    ],
    "Large Household Appliances": [
        "refrigerator", "fridge", "washing machine", "air conditioner", "ac", "freezer",
        "water cooler", "dishwasher",
    ],
    "Small Household Appliances": [
        "microwave", "oven", "mixer", "grinder", "iron", "kettle", "toaster", "fan", "heater",
        "geyser", "vacuum cleaner", "hair dryer", "induction", "cooktop", "water purifier",
    ],
    "Batteries and Power Storage": [
        "battery", "batteries", "power bank", "li-ion", "lithium", "inverter battery",
    ],
    "Lighting Equipment": [
        "bulb", "tube light", "tubelight", "led", "cfl", "fluorescent", "lamp", "street light",
    ],
    "Electrical and Electronic Tools": [
        "drill", "power tool", "soldering iron", "multimeter", "tester", "electric saw",
    ],
}

DEFAULT_CATEGORY = "Other Electronic/Electrical Equipment"

_PATTERNS = {
    cat: [(kw, re.compile(r"(?<![a-z0-9])" + re.escape(kw) + r"(?![a-z0-9])")) for kw in kws]
    for cat, kws in CATEGORY_KEYWORDS.items()
}


def classify_with_evidence(description):
    """Return {"category": str, "matched": [keywords]} for transparency."""
    text = (description or "").lower()
    best = None  # (hits, -first_position, category, matched)
    for category, pats in _PATTERNS.items():
        matched, first = [], None
        for kw, pat in pats:
            m = pat.search(text)
            if m:
                matched.append(kw)
                first = m.start() if first is None else min(first, m.start())
        if matched:
            key = (len(matched), -first)
            if best is None or key > best[0]:
                best = (key, category, matched)
    if best is None:
        return {"category": DEFAULT_CATEGORY, "matched": []}
    return {"category": best[1], "matched": best[2]}


def classify_device(description):
    return classify_with_evidence(description)["category"]
