from classify import classify_device, classify_with_evidence, DEFAULT_CATEGORY
from safety import safety_flags


def test_no_substring_false_positives():
    # v1 matched "ac" inside "backpack" and "led" inside "controlled"
    assert classify_device("an old backpack") == DEFAULT_CATEGORY
    assert classify_device("a controlled substance") == DEFAULT_CATEGORY
    assert classify_device("a placed order") == DEFAULT_CATEGORY


def test_basic_categories():
    assert classify_device("old refrigerator") == "Large Household Appliances"
    assert classify_device("AC not cooling") == "Large Household Appliances"
    assert classify_device("broken CFL bulb") == "Lighting Equipment"
    assert classify_device("dead power bank") == "Batteries and Power Storage"
    assert classify_device("mixer grinder motor burnt") == "Small Household Appliances"
    assert classify_device("Dell laptop, screen cracked") == "IT and Telecommunication Equipment"


def test_tie_goes_to_first_mentioned():
    assert classify_device("laptop battery") == "IT and Telecommunication Equipment"
    assert classify_device("battery from my phone") == "Batteries and Power Storage"


def test_evidence_lists_matches():
    out = classify_with_evidence("laptop and charger")
    assert set(out["matched"]) == {"laptop", "charger"}


def test_safety_swollen_battery_is_urgent():
    flags = safety_flags("my phone battery is swollen")
    assert any(f["level"] == "urgent" for f in flags)
    assert any("data" in f["title"].lower() for f in flags)  # phone -> data wipe tip


def test_safety_cfl_and_fridge():
    assert any("mercury" in f["title"].lower() for f in safety_flags("broken CFL tube"))
    assert any("dismantle" in f["title"].lower() for f in safety_flags("old fridge"))


def test_safety_nothing_for_unrelated():
    assert safety_flags("wooden chair") == []
    assert safety_flags("a fully charged battery") == []  # no damage word -> no alarm
