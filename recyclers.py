"""
Authorised recycler lookup from data/recyclers.csv.

The seed data comes from CPCB's "List of Dismantlers/Recyclers as per the
authorisation issued by SPCBs/PCCs" (as on 08-06-2023) and currently covers
only some states. Authorisations change: always verify with the State
Pollution Control Board before handing anything over. Add rows to the CSV
to extend coverage - the app picks them up with no code change.
"""

import csv
import os

import config

CPCB_LIST_URL = "https://www.ndmc.gov.in/pdf/cpcb_approved_list_of_e-waste_recyclers_dismantler.pdf"
DATA_AS_ON = "08-06-2023"

STATE_ALIASES = {
    "orissa": "Odisha", "odisha": "Odisha", "od": "Odisha",
    "delhi": "Delhi", "new delhi": "Delhi", "nct of delhi": "Delhi",
    "west bengal": "West Bengal", "bengal": "West Bengal", "wb": "West Bengal",
    "jharkhand": "Jharkhand", "chhattisgarh": "Chhattisgarh", "chattisgarh": "Chhattisgarh",
    "assam": "Assam",
}

# City -> state, only used when the user gives a city but no state.
CITY_STATE = {
    "bhubaneswar": "Odisha", "cuttack": "Odisha", "puri": "Odisha", "rourkela": "Odisha",
    "berhampur": "Odisha", "sambalpur": "Odisha", "khordha": "Odisha", "khurda": "Odisha",
    "kolkata": "West Bengal", "howrah": "West Bengal", "durgapur": "West Bengal",
    "ranchi": "Jharkhand", "jamshedpur": "Jharkhand", "dhanbad": "Jharkhand",
    "raipur": "Chhattisgarh", "bhilai": "Chhattisgarh", "durg": "Chhattisgarh",
    "guwahati": "Assam",
}


def normalize_state(state):
    if not state:
        return None
    key = state.strip().lower()
    return STATE_ALIASES.get(key, state.strip().title())


def load_recyclers(path=None):
    path = path or config.RECYCLERS_CSV
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def covered_states(rows=None):
    rows = rows if rows is not None else load_recyclers()
    return sorted({r["state"] for r in rows})


def find_recyclers(state=None, city=None, limit=6, rows=None):
    rows = rows if rows is not None else load_recyclers()
    city_l = (city or "").strip().lower()
    state_n = normalize_state(state)
    if not state_n and city_l:
        state_n = CITY_STATE.get(city_l)

    base_note = (
        f"Source: CPCB list of authorised dismantlers/recyclers, as on {DATA_AS_ON}. "
        "Authorisations change - confirm with your State Pollution Control Board before handing over any device."
    )

    if not state_n:
        return {"matches": [], "state": None, "note":
                "No state/city given, so no local recyclers were looked up. " + base_note,
                "covered_states": covered_states(rows)}

    in_state = [r for r in rows if r["state"] == state_n]
    if not in_state:
        return {"matches": [], "state": state_n, "covered_states": covered_states(rows), "note":
                f"This app's recycler data does not yet cover {state_n} (it covers: "
                f"{', '.join(covered_states(rows))}). Check the full CPCB list or your SPCB website. " + base_note}

    note = base_note
    matches = in_state
    if city_l:
        local = [r for r in in_state if city_l in (r.get("city") or "").lower()
                 or city_l in (r.get("district") or "").lower()
                 or city_l in (r.get("address") or "").lower()]
        if local:
            matches = local
        else:
            note = (f"No listed recycler mentions '{city}'; showing authorised recyclers elsewhere in "
                    f"{state_n}. " + base_note)
    return {"matches": matches[:limit], "state": state_n, "note": note,
            "covered_states": covered_states(rows)}
