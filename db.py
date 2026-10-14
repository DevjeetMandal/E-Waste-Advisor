"""
SQLite history. Migrates the v1 table in place (adds columns), so your existing
data/history.db keeps working.
"""

import json
import sqlite3
from datetime import datetime, timezone

import config

NEW_COLUMNS = {
    "hazard_level": "TEXT",
    "language": "TEXT",
    "used_photo": "INTEGER DEFAULT 0",
    "covered_by_rules": "INTEGER",
    "result_json": "TEXT",
    "trace_json": "TEXT",
}


def _conn(path=None):
    conn = sqlite3.connect(path or config.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(path=None):
    conn = _conn(path)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            device_description TEXT NOT NULL,
            category TEXT NOT NULL,
            response TEXT NOT NULL,
            sources TEXT,
            created_at TEXT NOT NULL)"""
    )
    existing = {r["name"] for r in conn.execute("PRAGMA table_info(queries)")}
    for col, decl in NEW_COLUMNS.items():
        if col not in existing:
            conn.execute(f"ALTER TABLE queries ADD COLUMN {col} {decl}")
    conn.commit()
    conn.close()


def log_query(agent_result, used_photo=False, path=None):
    r = agent_result.result
    sources = ", ".join(sorted({f"{c['source']} p.{c['page']}" if c.get("page") else c["source"]
                                for c in r["citations"]}))
    conn = _conn(path)
    conn.execute(
        """INSERT INTO queries (device_description, category, response, sources, created_at,
               hazard_level, language, used_photo, covered_by_rules, result_json, trace_json)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (
            agent_result.description, agent_result.classification["category"], r["summary"], sources,
            datetime.now(timezone.utc).isoformat(), r["hazard_level"], agent_result.language,
            int(used_photo), int(r["covered_by_rules"]), json.dumps(r), json.dumps(agent_result.trace),
        ),
    )
    conn.commit()
    conn.close()


def get_recent(limit=20, path=None):
    conn = _conn(path)
    rows = [dict(r) for r in conn.execute("SELECT * FROM queries ORDER BY id DESC LIMIT ?", (limit,))]
    conn.close()
    return rows


def get_stats(path=None):
    """Real usage numbers from the log - nothing estimated."""
    conn = _conn(path)
    total = conn.execute("SELECT COUNT(*) c FROM queries").fetchone()["c"]
    by_cat = [dict(r) for r in conn.execute(
        "SELECT category, COUNT(*) n FROM queries GROUP BY category ORDER BY n DESC")]
    by_haz = [dict(r) for r in conn.execute(
        "SELECT hazard_level, COUNT(*) n FROM queries WHERE hazard_level IS NOT NULL GROUP BY hazard_level")]
    covered = conn.execute(
        "SELECT SUM(covered_by_rules) s, COUNT(covered_by_rules) c FROM queries WHERE covered_by_rules IS NOT NULL"
    ).fetchone()
    conn.close()
    return {"total": total, "by_category": by_cat, "by_hazard": by_haz,
            "rule_backed": covered["s"] or 0, "rated": covered["c"] or 0}
