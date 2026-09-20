
import sqlite3
from datetime import datetime

DB_PATH = "data/history.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            device_description TEXT NOT NULL,
            category TEXT NOT NULL,
            response TEXT NOT NULL,
            sources TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def log_query(device_description, category, response, sources):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO queries (device_description, category, response, sources, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (device_description, category, response, sources, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def get_recent(limit=10):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM queries ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]
