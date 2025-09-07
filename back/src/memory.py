# src/memory.py
import sqlite3, os
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
DB_DIR = BASE_DIR / "state_db"
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_DIR / "memory.db"

def _conn():
    return sqlite3.connect(str(DB_PATH))

def init():
    with _conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS memory (
            user_id TEXT, key TEXT, value TEXT,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (user_id, key)
        )""")

def set_kv(user_id: str, key: str, value: str):
    with _conn() as c:
        c.execute("""INSERT INTO memory(user_id, key, value) VALUES(?,?,?)
                     ON CONFLICT(user_id,key) DO UPDATE SET value=excluded.value, updated_at=CURRENT_TIMESTAMP""",
                  (user_id, key, value))

def get(user_id: str, key: str, default: str = "") -> str:
    with _conn() as c:
        row = c.execute("SELECT value FROM memory WHERE user_id=? AND key=?", (user_id, key)).fetchone()
        return row[0] if row else default
