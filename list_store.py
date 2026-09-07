import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "lists.db")

LIST_ALIASES = {
    "groceries": "groceries",
    "grocery": "groceries",
    "grocery list": "groceries",
    "groceries list": "groceries",
}


def _normalize_list_name(list_name: str) -> str:
    key = list_name.lower().strip()
    return LIST_ALIASES.get(key, key)


def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS list_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            list_name TEXT NOT NULL,
            item TEXT NOT NULL
        )
    """)
    return conn


def add_item(list_name: str, item: str) -> None:
    conn = _get_conn()
    try:
        conn.execute(
            "INSERT INTO list_items (list_name, item) VALUES (?, ?)",
            (_normalize_list_name(list_name), item.strip()),
        )
        conn.commit()
    finally:
        conn.close()


def add_items(list_name: str, items: list[str]) -> None:
    conn = _get_conn()
    try:
        for item in items:
            conn.execute(
                "INSERT INTO list_items (list_name, item) VALUES (?, ?)",
                (_normalize_list_name(list_name), item.strip()),
            )
        conn.commit()
    finally:
        conn.close()


def get_items(list_name: str) -> list[str]:
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT item FROM list_items WHERE list_name = ? ORDER BY id",
            (_normalize_list_name(list_name),),
        ).fetchall()
        return [r[0] for r in rows]
    finally:
        conn.close()

def clear_list(list_name: str) -> None:
    conn = _get_conn()
    try:
        conn.execute(
            "DELETE FROM list_items WHERE list_name = ?",
            (_normalize_list_name(list_name),),
        )
        conn.commit()
    finally:
        conn.close()