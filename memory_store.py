import os
from datetime import datetime, timedelta
from dateparser.search import search_dates
from dotenv import load_dotenv
from mem0 import Memory

load_dotenv()

USER_ID = "veroush"

_memory = None
_pending_wipe = None  # {"ids": [...], "description": "..."} or None


def _get_memory():
    global _memory
    if _memory is None:
        config = {
            "llm": {
                "provider": "groq",
                "config": {"model": "openai/gpt-oss-20b", "temperature": 0},
            },
            "embedder": {
                "provider": "huggingface",
                "config": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
            },
            "vector_store": {
                "provider": "chroma",
                "config": {"collection_name": "sidenote_memory", "path": "memdb"},
            },
        }
        _memory = Memory.from_config(config)
    return _memory


def remember(text: str) -> None:
    """Store a memory verbatim -- no LLM extraction, so no Groq rate-limit
    risk and no paraphrasing of what was actually said (important for
    venting/notes where exact wording matters)."""
    _get_memory().add(text, user_id=USER_ID, infer=False)


def recall(query: str, limit: int = 5) -> list[str]:
    """Return up to `limit` memories most relevant to the query, as plain
    strings ready to drop into a prompt."""
    results = _get_memory().search(query, filters={"user_id": USER_ID}, limit=limit)
    return [r["memory"] for r in results["results"]]


def has_pending_wipe() -> bool:
    return _pending_wipe is not None


def stage_wipe(phrase: str) -> str | None:
    """Parse a date phrase (English or Dutch) out of the user's request,
    find every memory created on that single day, and stage it for
    confirmation -- nothing is deleted yet. Returns a human-readable
    description of what *would* be deleted, or None if no date could be
    parsed from the phrase."""
    global _pending_wipe

    matches = search_dates(phrase, languages=["en", "nl"])
    if not matches:
        return None

    _, matched_date = matches[0]
    day_start = matched_date.replace(hour=0, minute=0, second=0, microsecond=0)
    day_end = day_start + timedelta(days=1)

    all_memories = _get_memory().get_all(filters={"user_id": USER_ID})
    matched = [
        r for r in all_memories["results"]
        if day_start <= datetime.fromisoformat(r["created_at"]).replace(tzinfo=None) < day_end
    ]

    if not matched:
        _pending_wipe = {"ids": [], "description": f"nothing found from {day_start.date()}"}
        return f"I didn't find any memories from {day_start.date()}."

    preview = "; ".join(m["memory"] for m in matched[:5])
    if len(matched) > 5:
        preview += f", and {len(matched) - 5} more"

    _pending_wipe = {"ids": [m["id"] for m in matched], "description": preview}
    return (
        f"I found {len(matched)} memories from {day_start.date()}: {preview}. "
        f"Say 'confirm' to delete them, or 'cancel' to keep them."
    )


def confirm_wipe() -> str:
    global _pending_wipe
    if _pending_wipe is None:
        return "There's nothing pending to delete."
    ids = _pending_wipe["ids"]
    for mem_id in ids:
        _get_memory().delete(memory_id=mem_id)
    count = len(ids)
    _pending_wipe = None
    return f"Deleted {count} memories."


def cancel_wipe() -> str:
    global _pending_wipe
    _pending_wipe = None
    return "Okay, kept everything."