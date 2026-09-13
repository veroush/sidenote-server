"""
Short-term conversation memory -- keeps the last few (question, answer)
pairs in memory so Groq can resolve pronouns and follow-ups like "can it
have long-term effects" instead of treating every /answer call as the
first message ever sent. Deliberately NOT persisted to disk -- this is
just this run's conversation, not the long-term Mem0 memory (mem.py).
"""

MAX_TURNS = 6  # how many past (question, answer) pairs to keep/send

_history: list[dict] = []  # each item: {"question": ..., "answer": ...}


def get_history() -> list[dict]:
    """Return the stored turns, oldest first."""
    return list(_history)


def add_turn(question: str, answer: str) -> None:
    """Record a completed turn, trimming to the last MAX_TURNS."""
    _history.append({"question": question, "answer": answer})
    del _history[:-MAX_TURNS]


def clear_history() -> None:
    """Wipe conversation memory -- e.g. if you later add a 'forget this
    conversation' voice command, or on server restart if you want a
    manual reset endpoint."""
    _history.clear()