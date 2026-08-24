import os
from groq import Groq

_client = None


def _get_client():
    """Lazily create the Groq client so a missing key doesn't crash the
    server at import time — only when an answer is actually requested."""
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set in the environment")
        _client = Groq(api_key=api_key)
    return _client


def get_concise_answer(question: str, notes: str) -> str:
    """Given a matched topic's notes and the question that triggered the
    match, ask Groq for a short, direct answer. Raises on any failure —
    the caller (server.py) decides what to do (e.g. fall back to the
    local heuristic)."""
    prompt = f"""Use the following note to answer the question. Be concise —
answer only what was asked, in one or two sentences. If the note doesn't
contain the answer, say so briefly.

Note:
{notes}

Question: {question}

Answer:"""

    client = _get_client()
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()