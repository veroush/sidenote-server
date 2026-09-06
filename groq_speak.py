import os
from groq import Groq

_client = None


def _get_client():
    """Lazily create the Groq client so a missing key doesn't crash the
    server at import time — only when speech is actually requested."""
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set in the environment")
        _client = Groq(api_key=api_key)
    return _client


def synthesize_speech(text: str) -> bytes:
    """Given answer text, return raw WAV audio bytes from Groq's TTS.
    Raises on any failure — the caller (server.py) decides what to do."""
    client = _get_client()
    response = client.audio.speech.create(
        model="canopylabs/orpheus-v1-english",
        voice="troy",
        input=text,
        response_format="wav",
        speed=5,
    )
    return response.read()