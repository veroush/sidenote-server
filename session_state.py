"""
Session-boundary detection for Sidenote's conversational-loop pivot.

Wired into /answer in server.py. A session opens when the wake word
("severus") is heard, stays open across follow-up questions, and closes
on the close phrase ("close severus") or after SILENCE_TIMEOUT_SECONDS of
inactivity. Confirmed working end-to-end through the dictate Android app
over real speech (wake word, active Q&A, close phrase, and return to
idle all verified).
"""

import re
import time
from difflib import SequenceMatcher

WAKE_WORD = "severus"
CLOSE_PHRASE = "close severus"
SILENCE_TIMEOUT_SECONDS = 60

# How close a word has to be to count as a match. 1.0 = exact match only.
# 0.75 is fairly forgiving -- catches things like "severes" or "severus's"
# but shouldn't fire on unrelated words. Tune this once we see real
# mis-transcriptions from actual phone audio.
FUZZY_THRESHOLD = 0.75


def _tokenize(text: str) -> list[str]:
    """Split into lowercase words, stripping punctuation, so 'Severus,' and
    'severus' match the same way."""
    return re.findall(r"[a-zA-Z]+", text.lower())


def _best_match_ratio(word: str, target: str) -> float:
    return SequenceMatcher(None, word, target).ratio()


def contains_fuzzy(text: str, target: str, threshold: float = FUZZY_THRESHOLD) -> bool:
    """True if `text` contains a run of consecutive words closely matching
    `target` (which may itself be one or more words)."""
    target_words = _tokenize(target)
    text_words = _tokenize(text)
    n = len(target_words)
    target_str = " ".join(target_words)

    for i in range(len(text_words) - n + 1):
        window = " ".join(text_words[i:i + n])
        if _best_match_ratio(window, target_str) >= threshold:
            return True
    return False


class SessionState:
    """Tracks whether a conversation session is currently active.

    Server-side, in-memory, single-user -- matches the earlier decision
    that session state must survive the phone's browser tab being
    backgrounded or the screen locking.
    """

    def __init__(self):
        self.active = False
        self.last_activity: float | None = None

    def _timed_out(self, now: float) -> bool:
        if not self.active or self.last_activity is None:
            return False
        return (now - self.last_activity) > SILENCE_TIMEOUT_SECONDS

    def check_expired(self, now: float | None = None) -> bool:
        """Peek at whether the session has timed out, WITHOUT counting this
        call itself as activity. Used by a periodic heartbeat ping so idle
        checks don't reset the timer they're trying to detect."""
        if now is None:
            now = time.time()
        if self._timed_out(now):
            self.active = False
            self.last_activity = None
            return True
        return False

    def process(self, text: str, now: float | None = None) -> str:
        """Feed in a chunk of transcribed text. Returns one of:
        "started"    -- wake word detected, session just opened
        "ended"      -- close phrase detected
        "timed_out"  -- silence timeout hit (session closed itself)
        "active"     -- session continues, this chunk is part of it
        "idle"       -- no session, nothing relevant said
        """
        if now is None:
            now = time.time()

        # A timeout can fire even on an empty/irrelevant chunk, since time
        # has passed regardless of what (if anything) was said.
        if self._timed_out(now):
            self.active = False
            self.last_activity = None
            return "timed_out"

        if not self.active:
            if contains_fuzzy(text, WAKE_WORD):
                self.active = True
                self.last_activity = now
                return "started"
            return "idle"

        # Session is active from here on.
        if contains_fuzzy(text, CLOSE_PHRASE):
            self.active = False
            self.last_activity = None
            return "ended"

        self.last_activity = now
        return "active"
