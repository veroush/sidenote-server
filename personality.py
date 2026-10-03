"""
Centralized personality definitions for Sidenote's Groq-powered responses.
Adds a witty, JARVIS-style flavor to answers and confirmations without
touching the underlying tool-calling logic -- if this file is ever removed
or add_flavor() calls are stripped out, everything still works exactly as
before, just plainer.
"""

import random

# Short, in-character remarks occasionally appended to responses. Kept
# separate from the actual answer/confirmation text so the flavor can be
# turned off entirely just by not calling add_flavor().
QUIPS = [
    "Consider it done.",
    "Filed away with the rest of your excellent decisions.",
    "I live for this kind of thrilling admin work.",
    "Noted, logged, and mildly judged.",
    "Another task, flawlessly handled, as expected.",
    "I'll add that to the list of things I definitely won't forget.",
]

# How often a quip gets appended (1 in N responses), so it doesn't wear
# out its welcome by showing up on every single reply.
QUIP_CHANCE = 3  # 1 in 3


def add_flavor(text: str) -> str:
    """Occasionally append a short witty aside to an otherwise plain
    response, so replies feel less robotic without changing the actual
    information in them."""
    if not text:
        return text
    if random.randint(1, QUIP_CHANCE) == 1:
        return f"{text} {random.choice(QUIPS)}"
    return text


def personality_preamble() -> str:
    """A short system-prompt fragment establishing the assistant's tone,
    for use anywhere Groq is asked to generate free text. Deliberately
    strict about NOT adding jokes or commentary on its own -- any wit
    after a finished task comes from add_flavor(), which callers only
    use on completed actions."""
    return (
        "You are a concise, accurate personal assistant with a dry, "
        "understated tone. Answer exactly what was asked, then stop. "
        "Never tack unrequested jokes, quips, or remarks about yourself "
        "or your job onto an answer, and never add anything after your "
        "answer."
    )


def language_instruction(lang: str) -> str:
    """A short system-prompt fragment telling Groq which language to
    respond in, based on the detected language of the user's question."""
    if lang == "nl":
        return "Respond in Dutch (Nederlands), matching the user's language."
    return "Respond in English, matching the user's language."


LIST_TEMPLATES = {
    "en": {
        "added": "Added {items} to {list_name}.",
        "items": "On your {list_name} list: {items}.",
        "empty": "Your {list_name} list is empty.",
        "cleared": "Cleared your {list_name} list.",
    },
    "nl": {
        "added": "{items} toegevoegd aan {list_name}.",
        "items": "Op je {list_name} lijst: {items}.",
        "empty": "Je {list_name} lijst is leeg.",
        "cleared": "Je {list_name} lijst is leeggemaakt.",
    },
}


def list_template(lang: str, key: str) -> str:
    """Look up a language-appropriate template for the list-management
    confirmation strings (add/get/clear). These are built directly in
    Python rather than by Groq, so language_instruction() alone has no
    effect on them -- this is what actually controls their language."""
    templates = LIST_TEMPLATES.get(lang, LIST_TEMPLATES["en"])
    return templates[key]