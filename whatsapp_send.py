"""
WhatsApp sending via the local whatsmeow Go service (whatsapp-service/).

Replaces the earlier pywhatkit/pyautogui browser-automation approach
(see CLAUDE.md Section 133.5 for why that was abandoned). This version
sends by POSTing to the Go service's own HTTP endpoint on localhost --
no browser, no screen-coordinate clicking, no keyboard focus stealing.

Requires:
    whatsapp-service running separately (see its own README / main.go) --
    a one-time QR-code login via WhatsApp > Linked Devices, then it stays
    connected in the background at http://localhost:8081.

    Add one WHATSAPP_CONTACT_<NAME> line per person to .env, e.g.:
        WHATSAPP_CONTACT_MOM=+31612345678
    (mirrors how GROQ_API_KEY is kept out of git via .env already.)
"""

import os
import re
import requests
import difflib

_CONTACT_PREFIX = "WHATSAPP_CONTACT_"
_SEND_URL = "http://localhost:8081/send"


def _load_contacts() -> dict[str, str]:
    contacts = {}
    for key, value in os.environ.items():
        if key.startswith(_CONTACT_PREFIX):
            name = key[len(_CONTACT_PREFIX):].lower()
            contacts[name] = value
    return contacts


CONTACTS = _load_contacts()

# Single-user server, so a module-level dict is enough to hold the one
# draft-in-progress -- no database needed, cleared once sent or cancelled.
_pending: dict | None = None


def resolve_contact(name: str) -> tuple[str, str] | None:
    """Returns (canonical_name, phone) so callers can display/confirm
    using the name we actually matched to -- not whatever raw, possibly
    mis-transcribed spelling the user's speech came in as."""
    normalized = name.strip().lower()
    # Exact match first (cheap, and avoids any risk of fuzzy-matching
    # overriding a name that already matches perfectly).
    phone = CONTACTS.get(normalized)
    if phone is not None:
        return normalized, phone
    # Fall back to fuzzy matching against known contact names, since
    # speech-to-text on non-English names (e.g. "Veroushka") often comes
    # back as phonetic variants ("Verushka", "Varuschka"). cutoff=0.6 is
    # difflib's own default -- lenient enough to catch these variants,
    # strict enough to avoid matching two genuinely different names.
    matches = difflib.get_close_matches(normalized, CONTACTS.keys(), n=1, cutoff=0.6)
    if matches:
        return matches[0], CONTACTS[matches[0]]
    return None


def draft_message(contact_name: str, message: str) -> str:
    """Stage a message for sending. Does not send it."""
    global _pending
    resolved = resolve_contact(contact_name)
    if resolved is None:
        known = ", ".join(sorted(CONTACTS)) if CONTACTS else "none configured yet"
        return f"I don't have a contact called '{contact_name}'. Known contacts: {known}."
    canonical_name, phone = resolved
    # Capitalize the message text -- the model doesn't reliably do this
    # on its own when rewriting spoken/reported speech into message form.
    message = message[0].upper() + message[1:] if message else message
    display_name = canonical_name.capitalize()
    _pending = {"contact_name": display_name, "phone": phone, "message": message}
    return f'Drafted for {display_name}: "{message}". Say send it to confirm, or cancel it to discard.'


def update_pending(message: str) -> str:
    """Replace the pending draft's message text with a corrected version,
    keeping the same contact. Mirrors draft_message's capitalization fix
    and confirmation-string format."""
    global _pending
    if _pending is None:
        return "There's no pending message to update. Draft one first."
    message = message[0].upper() + message[1:] if message else message
    _pending["message"] = message
    return f'Updated: "{message}". Say send it to confirm, or cancel it to discard.'


def cancel_pending() -> str:
    global _pending
    if _pending is None:
        return "There's no pending message to cancel."
    contact = _pending["contact_name"]
    _pending = None
    return f"Cancelled the draft to {contact}."


def send_pending() -> str:
    """Actually sends the currently-drafted message via the local
    whatsapp-service (Go/whatsmeow), which must already be running and
    logged in (see module docstring)."""
    global _pending
    if _pending is None:
        return "There's no drafted message to send. Draft one first."
    draft = _pending
    _pending = None

    # whatsapp-service expects a bare number: digits only, no "+",
    # no spaces/dashes. Contacts in .env are stored like "+31612345678".
    phone = re.sub(r"\D", "", draft["phone"])

    try:
        response = requests.post(
            _SEND_URL,
            json={"phone": phone, "message": draft["message"]},
            timeout=15,
        )
        response.raise_for_status()
        return f"Sent to {draft['contact_name']}."
    except requests.exceptions.ConnectionError:
        return (
            f"Couldn't reach whatsapp-service at {_SEND_URL} -- "
            "is it running (go run main.go in whatsapp-service/)?"
        )
    except Exception as e:
        return f"Sending to {draft['contact_name']} failed: {e}"