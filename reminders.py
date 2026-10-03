"""
Reminder helpers for Jarvis.

The model never does date maths. It only passes what the user said:
  - delay_minutes   ("in 5 minutes"  -> 5)
  - time_24h        ("at 8pm"        -> "20:00")
  - days_ahead      ("tomorrow at 8" -> 1)
This module turns that into a delay in seconds using the laptop's real
clock and timezone. The phone adds the delay to ITS OWN clock, so a small
difference between the two clocks doesn't matter.
"""

from datetime import datetime, timedelta

_TEMPLATES = {
    "en": {
        "in": "Okay, I'll remind you to {task} in {when}.",
        "at": "Okay, I'll remind you to {task} at {when}.",
        "spoken": "Reminder to {task}.",
        "ask": "When should I remind you?",
        "minute": "minute",
        "minutes": "minutes",
    },
    "nl": {
        "in": "Oke, herinnering gezet: {task}, over {when}.",
        "at": "Oke, herinnering gezet: {task}, om {when}.",
        "spoken": "Herinnering: {task}.",
        "ask": "Wanneer moet ik je eraan herinneren?",
        "minute": "minuut",
        "minutes": "minuten",
    },
}


def _t(lang: str) -> dict:
    return _TEMPLATES.get(lang, _TEMPLATES["en"])


def resolve_delay_seconds(delay_minutes=None, time_24h=None, days_ahead=None, now=None):
    """Return the delay in whole seconds (at least 1), or None if the
    arguments don't describe a usable time."""
    now = now or datetime.now().astimezone()

    if delay_minutes is not None:
        try:
            seconds = round(float(delay_minutes) * 60)
        except (TypeError, ValueError):
            return None
        return max(1, seconds) if seconds > 0 else None

    if time_24h:
        try:
            hour_s, minute_s = str(time_24h).strip().split(":")
            hour, minute = int(hour_s), int(minute_s)
        except ValueError:
            return None
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            return None
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        target += timedelta(days=int(days_ahead or 0))
        if target <= now:
            target += timedelta(days=1)  # "at 8pm" said at 9pm means tomorrow
        return max(1, round((target - now).total_seconds()))

    return None


def confirmation(lang: str, task: str, delay_seconds: int, now=None) -> str:
    """Spoken confirmation, e.g. 'Okay, I'll remind you to call Ben in 5 minutes.'"""
    t = _t(lang)
    now = now or datetime.now().astimezone()
    if delay_seconds < 3600:
        minutes = max(1, round(delay_seconds / 60))
        unit = t["minute"] if minutes == 1 else t["minutes"]
        return t["in"].format(task=task, when=f"{minutes} {unit}")
    target = now + timedelta(seconds=delay_seconds)
    target = (target + timedelta(seconds=30)).replace(second=0, microsecond=0)
    return t["at"].format(task=task, when=target.strftime("%H:%M"))


def spoken_text(lang: str, task: str) -> str:
    """What Jarvis says when you ask about the notification."""
    return _t(lang)["spoken"].format(task=task)


def ask_when(lang: str) -> str:
    return _t(lang)["ask"]