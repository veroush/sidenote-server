import os
from groq import Groq

_client = None

import json
from list_store import add_item, add_items, get_items, clear_list
import whatsapp_send

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "add_item",
            "description": "Add one or more items to a named list, e.g. add 'milk' and 'bread' to the 'groceries' list.",
            "parameters": {
                "type": "object",
                "properties": {
                    "list_name": {"type": "string", "description": "Name of the list, e.g. 'groceries'"},
                    "items": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "One or more items to add, e.g. ['milk', 'bread']",
                    },
                },
                "required": ["list_name", "items"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_items",
            "description": "Get all items currently on a named list.",
            "parameters": {
                "type": "object",
                "properties": {
                    "list_name": {"type": "string", "description": "Name of the list, e.g. 'groceries'"},
                },
                "required": ["list_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "clear_list",
            "description": "Remove all items from a named list, wiping it clean.",
            "parameters": {
                "type": "object",
                "properties": {
                    "list_name": {"type": "string", "description": "Name of the list, e.g. 'groceries'"},
                },
                "required": ["list_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "draft_whatsapp_message",
            "description": "Draft a WhatsApp message to a named contact. Does NOT send it -- only stages it for the user to confirm.",
            "parameters": {
                "type": "object",
                "properties": {
                    "contact_name": {"type": "string", "description": "Name of the contact, e.g. 'mom'"},
                    "message": {"type": "string", "description": "The message text to send"},
                },
                "required": ["contact_name", "message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_whatsapp_message",
            "description": "Send the currently drafted WhatsApp message. Only call this after the user has explicitly confirmed (e.g. said 'send it' or 'yes').",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_whatsapp_message",
            "description": "Discard the currently drafted WhatsApp message without sending it.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

_CONFIRM_WORDS = {"send it", "yes", "yep", "yeah", "confirm", "go ahead", "do it"}
_CANCEL_WORDS = {"cancel", "cancel it", "discard", "never mind", "nevermind", "no", "stop"}

def natural_join(items: list[str]) -> str:
    """Join a list of items the way a person would say them out loud:
    'milk', 'milk and bread', 'milk, bread and eggs'. Used anywhere a list
    of items gets turned into a spoken/displayed confirmation, so we don't
    read out a bare comma-separated dump."""
    items = list(items)
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return f"{', '.join(items[:-1])} and {items[-1]}"


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

def get_list_tool_response(question: str):
    """Ask Groq whether this question is a list add/read request. If the
    model calls one of the list tools, execute it and return a plain-text
    confirmation. If it doesn't call a tool, return None so the caller
    falls back to the existing RAG note-answering path."""

    # If a WhatsApp draft is already pending, a short confirm/cancel reply
    # carries no context Groq can see on its own (each /answer call is a
    # fresh, stateless request) -- so we check locally first, deterministically,
    # rather than trusting the model to infer state it was never given.
    if whatsapp_send._pending is not None:
        normalized = question.strip().lower()
        if normalized in _CONFIRM_WORDS:
            return whatsapp_send.send_pending()
        if normalized in _CANCEL_WORDS:
            return whatsapp_send.cancel_pending()
        # Anything else falls through to the normal Groq path below --
        # e.g. the user might be drafting a new message instead.

    client = _get_client()
    messages = [
        {
            "role": "system",
            "content": (
                "You help manage simple named lists (like groceries, todo, "
                "packing) and draft/send WhatsApp messages. If the user is "
                "asking to add something to a list, asking what's on a list, "
                "or asking to clear/wipe/empty a list, call the appropriate "
                "list tool. If the user asks you to send a WhatsApp message "
                "to someone, call draft_whatsapp_message with the message "
                "text rewritten as the actual message to send -- not the "
                "user's framing of it. For example, if the user says 'tell "
                "her she needs to pick up the groceries tomorrow', the "
                "message text should be 'pick up the groceries tomorrow', "
                "not 'she needs to pick up the groceries tomorrow'. Strip "
                "framing like 'saying', 'telling him/her', and convert "
                "third-person phrasing into direct message text. Do NOT "
                "call send_whatsapp_message yet. Only call send_whatsapp_message "
                "if the user's message is an explicit confirmation of an "
                "already-drafted message (e.g. 'send it', 'yes', 'go ahead'). "
                "If the user wants to discard a draft, call "
                "cancel_whatsapp_message. Otherwise, do not call any tool."
            ),
        },
        {"role": "user", "content": question},
    ]
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        temperature=0,
    )
    msg = response.choices[0].message
    if not msg.tool_calls:
        return None

    results = []
    for tool_call in msg.tool_calls:
        args = json.loads(tool_call.function.arguments)

        if tool_call.function.name == "add_item":
            add_items(args["list_name"], args["items"])
            results.append(f"Added {natural_join(args['items'])} to {args['list_name']}.")
        elif tool_call.function.name == "get_items":
            items = get_items(args["list_name"])
            if items:
                results.append(f"On your {args['list_name']} list: {natural_join(items)}.")
            else:
                results.append(f"Your {args['list_name']} list is empty.")
        elif tool_call.function.name == "clear_list":
            clear_list(args["list_name"])
            results.append(f"Cleared your {args['list_name']} list.")
        elif tool_call.function.name == "draft_whatsapp_message":
            results.append(whatsapp_send.draft_message(args["contact_name"], args["message"]))
        elif tool_call.function.name == "send_whatsapp_message":
            results.append(whatsapp_send.send_pending())
        elif tool_call.function.name == "cancel_whatsapp_message":
            results.append(whatsapp_send.cancel_pending())

    return " ".join(results) if results else None