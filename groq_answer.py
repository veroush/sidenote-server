import os
from groq import Groq

_client = None

import json
from list_store import add_item, add_items, get_items, clear_list
from personality import add_flavor, personality_preamble, language_instruction, list_template
from langdetect import detect, DetectorFactory
import whatsapp_send
import memory_store as mem
from web_search import search_web
import conversation_state as convo

# langdetect samples n-grams randomly under the hood, so without a fixed
# seed the same text can occasionally get classified differently between
# runs. Fixing it makes detection repeatable.
DetectorFactory.seed = 0


def detect_language(text: str) -> str:
    """Best-effort detection of whether text is Dutch or English. Falls
    back to English on anything short/ambiguous/undetectable -- langdetect
    is unreliable on very short phrases, and a wrong guess toward Dutch is
    more disruptive than defaulting to English."""
    try:
        code = detect(text)
    except Exception:
        return "en"
    return "nl" if code == "nl" else "en"

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
            "description": "Send the currently drafted WhatsApp message. Only call this after the user has explicitlyconfirmed (e.g. said 'send it' or 'yes').",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_whatsapp_message",
            "description": "Replace the currently pending WhatsApp draft's message text with a corrected version. The contact stays the same as the existing draft -- only pass the revised message.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "The corrected message text to replace the current draft with"},
                },
                "required": ["message"],
            },
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
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for current information -- news, weather, sports scores, recent events, prices, or any fact you're not confident about or that could have changed since you last knew about it. Use a short, focused search query, not the user's full sentence.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "A short web search query, e.g. 'weather Amsterdam today' or 'Lakers score last night'"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wipe_memories",
            "description": "Delete stored memories from a specific day the user mentions, e.g. 'wipe everything from this day 3 months ago' or 'forget what I said 3 maanden geleden'. Pass the date phrase from the user's own words -- do not convert it to a calendar date yourself.",
            "parameters": {
                "type": "object",
                "properties": {
                    "phrase": {"type": "string", "description": "The date phrase as the user said it, e.g. 'this day 3 months ago'"},
                },
                "required": ["phrase"],
            },
        },
    },
]

_CONFIRM_WORDS = {"send it", "yes", "yep", "yeah", "confirm", "go ahead", "do it"}
_CANCEL_WORDS = {"cancel", "cancel it", "discard", "never mind", "nevermind", "stop"}
_BARE_NO_WORDS = {"no"}

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
    """Answer a question, using the given notes as context when relevant.
    No longer hard-gated on a PDF being loaded -- if the notes are empty
    or don't cover the question, this falls back to Groq's own general
    knowledge instead. Raises on any failure -- the caller (server.py)
    decides what to do (e.g. return an error to the client)."""
    lang = detect_language(question)
    prompt = f"""{personality_preamble()}

{language_instruction(lang)}

Answer the question concisely, in one or two sentences, suitable for
being spoken aloud.

If the notes below are relevant to the question, use them as your
primary source. If they don't cover the question, or there are no notes
at all, answer from your own general knowledge instead -- don't say the
notes don't contain an answer, just answer normally.

Notes:
{notes if notes else "(none)"}

Question: {question}

Answer:"""

    client = _get_client()
    history_messages = []
    for turn in convo.get_history():
        history_messages.append({"role": "user", "content": turn["question"]})
        history_messages.append({"role": "assistant", "content": turn["answer"]})

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=history_messages + [{"role": "user", "content": prompt}],
        temperature=0,
    )
    answer = add_flavor(response.choices[0].message.content.strip())
    convo.add_turn(question, answer)
    return answer

def _get_draft_correction(question: str) -> tuple[str, str]:
    """Ask Groq to rewrite the currently pending WhatsApp draft's message
    based on a correction like 'no, it should say X' or 'make it more
    formal'. Includes the current draft text as context since each
    /answer call is otherwise stateless and Groq can't see it on its own.

    Deliberately does NOT record anything into conversation_state -- this
    is an intermediate step in a still-pending draft, not a completed
    action. Only a successful send gets recorded (see send_pending's
    call sites in get_list_tool_response)."""
    current_message = whatsapp_send._pending["message"]
    client = _get_client()
    lang = detect_language(question)
    messages = [
        {
            "role": "system",
            "content": (
                personality_preamble() + " " + language_instruction(lang) + " "
                "The user has a pending WhatsApp draft and is correcting "
                "it. Rewrite the message text based on their correction, "
                "applying the same reported-speech-to-direct-message "
                "rewriting rule as before (strip framing like 'saying', "
                "'tell him/her', convert third-person phrasing into direct "
                "message text). Then call update_whatsapp_message with the "
                "full corrected message text -- not just the changed part. "
                "For example, if the current draft is 'Pick up the "
                "groceries tomorrow' and the user says 'no, make it "
                "Friday instead', call update_whatsapp_message with "
                "message='Pick up the groceries Friday'. Always call "
                "update_whatsapp_message -- never respond with plain text."
            ),
        },
        {
            "role": "user",
            "content": f"Current draft: \"{current_message}\"\n\nCorrection: {question}",
        },
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
        return add_flavor("Sorry, I didn't catch what to change it to -- what should it say instead?"), "whatsapp"

    results = []
    for tool_call in msg.tool_calls:
        if tool_call.function.name == "update_whatsapp_message":
            args = json.loads(tool_call.function.arguments)
            results.append(whatsapp_send.update_pending(args["message"]))
    answer = add_flavor(" ".join(results)) if results else add_flavor("Sorry, I didn't catch what to change it to -- what should it say instead?")
    return answer, "whatsapp"


def _get_search_answer(question: str, search_query: str) -> tuple[str, str]:
    """Run a web search and ask Groq to turn the raw results into a
    concise, spoken-friendly answer -- never just reads back snippets/URLs.
    Falls back to a plain spoken error if the search itself fails (missing
    key, network issue, rate limit) rather than raising all the way up."""
    lang = detect_language(question)
    try:
        raw_results = search_web(search_query)
    except Exception as e:
        print(f"Web search failed: {e}")
        return add_flavor("I couldn't search the web just now -- want me to try again in a bit?"), "search"

    client = _get_client()
    prompt = f"""{personality_preamble()}

{language_instruction(lang)}

Answer the question concisely, in one or two sentences, suitable for
being spoken aloud, using the web search results below as your source.
Don't read out URLs. If the results don't actually answer the question,
say so briefly rather than guessing.

Web search results for "{search_query}":
{raw_results}

Question: {question}

Answer:"""
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    answer = add_flavor(response.choices[0].message.content.strip())
    # Recorded so a follow-up like "what about tomorrow" right after a
    # weather/news search has something to resolve against.
    convo.add_turn(question, answer)
    return answer, "search"

def get_list_tool_response(question: str):
    """Ask Groq whether this question is a list/WhatsApp/search/memory
    request. If the model calls one of the tools, execute it and return
    (answer, source). If it doesn't call a tool, return None so the
    caller falls back to the existing RAG note-answering path."""

    # A pending memory wipe takes priority over anything else -- deleting
    # data is higher-stakes than a WhatsApp draft, so we resolve this
    # first, deterministically, before Groq sees the message at all.
    if mem.has_pending_wipe():
        normalized = question.strip().lower()
        if normalized in _CONFIRM_WORDS:
            return add_flavor(mem.confirm_wipe()), "memory"
        if normalized in _CANCEL_WORDS:
            return add_flavor(mem.cancel_wipe()), "memory"
        return add_flavor("I still need a yes or no on that -- confirm the wipe, or cancel it?"), "memory"

    # If a WhatsApp draft is already pending, a short confirm/cancel reply
    # carries no context Groq can see on its own (each /answer call is a
    # fresh, stateless request) -- so we check locally first, deterministically,
    # rather than trusting the model to infer state it was never given.
    if whatsapp_send._pending is not None:
        normalized = question.strip().lower()
        if normalized in _CONFIRM_WORDS:
            # Capture the draft before send_pending() clears it -- we need
            # its contact/message to build a useful synthetic turn, and
            # only record on an actual successful send (see docstring note
            # in _get_draft_correction for why intermediate steps aren't
            # recorded).
            draft_before = whatsapp_send._pending
            result = whatsapp_send.send_pending()
            answer = add_flavor(result)
            if draft_before is not None and result.startswith("Sent to"):
                synthetic_q = f"send a WhatsApp message to {draft_before['contact_name']}: {draft_before['message']}"
                convo.add_turn(synthetic_q, answer)
            return answer, "whatsapp"
        if normalized in _CANCEL_WORDS:
            return add_flavor(whatsapp_send.cancel_pending()), "whatsapp"
        if normalized in _BARE_NO_WORDS:
            return add_flavor("Okay, what should it say instead?"), "whatsapp"
        # Anything else with a draft pending is treated as a probable
        # correction (e.g. "no, it should say X") -- route to Groq with
        # the current draft's text included as context, so it can call
        # update_whatsapp_message with a properly rewritten replacement.
        return _get_draft_correction(question)

    client = _get_client()
    lang = detect_language(question)
    history_messages = []
    for turn in convo.get_history():
        history_messages.append({"role": "user", "content": turn["question"]})
        history_messages.append({"role": "assistant", "content": turn["answer"]})
    messages = [
        {
            "role": "system",
            "content": (
                personality_preamble() + " " + language_instruction(lang) + " "
                "You help manage simple named lists (like groceries, todo, "
                "packing) and draft/send WhatsApp messages. If the user is "
                "asking to add something to a list, asking what's on a list, "
                "or asking to clear/wipe/empty a list, call the appropriate "
                "list tool. If the user asks to forget, wipe, or delete "
                "memories from a specific day (e.g. 'wipe everything from "
                "this day 3 months ago'), call wipe_memories with the date "
                "phrase exactly as they said it. "
                "If the user asks something that needs current information -- "
                "news, weather, sports scores, recent events, prices, or any "
                "fact you're not confident about or that could have changed "
                "recently -- call search_web with a short, focused search "
                "query. This includes casual phrasings like 'can you Google "
                "it', 'look that up', 'check online', or 'search for that' -- "
                "treat these exactly like a direct request for current "
                "information and call search_web. When in doubt about whether "
                "something might be outdated, prefer calling search_web over "
                "answering from memory. "
                "If the user asks you to send a WhatsApp message "
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
        messages=history_messages + messages,
        tools=TOOLS,
        tool_choice="auto",
        temperature=0,
    )
    msg = response.choices[0].message
    if not msg.tool_calls:
        return None

    results = []
    source = "list"
    for tool_call in msg.tool_calls:
        args = json.loads(tool_call.function.arguments)

        if tool_call.function.name == "add_item":
            add_items(args["list_name"], args["items"])
            results.append(list_template(lang, "added").format(
                items=natural_join(args["items"]), list_name=args["list_name"]))
        elif tool_call.function.name == "get_items":
            items = get_items(args["list_name"])
            if items:
                results.append(list_template(lang, "items").format(
                    list_name=args["list_name"], items=natural_join(items)))
            else:
                results.append(list_template(lang, "empty").format(list_name=args["list_name"]))
        elif tool_call.function.name == "clear_list":
            clear_list(args["list_name"])
            results.append(list_template(lang, "cleared").format(list_name=args["list_name"]))
        elif tool_call.function.name == "draft_whatsapp_message":
            results.append(whatsapp_send.draft_message(args["contact_name"], args["message"]))
            source = "whatsapp"
        elif tool_call.function.name == "send_whatsapp_message":
            # Same capture-before-clear approach as the confirm-shortcut
            # path above -- only record a synthetic turn if this actually
            # succeeded.
            draft_before = whatsapp_send._pending
            result = whatsapp_send.send_pending()
            results.append(result)
            source = "whatsapp"
            if draft_before is not None and result.startswith("Sent to"):
                synthetic_q = f"send a WhatsApp message to {draft_before['contact_name']}: {draft_before['message']}"
                convo.add_turn(synthetic_q, add_flavor(result))
        elif tool_call.function.name == "cancel_whatsapp_message":
            results.append(whatsapp_send.cancel_pending())
            source = "whatsapp"
        elif tool_call.function.name == "update_whatsapp_message":
            results.append(whatsapp_send.update_pending(args["message"]))
            source = "whatsapp"
        elif tool_call.function.name == "search_web":
            # Returns immediately rather than joining into `results` --
            # this needs its own follow-up Groq call to turn raw search
            # results into an actual answer, unlike the other tools here
            # which just return a plain confirmation string. (Recording
            # into conversation_state happens inside _get_search_answer
            # itself.)
            return _get_search_answer(question, args["query"])
        elif tool_call.function.name == "wipe_memories":
            result = mem.stage_wipe(args["phrase"])
            if result is None:
                results.append("I couldn't tell which date you meant -- can you say it differently?")
            else:
                results.append(result)
            source = "memory"

    if not results:
        return None
    answer = add_flavor(" ".join(results))
    # Record list/memory outcomes so follow-ups like "add bread too" have
    # something to resolve against. WhatsApp is handled separately above
    # (only on a completed send, via a synthetic turn) since its
    # draft/correct/confirm loop is already resolved locally without
    # needing conversation_state.
    if source in ("list", "memory"):
        convo.add_turn(question, answer)
    return answer, source