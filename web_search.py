"""
Web search via DuckDuckGo's HTML results page. No signup, no API key,
no card, no query limit -- chosen after Brave killed its free tier
(Feb 2026) and Google's Custom Search JSON API turned out to be closed
to new customers entirely. Trade-off accepted: this scrapes a plain HTML
page rather than calling a real API, so it can break if DuckDuckGo
changes their markup -- reasonable for a personal tool, not something
to build a business on.

This module only fetches raw results -- it never tries to answer the
question itself. groq_answer.py feeds the raw results back to Groq to
write an actual concise, spoken answer.
"""

import requests
from bs4 import BeautifulSoup

DDG_HTML_URL = "https://html.duckduckgo.com/html/"

# A plain requests default User-Agent gets blocked/served a captcha page
# by DuckDuckGo fairly often -- a normal browser-looking one avoids that
# in most cases.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}


def search_web(query: str, count: int = 5) -> str:
    """Search DuckDuckGo's no-JS HTML results page and return a short,
    formatted block of results (title, snippet, url per result). Raises
    on a failed request -- the caller decides how to handle that."""
    response = requests.post(
        DDG_HTML_URL,
        data={"q": query},
        headers=_HEADERS,
        timeout=10,
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    results = []
    for result in soup.select(".result")[:count]:
        title_el = result.select_one(".result__title a")
        snippet_el = result.select_one(".result__snippet")
        if not title_el:
            continue
        title = title_el.get_text(strip=True)
        url = title_el.get("href", "")
        snippet = snippet_el.get_text(strip=True) if snippet_el else ""
        results.append(f"- {title}: {snippet} ({url})")

    if not results:
        return "No web results found."

    return "\n".join(results)