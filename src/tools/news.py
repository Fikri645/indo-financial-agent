"""News tool — fetches recent Indonesian market news for a company.

The *context* leg of the agent. Defaults to DuckDuckGo (free, no API key) and
upgrades to Tavily automatically if ``TAVILY_API_KEY`` is set.

Design note: the free DuckDuckGo/Yahoo/Bing scraping backends used by ``ddgs``
are inherently flaky (rate limits, intermittent timeouts). We therefore retry a
few times and — critically — log *loudly* when a dependency is missing or every
attempt fails, so "no news" is never silently confused with "tool is broken".
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass

from src import config

logger = logging.getLogger(__name__)

# Common Indonesian legal-entity suffixes that hurt news recall when included
# verbatim in a search query (e.g. "(Persero) Tbk").
_LEGAL_NOISE = re.compile(
    r"\b(PT|Tbk|Persero|Perseroan|\(Persero\))\b|[()]",
    flags=re.IGNORECASE,
)


@dataclass
class NewsItem:
    title: str
    snippet: str
    url: str
    source: str = "news"


def _clean_company(name: str) -> str:
    """Strip legal-entity noise so the query matches how the press refers to a firm."""
    cleaned = _LEGAL_NOISE.sub(" ", name)
    return re.sub(r"\s+", " ", cleaned).strip()


def fetch_news(company: str, max_results: int = 6, ticker: str = "") -> list[NewsItem]:
    """Return recent news items for ``company`` (name or ticker).

    Query is biased toward Indonesian financial coverage. ``ticker`` (if given)
    is appended because the IDX code is often the most reliable search term.
    """
    short = _clean_company(company) or company
    base_ticker = ticker.replace(".JK", "").strip()
    parts = [short]
    if base_ticker and base_ticker.upper() not in short.upper():
        parts.append(base_ticker)
    parts.append("saham kinerja keuangan")
    query = " ".join(parts)

    return search_news(query, max_results)


def search_news(query: str, max_results: int = 6) -> list[NewsItem]:
    """Run a news search for a caller-crafted ``query`` verbatim.

    Used by the tool-calling News Agent, which composes its own queries (so we do
    NOT append boilerplate here). Routes Tavily → DuckDuckGo like ``fetch_news``.
    """
    if config.TAVILY_API_KEY:
        items = _tavily_search(query, max_results)
        if items:
            return items
        logger.warning("Tavily returned no results; falling back to DuckDuckGo.")
    return _ddg_search(query, max_results)


def _ddg_search(query: str, max_results: int, retries: int = 3) -> list[NewsItem]:
    try:
        from ddgs import DDGS  # local import
    except ImportError:
        logger.warning(
            "ddgs not installed — News Agent disabled. Run `pip install ddgs` "
            "(it is listed in requirements.txt)."
        )
        return []

    for attempt in range(1, retries + 1):
        try:
            with DDGS() as ddgs:
                results = list(
                    ddgs.news(query, region="id-id", max_results=max_results)
                )
            if results:
                return [
                    NewsItem(
                        title=r.get("title", ""),
                        snippet=r.get("body", "") or r.get("excerpt", ""),
                        url=r.get("url", "") or r.get("link", ""),
                        source=r.get("source", "ddg"),
                    )
                    for r in results
                ]
            logger.info("DuckDuckGo attempt %d/%d returned 0 results.", attempt, retries)
        except Exception as exc:  # network/backend flakiness
            logger.info(
                "DuckDuckGo attempt %d/%d failed: %s", attempt, retries, exc
            )
        if attempt < retries:
            time.sleep(1.0 * attempt)  # simple linear backoff

    logger.warning(
        "News search returned no results after %d attempts (free backend is "
        "flaky — set TAVILY_API_KEY for reliable coverage).", retries
    )
    return []


def _tavily_search(query: str, max_results: int) -> list[NewsItem]:
    try:
        from tavily import TavilyClient  # local import
    except ImportError:
        logger.info("tavily not installed — using DuckDuckGo.")
        return []

    try:
        client = TavilyClient(api_key=config.TAVILY_API_KEY)
        resp = client.search(query, topic="news", max_results=max_results)
        return [
            NewsItem(
                title=r.get("title", ""),
                snippet=r.get("content", ""),
                url=r.get("url", ""),
                source="tavily",
            )
            for r in resp.get("results", [])
        ]
    except Exception as exc:
        logger.warning("Tavily search failed: %s", exc)
        return []
