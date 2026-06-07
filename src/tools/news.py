"""News tool — fetches recent Indonesian market news for a company.

The *context* leg of the agent. Defaults to DuckDuckGo (free, no API key) and
upgrades to Tavily automatically if ``TAVILY_API_KEY`` is set.
"""
from __future__ import annotations

from dataclasses import dataclass

from src import config


@dataclass
class NewsItem:
    title: str
    snippet: str
    url: str
    source: str = "news"


def fetch_news(company: str, max_results: int = 6) -> list[NewsItem]:
    """Return recent news items for ``company`` (name or ticker).

    Query is biased toward Indonesian financial coverage.
    """
    query = f"{company} kinerja keuangan saham berita terbaru"
    if config.TAVILY_API_KEY:
        return _tavily_search(query, max_results)
    return _ddg_search(query, max_results)


def _ddg_search(query: str, max_results: int) -> list[NewsItem]:
    try:
        from ddgs import DDGS  # local import
    except ImportError:  # pragma: no cover
        return []

    items: list[NewsItem] = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.news(query, region="id-id", max_results=max_results):
                items.append(
                    NewsItem(
                        title=r.get("title", ""),
                        snippet=r.get("body", "") or r.get("excerpt", ""),
                        url=r.get("url", "") or r.get("link", ""),
                        source=r.get("source", "ddg"),
                    )
                )
    except Exception:  # pragma: no cover - network dependent
        return []
    return items


def _tavily_search(query: str, max_results: int) -> list[NewsItem]:
    try:
        from tavily import TavilyClient  # local import
    except ImportError:  # pragma: no cover
        return _ddg_search(query, max_results)

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
    except Exception:  # pragma: no cover - network dependent
        return _ddg_search(query, max_results)
