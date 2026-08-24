from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

from tavily import AsyncTavilyClient

OFFICIAL_ANDROID_DOMAINS = (
    "support.google.com",
    "source.android.com",
    "android.com",
)

_EXPLICIT_LOOKUP = re.compile(
    r"\b(search (?:the )?web|look up online|find official|official android|"
    r"what (?:is|does) android|android documentation)\b",
    re.IGNORECASE,
)
_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d ()-]{7,}\d)(?!\w)")
_LONG_NUMBER = re.compile(r"\b\d{6,}\b")


class TavilyError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class SearchSource:
    title: str
    url: str
    content: str


@dataclass(frozen=True, slots=True)
class SearchAnswer:
    answer: str
    sources: tuple[SearchSource, ...]
    latency_ms: int


def is_explicit_public_lookup(text: str) -> bool:
    return _EXPLICIT_LOOKUP.search(text) is not None


def redact_search_query(text: str) -> str:
    redacted = _EMAIL.sub("[email]", text)
    redacted = _PHONE.sub("[phone]", redacted)
    return _LONG_NUMBER.sub("[private number]", redacted)


class TavilySearchClient:
    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise ValueError("TAVILY_API_KEY is required")
        self.client = AsyncTavilyClient(api_key=api_key)

    async def search(self, query: str) -> SearchAnswer:
        safe_query = redact_search_query(query)
        started = time.perf_counter()
        try:
            payload = await self.client.search(
                safe_query,
                search_depth="basic",
                max_results=3,
                include_answer="basic",
                include_domains=list(OFFICIAL_ANDROID_DOMAINS),
                include_raw_content=False,
            )
        except Exception as exc:
            raise TavilyError(f"Tavily search failed: {exc}") from exc
        latency_ms = round((time.perf_counter() - started) * 1000)
        if not isinstance(payload, dict):
            raise TavilyError("Tavily returned an unexpected response")

        sources: list[SearchSource] = []
        seen_urls: set[str] = set()
        for item in payload.get("results", []):
            if not isinstance(item, dict):
                continue
            title = item.get("title")
            url = item.get("url")
            content = item.get("content")
            if all(isinstance(value, str) and value for value in (title, url, content)):
                canonical_url = url.split("?", 1)[0]
                if canonical_url in seen_urls:
                    continue
                seen_urls.add(canonical_url)
                sources.append(SearchSource(title, url, content))
        answer: Any = payload.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            answer = sources[0].content if sources else "No official result was found."
        return SearchAnswer(answer.strip(), tuple(sources), latency_ms)
