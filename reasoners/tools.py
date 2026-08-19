"""Research tools — deterministic skills the research reasoners call through
the control plane.

They are registered as ``@router.skill`` (not plain helpers) on purpose: the
tool-using research reasoners discover them at runtime via
``ToolCallConfig(agent_ids=[NODE_ID], tags=["research-tool"])`` and every
invocation lands in the workflow DAG, so a research lane's trail of searches
and page fetches is inspectable after the fact.
"""

from __future__ import annotations

import os

from agentfield import AgentRouter

from media import websearch

NODE_ID = os.getenv("AGENT_NODE_ID", "video-essay-af")
router = AgentRouter(prefix="", tags=["research-tool"])


@router.skill(
    description=(
        "Search the web for a query. Returns up to max_results results as "
        "{title, url, snippet, source}. Use specific queries (names, dates, "
        "numbers) and vary phrasing between calls."
    )
)
async def web_search(query: str, max_results: int = 6) -> dict:
    results = await websearch.search(query, max_results=max(1, min(int(max_results), 10)))
    return {"query": query, "results": results, "count": len(results)}


@router.skill(
    description=(
        "Fetch a web page and return its readable text (truncated to max_chars). "
        "Use it to read a source before citing it; quotes and numbers must come "
        "from fetched text, not from memory."
    )
)
async def fetch_page(url: str, max_chars: int = 6000) -> dict:
    page = await websearch.fetch(url, max_chars=max(500, min(int(max_chars), 12000)))
    return page


__all__ = ["router", "web_search", "fetch_page"]
