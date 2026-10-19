"""Web search tool backed by Tavily, exposed to the agent as `web_search()`."""


def make_web_search(api_key: str | None):
    if not api_key:
        def unavailable(query: str, max_results: int = 5) -> list[dict]:
            return [{"error": "web_search is not configured (no TAVILY_API_KEY)."}]
        return unavailable

    from tavily import TavilyClient

    client = TavilyClient(api_key=api_key)

    def web_search(query: str, max_results: int = 5) -> list[dict]:
        try:
            response = client.search(query=query, max_results=max_results)
        except Exception as exc:
            return [{"error": f"web_search failed: {exc}"}]
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": (r.get("content") or "")[:600],
            }
            for r in response.get("results", [])
        ]

    return web_search
