from langchain_community.tools import DuckDuckGoSearchResults


def get_ddg_search_tool():
    """DuckDuckGo web search — free, no API key needed."""
    return DuckDuckGoSearchResults(max_results=3)
