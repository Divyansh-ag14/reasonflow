from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper


def get_wikipedia_tool():
    """Returns a Wikipedia search tool for structured factual knowledge lookups."""
    wiki_wrapper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=600)
    return WikipediaQueryRun(api_wrapper=wiki_wrapper)
