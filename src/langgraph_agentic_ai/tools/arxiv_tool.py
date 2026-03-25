from langchain_community.tools import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper


def get_arxiv_tool():
    """Returns an ArXiv search tool for finding academic research papers."""
    arxiv_wrapper = ArxivAPIWrapper(top_k_results=3, doc_content_chars_max=600)
    return ArxivQueryRun(api_wrapper=arxiv_wrapper)
