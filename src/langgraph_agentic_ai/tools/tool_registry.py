from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode
from src.langgraph_agentic_ai.tools.calculator_tool import calculator
from src.langgraph_agentic_ai.tools.arxiv_tool import get_arxiv_tool
from src.langgraph_agentic_ai.tools.wikipedia_tool import get_wikipedia_tool
from src.langgraph_agentic_ai.tools.finance_tool import get_stock_info
from src.langgraph_agentic_ai.tools.python_repl_tool import get_python_repl_tool
from src.langgraph_agentic_ai.tools.ddg_search_tool import get_ddg_search_tool
from src.langgraph_agentic_ai.tools.web_scraper_tool import scrape_webpage
from src.langgraph_agentic_ai.tools.youtube_tool import get_youtube_transcript


def get_tools(usecase: str = "Basic Chatbot") -> list:
    """
    Returns the list of tools available for a given use case.
    - Basic Chatbot:     no tools
    - Chatbot With Web:  Tavily + Wikipedia + ArXiv + Finance + Calculator
    - AI News:           Tavily + Wikipedia + ArXiv
    - ReasonFlow Agent:  all of the above + Python REPL
    """
    if usecase == "Basic Chatbot":
        return []

    base_tools = [
        calculator,
        get_wikipedia_tool(),
    ]

    if usecase in ("Chatbot With Web", "ReasonFlow Agent"):
        base_tools += [
            TavilySearchResults(max_results=3),
            get_arxiv_tool(),
            get_stock_info,
        ]

    if usecase == "AI News":
        base_tools += [
            TavilySearchResults(max_results=5),
            get_arxiv_tool(),
        ]

    if usecase == "ReasonFlow Agent":
        base_tools += [get_python_repl_tool()]

    return base_tools


# ── Specialist agent tool getters (for multi-agent collaboration) ──────────

def get_researcher_tools() -> list:
    """Researcher: richest toolset — web search, DuckDuckGo, Wikipedia, ArXiv, web scraper, YouTube."""
    return [
        TavilySearchResults(max_results=3),
        get_ddg_search_tool(),
        get_wikipedia_tool(),
        get_arxiv_tool(),
        scrape_webpage,
        get_youtube_transcript,
    ]


def get_coder_tools() -> list:
    """Coder: Python REPL for code execution."""
    return [get_python_repl_tool()]


def get_analyst_tools() -> list:
    """Analyst: finance data, calculator, web search."""
    return [
        get_stock_info,
        calculator,
        TavilySearchResults(max_results=2),
    ]


def get_writer_tools() -> list:
    """Writer: pure LLM — no tools needed."""
    return []


def get_planner_tools() -> list:
    """Planner: web search for options research + calculator."""
    return [
        TavilySearchResults(max_results=2),
        calculator,
    ]


def get_fact_checker_tools() -> list:
    """Fact-Checker: multiple search sources for claim verification."""
    return [
        TavilySearchResults(max_results=3),
        get_ddg_search_tool(),
        scrape_webpage,
    ]


def get_math_solver_tools() -> list:
    """Math Solver: calculator + Python REPL for verification."""
    return [
        calculator,
        get_python_repl_tool(),
    ]


def get_visualizer_tools() -> list:
    """Data Visualizer: no tools — generates code as text, executed by specialist runner."""
    return []


def get_critic_tools() -> list:
    """Critic: pure LLM — reviews quality without tools."""
    return []


def get_specialist_tools(agent_type: str) -> list:
    """Returns tools for a given specialist agent type."""
    registry = {
        "researcher": get_researcher_tools,
        "coder": get_coder_tools,
        "analyst": get_analyst_tools,
        "writer": get_writer_tools,
        "planner": get_planner_tools,
        "fact_checker": get_fact_checker_tools,
        "math_solver": get_math_solver_tools,
        "visualizer": get_visualizer_tools,
        "critic": get_critic_tools,
    }
    getter = registry.get(agent_type, get_researcher_tools)
    return getter()


def create_tool_node(tools: list) -> ToolNode:
    """Creates and returns a ToolNode for the given tools."""
    return ToolNode(tools=tools)
