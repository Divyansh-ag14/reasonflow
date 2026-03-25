from typing_extensions import TypedDict, List
from langgraph.graph.message import add_messages
from typing import Annotated, Optional


def _merge_agent_results(existing, new):
    """Reducer that safely concatenates agent result lists (handles None)."""
    if existing is None:
        existing = []
    if new is None:
        return existing
    return existing + new


class State(TypedDict):
    """
    Represent the structure of the state used in graph.
    Extended with agentic fields for planning, reflection, memory,
    multi-agent collaboration, and human-in-the-loop control.
    """
    messages: Annotated[List, add_messages]
    thread_id: str
    usecase: str
    # Agentic planning fields
    plan: Optional[List[str]]
    current_step: Optional[int]
    # Self-reflection fields
    reflection_count: Optional[int]
    reflection_feedback: Optional[str]
    verdict: Optional[str]
    # AI News specific fields
    news_data: Optional[List[dict]]
    summary: Optional[str]
    frequency: Optional[str]
    filename: Optional[str]
    # Multi-agent collaboration fields
    delegation_plan: Optional[List[dict]]       # [{agent, task, depends_on}, ...]
    current_agent: Optional[str]                # Which specialist is running
    agent_task: Optional[str]                   # Task assigned to current specialist
    agent_results: Annotated[list, _merge_agent_results]  # Parallel results collector
    current_phase: Optional[int]               # Sequential chain phase (0, 1, 2...)
    # Router fields
    route_type: Optional[str]                  # "DIRECT" or "PIPELINE"
    # Human-in-the-loop fields
    auto_approve: Optional[bool]                # Skip approval gates when True
    # System design fields
    token_usage: Optional[int]                 # Running total of estimated tokens
