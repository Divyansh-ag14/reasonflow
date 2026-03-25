import json
from langchain_core.messages import HumanMessage
from src.langgraph_agentic_ai.state.state import State


class PlanningNode:
    """
    Breaks down a complex user request into a structured list of actionable subtasks.
    Enables the ReasonFlow agent to work through tasks systematically.
    """

    def __init__(self, llm):
        self.llm = llm

    def create_plan(self, state: State) -> dict:
        """
        Analyzes the latest user request and generates a step-by-step execution plan.
        Returns updated state with 'plan' (list of steps) and 'current_step' set to 0.
        """
        # Get the most recent HumanMessage
        user_request = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                user_request = msg.content
                break

        prompt = f"""You are a planning assistant for an AI agent. Break down the following user request into 3-5 concrete, actionable subtasks.

User Request: {user_request}

Rules:
- Each subtask should be specific and achievable with available tools
- Order them logically (research/gather first, synthesize/answer last)
- Keep each step concise (one sentence)

Respond with ONLY a valid JSON array of strings. No explanation, no markdown, just the JSON array.
Example: ["Search for recent news about X using web search", "Look up background on Y using Wikipedia", "Summarize findings into a structured report"]"""

        response = self.llm.invoke([HumanMessage(content=prompt)])

        try:
            plan = json.loads(response.content.strip())
            if not isinstance(plan, list) or len(plan) == 0:
                raise ValueError("Invalid plan format")
        except (json.JSONDecodeError, ValueError):
            # Fallback: extract numbered/bulleted lines
            lines = [
                line.strip().lstrip("0123456789.-) ").strip()
                for line in response.content.split("\n")
                if line.strip()
            ]
            plan = [l for l in lines if len(l) > 5] or [user_request]

        return {
            "plan": plan,
            "current_step": 0,
            "reflection_count": 0,
        }
