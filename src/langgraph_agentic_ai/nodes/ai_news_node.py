import os
from langchain_core.messages import AIMessage
from src.langgraph_agentic_ai.state.state import State


class AINewsNode:
    """
    AI News node for saving agent-generated news summaries to markdown files.
    In the new agent mode, the ReAct agent handles fetching and summarizing via tools.
    This node handles the final save step after the agent completes.
    """

    def __init__(self, llm):
        self.llm = llm

    def save_result(self, state: State) -> dict:
        """
        Saves the AI news summary to a markdown file.
        Extracts the summary from the last AIMessage in state.
        """
        frequency = state.get("frequency") or "daily"

        # Extract the summary from the last meaningful AI message
        summary = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                summary = msg.content
                break

        os.makedirs("./AINews", exist_ok=True)
        filename = f"./AINews/{frequency.lower()}_summary.md"

        with open(filename, "w") as f:
            f.write(f"# {frequency.capitalize()} AI News Summary\n\n")
            f.write(summary)

        return {"filename": filename, "summary": summary}
