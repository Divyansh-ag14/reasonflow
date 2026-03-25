from langchain_core.messages import SystemMessage
from src.langgraph_agentic_ai.state.state import State


class ChatbotWithToolNode:
    """
    Chatbot logic enhanced with tool integration and ReAct reasoning.
    Supports both basic tool binding and full ReAct agent mode with
    dynamic context injection (plan + reflection feedback).
    """

    REACT_SYSTEM_PROMPT = """You are ReasonFlow, an advanced AI reasoning agent. For every request, follow this process:

1. THINK: Analyze what information or actions are needed
2. ACT: Call the appropriate tool(s) if needed
3. OBSERVE: Read tool results carefully
4. REPEAT: Continue until you have sufficient information
5. ANSWER: Provide a comprehensive, well-structured final response

Available tools and when to use them:
- tavily_search_results_json: Real-time web search for current news and events
- wikipedia: Factual background information and definitions
- arxiv: Academic papers and AI research
- get_stock_info: Stock prices and company financial data
- python_repl: Write and execute Python code for games, analysis, or computation
- calculator: Safe mathematical calculations

Always cite sources when using search tools. Be thorough and accurate."""

    def __init__(self, model):
        self.llm = model

    def create_chatbot(self, tools):
        """
        Returns a basic chatbot node function with tool binding.
        Kept for backward compatibility.
        """
        llm_with_tools = self.llm.bind_tools(tools)

        def chatbot_node(state: State):
            return {"messages": [llm_with_tools.invoke(state["messages"])]}

        return chatbot_node

    def create_react_chatbot(self, tools):
        """
        Returns a ReAct-enabled chatbot node with:
        - System prompt guiding THINK→ACT→OBSERVE→ANSWER reasoning
        - Tool binding for all available tools
        - Dynamic context injection (plan + reflection feedback)
        """
        llm_with_tools = self.llm.bind_tools(tools)

        def react_chatbot_node(state: State):
            messages = list(state["messages"])

            # Build dynamic system message with context
            context_parts = [self.REACT_SYSTEM_PROMPT]

            plan = state.get("plan")
            if plan:
                plan_text = "\n".join(f"  {i+1}. {step}" for i, step in enumerate(plan))
                context_parts.append(
                    f"\nYour execution plan:\n{plan_text}\nWork through these steps systematically."
                )

            feedback = state.get("reflection_feedback")
            if feedback and state.get("verdict") == "RETRY":
                context_parts.append(
                    f"\nPREVIOUS RESPONSE WAS INSUFFICIENT.\n"
                    f"Feedback: {feedback}\n"
                    f"Please specifically address this gap in your response."
                )

            system_msg = SystemMessage(content="\n".join(context_parts))
            full_messages = [system_msg] + messages

            return {"messages": [llm_with_tools.invoke(full_messages)]}

        return react_chatbot_node
