from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import tools_condition
from src.langgraph_agentic_ai.state.state import State
from src.langgraph_agentic_ai.nodes.basic_chatbot_node import BasicChatbotNode
from src.langgraph_agentic_ai.nodes.chatbot_with_tool_node import ChatbotWithToolNode
from src.langgraph_agentic_ai.nodes.ai_news_node import AINewsNode
from src.langgraph_agentic_ai.nodes.reflection_node import ReflectionNode
from src.langgraph_agentic_ai.nodes.supervisor_node import SupervisorNode
from src.langgraph_agentic_ai.nodes.specialist_agents import SpecialistAgents
from src.langgraph_agentic_ai.nodes.router_node import RouterNode
from src.langgraph_agentic_ai.tools.tool_registry import get_tools, create_tool_node

try:
    from langgraph.types import Send
except ImportError:
    Send = None


class GraphBuilder:
    def __init__(self, model, checkpointer=None):
        self.llm = model
        self.checkpointer = checkpointer
        self.graph_builder = StateGraph(State)

    def basic_chatbot_build_graph(self):
        """
        Builds a stateful basic chatbot graph.
        Graph: START -> chatbot -> END
        Memory is provided by the checkpointer, enabling multi-turn conversations.
        """
        basic_chatbot_node = BasicChatbotNode(self.llm)
        self.graph_builder.add_node("chatbot", basic_chatbot_node.process)
        self.graph_builder.add_edge(START, "chatbot")
        self.graph_builder.add_edge("chatbot", END)

    def chatbot_with_tools_build_graph(self):
        """
        Builds an enriched ReAct chatbot with multiple tools.
        Graph: START -> chatbot <-> tools (ReAct loop) -> END
        Tools: Tavily web search, Wikipedia, ArXiv, Finance, Calculator
        """
        tools = get_tools("Chatbot With Web")
        tool_node = create_tool_node(tools)
        chatbot_node = ChatbotWithToolNode(self.llm).create_react_chatbot(tools)

        self.graph_builder.add_node("chatbot", chatbot_node)
        self.graph_builder.add_node("tools", tool_node)
        self.graph_builder.add_edge(START, "chatbot")
        self.graph_builder.add_conditional_edges("chatbot", tools_condition)
        self.graph_builder.add_edge("tools", "chatbot")

    def ai_news_agent_build_graph(self):
        """
        Builds an agent-driven AI news graph.
        The LLM agent autonomously crafts search queries and gathers news via tools.
        Graph: START -> news_agent <-> tools -> save_result -> END
        """
        tools = get_tools("AI News")
        tool_node = create_tool_node(tools)
        ai_news_node = AINewsNode(self.llm)
        news_agent_node = ChatbotWithToolNode(self.llm).create_react_chatbot(tools)

        self.graph_builder.add_node("news_agent", news_agent_node)
        self.graph_builder.add_node("tools", tool_node)
        self.graph_builder.add_node("save_result", ai_news_node.save_result)

        self.graph_builder.add_edge(START, "news_agent")
        self.graph_builder.add_conditional_edges(
            "news_agent",
            tools_condition,
            {"tools": "tools", END: "save_result"},
        )
        self.graph_builder.add_edge("tools", "news_agent")
        self.graph_builder.add_edge("save_result", END)

    def reasonflow_agent_build_graph(self):
        """
        Builds the flagship ReasonFlow Agent — a multi-agent system with:
        - Router: Smart classification (DIRECT vs PIPELINE) to skip pipeline for casual chat
        - Supervisor: Plans which specialists to dispatch (with depends_on for sequential chains)
        - Parallel Dispatch: Sends tasks to specialist agents concurrently via Send()
        - 9 Specialists: Researcher, Coder, Analyst, Writer, Planner, Fact-Checker,
                         Math Solver, Visualizer, Critic
        - Phase Loop: Sequential agent chains via depends_on (Phase 0 → Phase 1 → ...)
        - Human-in-the-Loop: Optional plan approval gate via interrupt()
        - Self-Reflection: Evaluates output quality, retries if needed

        Graph:
            START -> router --DIRECT--> END
                           --PIPELINE--> supervisor_plan -> [specialists] (parallel per phase)
                                -> supervisor_synthesize --(more phases?)--> supervisor_plan
                                                        --(all done)--> reflector -> END
        """
        router = RouterNode(self.llm)
        supervisor = SupervisorNode(self.llm)
        specialist = SpecialistAgents(self.llm)
        reflection_node = ReflectionNode(self.llm)

        # ── Route decision: DIRECT → END, PIPELINE → supervisor ──────────
        def route_decision(state: State) -> str:
            return "done" if state.get("route_type") == "DIRECT" else "pipeline"

        # ── Parallel dispatch via Send (phase-aware) ─────────────────────
        def dispatch_to_specialists(state: State):
            """Routes to specialist agents using Send(). Phase-aware:
            only dispatches agents whose depends_on are all satisfied."""
            plan = state.get("delegation_plan") or []
            if not plan or Send is None:
                return "supervisor_synthesize"

            completed = {r["agent"] for r in (state.get("agent_results") or [])}

            # Find agents ready to run: not yet completed, all deps satisfied
            pending = [
                t for t in plan
                if t["agent"] not in completed
                and all(d in completed for d in t.get("depends_on", []))
            ]

            if not pending:
                # All done or circular dependency — go to synthesize
                return "supervisor_synthesize"

            # Inject dependency context into tasks
            agent_results = state.get("agent_results") or []
            sends = []
            for t in pending:
                task_text = t["task"]
                for dep in t.get("depends_on", []):
                    dep_result = next(
                        (r for r in agent_results if r["agent"] == dep), None
                    )
                    if dep_result:
                        task_text = (
                            f"=== DATA FROM {dep.upper()} AGENT (use this real data, NOT placeholder/dummy data) ===\n"
                            f"{dep_result['result'][:2000]}\n"
                            f"=== END OF {dep.upper()} AGENT DATA ===\n\n"
                            f"YOUR TASK: {task_text}"
                        )
                sends.append(Send("specialist", {
                    "messages": state["messages"],
                    "current_agent": t["agent"],
                    "agent_task": task_text,
                    "agent_results": [],
                    "delegation_plan": state.get("delegation_plan"),
                    "auto_approve": state.get("auto_approve"),
                }))
            return sends

        # ── After synthesis: more phases or reflect? ─────────────────────
        def check_phases_or_reflect(state: State) -> str:
            """Routes to 'next_phase' if more agents pending, else 'reflect'."""
            plan = state.get("delegation_plan") or []
            completed = {r["agent"] for r in (state.get("agent_results") or [])}
            remaining = [t for t in plan if t["agent"] not in completed]
            if remaining:
                return "next_phase"
            return "reflect"

        # ── Reflection routing ────────────────────────────────────────────
        def should_retry(state: State) -> str:
            """Routes to 'retry' if reflection verdict is RETRY, else 'done'."""
            if (
                state.get("verdict") == "RETRY"
                and (state.get("reflection_count") or 0) < ReflectionNode.MAX_RETRIES
            ):
                return "retry"
            return "done"

        # ── Register nodes ────────────────────────────────────────────────
        self.graph_builder.add_node("router", router.classify)
        self.graph_builder.add_node("supervisor_plan", supervisor.plan)
        self.graph_builder.add_node("specialist", specialist.run)
        self.graph_builder.add_node("supervisor_synthesize", supervisor.synthesize)
        self.graph_builder.add_node("reflector", reflection_node.reflect)

        # ── Wire edges ────────────────────────────────────────────────────
        self.graph_builder.add_edge(START, "router")

        # Router decides: DIRECT → END, PIPELINE → supervisor
        self.graph_builder.add_conditional_edges(
            "router",
            route_decision,
            {"done": END, "pipeline": "supervisor_plan"},
        )

        # Supervisor plan dispatches to specialists (phase-aware).
        self.graph_builder.add_conditional_edges(
            "supervisor_plan",
            dispatch_to_specialists,
            ["specialist", "supervisor_synthesize"],
        )

        # All specialist branches converge to synthesize
        self.graph_builder.add_edge("specialist", "supervisor_synthesize")

        # Synthesize decides: more phases → back to supervisor_plan, or → reflect
        self.graph_builder.add_conditional_edges(
            "supervisor_synthesize",
            check_phases_or_reflect,
            {"next_phase": "supervisor_plan", "reflect": "reflector"},
        )

        # Reflection decides: retry (back to supervisor) or done
        self.graph_builder.add_conditional_edges(
            "reflector",
            should_retry,
            {"retry": "supervisor_plan", "done": END},
        )

    def setup_graph(self, usecase: str):
        """Sets up and compiles the graph for the selected use case."""
        if usecase == "Basic Chatbot":
            self.basic_chatbot_build_graph()
        elif usecase == "Chatbot With Web":
            self.chatbot_with_tools_build_graph()
        elif usecase == "AI News":
            self.ai_news_agent_build_graph()
        elif usecase == "ReasonFlow Agent":
            self.reasonflow_agent_build_graph()

        return self.graph_builder.compile(checkpointer=self.checkpointer)
