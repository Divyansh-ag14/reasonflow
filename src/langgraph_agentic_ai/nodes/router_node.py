import re
from langchain_core.messages import HumanMessage, AIMessage
from src.langgraph_agentic_ai.state.state import State
from src.langgraph_agentic_ai.utils.llm_fallback import invoke_with_fallback


# ── Layer 1: Regex fast-path (zero LLM cost) ────────────────────────────
DIRECT_PATTERNS = [
    r"^(hi|hello|hey|howdy|hola|sup|yo)[\s!?.]*$",
    r"^(thanks?|thank you|thx|ty|thankyou)[\s!?.]*$",
    r"^(bye|goodbye|see ya|later|cya)[\s!?.]*$",
    r"^(ok|okay|sure|yes|no|yep|nope|cool|nice|great|awesome)[\s!?.]*$",
    r"^(what can you do|help|who are you|what are you)[\s!?.]*$",
    r"^(good morning|good night|good evening|good afternoon|gm|gn)[\s!?.]*$",
    r"^(lol|haha|hehe|wow|omg|oh)[\s!?.]*$",
]
COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in DIRECT_PATTERNS]


class RouterNode:
    """
    Smart router that classifies queries BEFORE the supervisor.
    - DIRECT: Simple chat (greetings, thanks, meta-questions) -> respond directly
    - PIPELINE: Needs agents (research, code, analysis) -> full multi-agent pipeline

    Uses automatic model fallback on rate limit errors.
    """

    def __init__(self, llm):
        self.llm = llm

    def classify(self, state: State) -> dict:
        """
        Classifies the user's message and either responds directly
        or routes to the multi-agent pipeline.
        """
        # Extract most recent user message
        user_message = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                user_message = msg.content
                break

        if not user_message:
            return {
                "route_type": "DIRECT",
                "messages": [AIMessage(content="Hello! How can I help you today?")],
            }

        # ── Layer 1: Regex fast-path ──────────────────────────────────
        clean = user_message.strip()
        for pattern in COMPILED_PATTERNS:
            if pattern.match(clean):
                # Generate a quick conversational response
                response, model_note = self._generate_direct_response(user_message)
                print(f"[router] DIRECT (regex match){f' | {model_note}' if model_note else ''}")
                return {
                    "route_type": "DIRECT",
                    "messages": [response],
                    "token_usage": self._estimate_tokens(user_message) + self._estimate_tokens(response.content),
                }

        # ── Layer 2: LLM classification ───────────────────────────────
        classify_prompt = """Classify this user message. Reply with ONLY one word: DIRECT or PIPELINE.

DIRECT — casual conversation, greetings, thanks, feelings, jokes, opinions, meta-questions about yourself, simple chat that needs no research or tools.
Examples: "how's your day?", "tell me a joke", "that's cool", "I don't understand", "what do you think about cats?"

PIPELINE — needs web research, code execution, data analysis, financial info, academic papers, comparisons, math, or any task requiring tools or specialist knowledge.
Examples: "compare NVIDIA and AMD", "write a python script", "what's NVIDIA stock price?", "research AI trends", "solve this integral"

When uncertain, respond PIPELINE."""

        try:
            classification, model_note = invoke_with_fallback(
                self.llm,
                [HumanMessage(content=f"{classify_prompt}\n\nUser message: {user_message}")],
                label="router-classify",
            )
            route = classification.content.strip().upper()
            tokens_used = self._estimate_tokens(classify_prompt + user_message) + self._estimate_tokens(classification.content)
            if model_note:
                print(f"[router] Classification used fallback: {model_note}")
        except Exception as e:
            # On error, default to PIPELINE (safer)
            print(f"[router] Classification failed ({e}), defaulting to PIPELINE")
            route = "PIPELINE"
            tokens_used = 0

        if route == "DIRECT":
            response, resp_note = self._generate_direct_response(user_message)
            tokens_used += self._estimate_tokens(user_message) + self._estimate_tokens(response.content)
            print(f"[router] DIRECT (LLM classification){f' | {resp_note}' if resp_note else ''}")
            return {
                "route_type": "DIRECT",
                "messages": [response],
                "token_usage": tokens_used,
            }

        # Default: PIPELINE
        print(f"[router] PIPELINE")
        return {
            "route_type": "PIPELINE",
            "token_usage": tokens_used,
        }

    def _generate_direct_response(self, user_message: str):
        """
        Generate a friendly conversational response for DIRECT messages.
        Returns (AIMessage, model_note).
        """
        prompt = f"""You are ReasonFlow, a helpful multi-agent AI assistant.
Respond naturally and conversationally to this message. Keep it brief and friendly.
If asked what you can do, mention you have 9 specialist agents: Researcher, Coder, Analyst, Writer, Planner, Fact-Checker, Math Solver, Data Visualizer, and Critic.

User: {user_message}"""
        try:
            response, model_note = invoke_with_fallback(
                self.llm,
                [HumanMessage(content=prompt)],
                label="router-respond",
            )
            return response, model_note
        except Exception:
            return AIMessage(
                content="Hello! I'm ReasonFlow, a multi-agent AI system. "
                "I can research topics, write code, analyze data, create visualizations, "
                "fact-check claims, solve math problems, and more. What would you like me to help with?"
            ), ""

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough estimate: ~4 chars per token."""
        return len(text) // 4 if text else 0
