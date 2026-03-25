import json
from langchain_core.messages import HumanMessage, AIMessage
from src.langgraph_agentic_ai.state.state import State
from src.langgraph_agentic_ai.utils.llm_fallback import invoke_with_fallback


class ReflectionNode:
    """
    Self-reflection node that evaluates the agent's response quality.
    Implements a PASS/RETRY loop to improve output before finalizing.
    """

    MAX_RETRIES = 2

    def __init__(self, llm):
        self.llm = llm

    def reflect(self, state: State) -> dict:
        """
        Evaluates the latest AI response against the original user request.
        Returns: verdict ('PASS' or 'RETRY'), feedback, and updated retry count.
        Skips evaluation for DIRECT-routed messages (no pipeline to evaluate).
        """
        # Skip reflection for DIRECT route — no pipeline was used
        if state.get("route_type") == "DIRECT":
            return {
                "verdict": "PASS",
                "reflection_feedback": "Direct response — no pipeline to evaluate.",
                "reflection_count": 0,
            }

        # Get the most recent user request (same as supervisor)
        user_request = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                user_request = msg.content
                break

        # Get latest AI response
        last_ai_response = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                last_ai_response = msg.content
                break

        current_count = state.get("reflection_count") or 0

        # Force PASS if max retries exceeded to prevent infinite loops
        if current_count >= self.MAX_RETRIES:
            return {
                "verdict": "PASS",
                "reflection_feedback": "Maximum reflection cycles reached. Response accepted.",
                "reflection_count": current_count,
            }

        # Determine complexity for leniency
        agent_results = state.get("agent_results") or []
        num_agents = len(agent_results)
        complexity_note = ""
        if num_agents <= 1:
            complexity_note = "\nNote: This was a simple single-agent task. Be lenient — only RETRY if the response is clearly wrong or empty."
        else:
            complexity_note = f"\nNote: This was a complex {num_agents}-agent task. Verify all parts of the request are addressed."

        # Check if visualizer was involved (code blocks = chart generation)
        viz_note = ""
        agent_types = [r.get("agent", "") for r in agent_results]
        if "visualizer" in agent_types:
            viz_note = "\nIMPORTANT: If the response contains a Python code block with plt.savefig(), the Visualizer agent has ALREADY generated and saved the chart. The chart image is displayed separately in the UI. A code block with savefig IS a successful visualization — do NOT mark as RETRY for missing chart."

        prompt = f"""You are a quality evaluator. Assess whether this AI response fully addresses the user's request.

Original Request: {user_request[:500]}

AI Response: {last_ai_response[:1500]}
{complexity_note}{viz_note}
Evaluate:
1. Does it directly answer what was asked?
2. Is the information complete and accurate?
3. Is it well-structured and actionable?

Reply with ONLY valid JSON — no other text:
{{"verdict": "PASS", "feedback": "Brief reason why it's good enough."}}
OR
{{"verdict": "RETRY", "feedback": "Specific gap: [exactly what is missing or incorrect]"}}"""

        try:
            print(f"[reflector] Evaluating response quality...")
            response, model_note = invoke_with_fallback(
                self.llm,
                [HumanMessage(content=prompt)],
                label="reflector",
            )
            if model_note:
                print(f"[reflector] {model_note}")
            result = json.loads(response.content.strip())
            verdict = result.get("verdict", "PASS")
            feedback = result.get("feedback", "")
            print(f"[reflector] Verdict: {verdict}")
        except Exception:
            verdict = "PASS"
            feedback = "Reflection could not be parsed. Accepting response."

        return {
            "verdict": verdict,
            "reflection_feedback": feedback,
            "reflection_count": current_count + 1,
        }
