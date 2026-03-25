import json
from langchain_core.messages import HumanMessage, AIMessage
from src.langgraph_agentic_ai.state.state import State
from src.langgraph_agentic_ai.utils.llm_fallback import invoke_with_fallback

try:
    from langgraph.types import interrupt
except ImportError:
    interrupt = None


class SupervisorNode:
    """
    Supervisor agent that orchestrates multi-agent collaboration.
    - plan(): Analyzes user request, decides which specialists to dispatch,
              and optionally pauses for human approval (HITL).
    - synthesize(): Combines results from all specialist agents into
                    a unified, comprehensive response.

    Uses automatic model fallback on rate limit errors.
    """

    AVAILABLE_AGENTS = {
        "researcher": "Web search, DuckDuckGo, Wikipedia, ArXiv, web scraper, YouTube transcripts. Use for information gathering, current events, factual lookups, academic research, reading web pages, analyzing videos.",
        "coder": "Python code execution. Use for writing programs, algorithms, games, data processing, automation scripts.",
        "analyst": "Financial data (stock prices, company info), calculator, web search. Use for stock analysis, numerical reasoning, financial comparisons, data interpretation.",
        "writer": "Pure language craft — no tools. Use for creating polished prose, blog posts, essays, reports, creative writing, transforming research into engaging content.",
        "planner": "Web search, calculator. Use for task decomposition, structured analysis, comparison matrices, pros/cons lists, decision trees, step-by-step action plans.",
        "fact_checker": "Tavily, DuckDuckGo, web scraper. Use for verifying claims, cross-referencing sources, detecting misinformation. ALWAYS depends_on another agent whose output needs verification.",
        "math_solver": "Calculator, Python REPL. Use for step-by-step mathematical problem solving, equations, calculus, algebra, with code verification.",
        "visualizer": "Python REPL with matplotlib. Use for creating charts, graphs, data visualizations. ALWAYS depends_on another agent whose data needs visualizing.",
        "critic": "Pure LLM — no tools. Use for reviewing and improving other agents' output quality. ALWAYS depends_on another agent whose work needs critique.",
    }

    def __init__(self, llm):
        self.llm = llm

    def plan(self, state: State) -> dict:
        """
        Creates a delegation plan: which specialist agents to dispatch and what tasks to give each.
        Supports depends_on for sequential agent chains.
        If auto_approve is False, pauses for human approval via interrupt().

        On phase continuation (delegation_plan already exists, some agents completed),
        skips re-planning and returns the existing plan for dispatch routing.
        """
        # Phase continuation: if plan already exists and some agents completed,
        # just return existing plan so dispatch_to_specialists handles next batch
        existing_plan = state.get("delegation_plan") or []
        completed = {r["agent"] for r in (state.get("agent_results") or [])}
        if existing_plan and completed:
            print(f"[supervisor] Phase continuation — {len(completed)} agents done, continuing plan")
            return {
                "delegation_plan": existing_plan,
            }

        # Get the most recent user request
        user_request = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                user_request = msg.content
                break

        agents_desc = "\n".join(
            f"- {name}: {desc}" for name, desc in self.AVAILABLE_AGENTS.items()
        )

        prompt = f"""You are a supervisor agent that delegates work to specialist agents.
Analyze this request and create a delegation plan.

Available specialists:
{agents_desc}

User request: {user_request}

Rules:
- Only include agents that are genuinely needed for this request
- Each agent should get a specific, actionable task
- Include at least 1 agent, at most 3
- Use "depends_on" to create sequential chains when one agent needs another's output
- Agents with empty depends_on run in parallel (Phase 0)
- Agents with depends_on wait until those agents finish (Phase 1+)
- fact_checker, visualizer, and critic MUST have depends_on (they process other agents' output)
- writer CAN have depends_on if it needs research/data to write about
- If the request is simple (just chat/info), use only researcher
- If code is needed, include coder
- If financial/numerical analysis is needed, include analyst
- If the user wants polished content/writing, include writer (with depends_on researcher if research needed)
- If mathematical reasoning is needed, include math_solver
- If the user asks for charts/graphs, include visualizer (with depends_on the data-providing agent)
- If the user wants fact-checking, include fact_checker (with depends_on the agent to verify)
- If the user wants critique/review, include critic (with depends_on the agent to review)

Respond with ONLY a valid JSON array. No explanation, no markdown, no code fences.
Example: [{{"agent": "researcher", "task": "Research AI trends in 2024", "depends_on": []}}, {{"agent": "writer", "task": "Write a blog post using the research findings", "depends_on": ["researcher"]}}]
Example: [{{"agent": "analyst", "task": "Fetch NVIDIA and AMD stock data and compare", "depends_on": []}}, {{"agent": "visualizer", "task": "Create a comparison chart of the stock data", "depends_on": ["analyst"]}}]
Example: [{{"agent": "coder", "task": "Write a fizzbuzz implementation", "depends_on": []}}, {{"agent": "critic", "task": "Review the code for quality and edge cases", "depends_on": ["coder"]}}]"""

        print(f"[supervisor] Planning for: {user_request[:80]}...")
        response, model_note = invoke_with_fallback(
            self.llm,
            [HumanMessage(content=prompt)],
            label="supervisor-plan",
        )
        if model_note:
            print(f"[supervisor] Plan used fallback: {model_note}")

        try:
            # Strip markdown code fences if present
            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1] if "\n" in content else content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            plan = json.loads(content)
            if not isinstance(plan, list) or len(plan) == 0:
                raise ValueError("Invalid plan format")
            # Validate agent types and ensure depends_on field exists
            validated = []
            for p in plan:
                if isinstance(p, dict) and p.get("agent") in self.AVAILABLE_AGENTS:
                    if "depends_on" not in p:
                        p["depends_on"] = []
                    # Validate depends_on references
                    valid_agents = {pp.get("agent") for pp in plan if isinstance(pp, dict)}
                    p["depends_on"] = [d for d in p["depends_on"] if d in valid_agents]
                    validated.append(p)
            plan = validated
            if not plan:
                raise ValueError("No valid agents in plan")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"[supervisor] Plan parsing failed ({e}), using researcher fallback")
            # Fallback: single researcher agent with the full request
            plan = [{"agent": "researcher", "task": user_request, "depends_on": []}]

        agent_names = [p["agent"] for p in plan]
        print(f"[supervisor] Plan: {agent_names}")

        # Human-in-the-loop: pause for approval if auto_approve is off
        if not state.get("auto_approve", True) and interrupt is not None:
            approval = interrupt({
                "type": "plan_approval",
                "delegation_plan": plan,
                "message": "Review the delegation plan before agents execute.",
            })
            if isinstance(approval, dict) and approval.get("action") == "reject":
                return {
                    "messages": [AIMessage(content="Delegation plan was rejected. Please submit a new request or modify your query.")],
                    "delegation_plan": [],
                    "agent_results": [],
                }

        return {
            "delegation_plan": plan,
            "agent_results": [],
        }

    def synthesize(self, state: State) -> dict:
        """
        Combines results from all specialist agents into a unified response.
        Uses confidence scores to prioritize high-quality results.
        Handles partial failures gracefully.
        """
        results = state.get("agent_results") or []
        if not results:
            return {
                "messages": [AIMessage(content="No agent results to synthesize.")],
            }

        # Check if more phases remain (some agents still pending)
        plan = state.get("delegation_plan") or []
        completed = {r["agent"] for r in results}
        remaining = [t for t in plan if t["agent"] not in completed]
        if remaining:
            # More phases to go — don't synthesize yet, route back
            print(f"[supervisor] {len(remaining)} agents remaining — routing to next phase")
            return {
                "current_phase": (state.get("current_phase") or 0) + 1,
            }

        user_request = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                user_request = msg.content
                break

        # Sort results by confidence (highest first)
        sorted_results = sorted(results, key=lambda r: r.get("confidence", 0.5), reverse=True)

        # Build results summary for the LLM
        results_text = ""
        for r in sorted_results:
            agent_name = r.get("agent", "unknown").title()
            task = r.get("task", "")
            result = r.get("result", "")
            tools = r.get("tools_used", [])
            confidence = r.get("confidence", 0.5)
            exec_time = r.get("execution_time", 0)
            model_note = r.get("model_note", "")
            tools_str = f" (used: {', '.join(tools)})" if tools else ""

            confidence_note = ""
            if confidence <= 0.3:
                confidence_note = " [NOTE: Low confidence - treat with caution]"
            elif confidence == 0.0:
                confidence_note = " [WARNING: Agent failed or produced no usable output]"

            model_info = f" [{model_note}]" if model_note else ""

            results_text += (
                f"\n\n### {agent_name} Agent{tools_str} (confidence: {confidence}{confidence_note}, {exec_time}s){model_info}"
                f"\nTask: {task}\nResult:\n{result}"
            )

        # Check for any failed agents
        failed_agents = [r for r in results if r.get("confidence", 0.5) == 0.0]
        failure_note = ""
        if failed_agents:
            names = ", ".join(r.get("agent", "unknown").title() for r in failed_agents)
            failure_note = f"\n\nNote: The following agent(s) were unable to complete their tasks: {names}. Synthesize the best response from the remaining agents' findings."

        prompt = f"""You are a supervisor synthesizing results from specialist agents into a final response.

Original user request: {user_request}

Agent results (sorted by confidence):{results_text}{failure_note}

Instructions:
- Combine all agent results into a single, comprehensive, well-structured response
- Address every part of the original request
- Prioritize information from high-confidence agents
- Maintain technical accuracy and cite sources where applicable
- Use clear formatting with headers, bullet points, or code blocks as appropriate
- If any agent reported an error or had low confidence, note what information may be incomplete
- If a chart/visualization was produced, refer to it in prose only. Do NOT include ASCII charts, pseudo-graphics, image placeholders, or fenced text-art blocks.
- Do NOT repeat raw plotting code, execution logs, or code-like chart summaries in the final answer.
- Do NOT mention the internal agent delegation process — respond as if you gathered all information yourself"""

        print(f"[supervisor] Synthesizing {len(results)} agent results...")
        response, model_note = invoke_with_fallback(
            self.llm,
            [HumanMessage(content=prompt)],
            label="supervisor-synthesize",
        )
        if model_note:
            print(f"[supervisor] Synthesis used fallback: {model_note}")

        return {
            "messages": [response],
        }
