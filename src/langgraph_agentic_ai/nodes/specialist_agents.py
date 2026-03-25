import ast
import glob
import os
import re
import time
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from src.langgraph_agentic_ai.state.state import State
from src.langgraph_agentic_ai.tools.tool_registry import get_specialist_tools
from src.langgraph_agentic_ai.utils.llm_fallback import (
    invoke_with_fallback,
    is_rate_limit_error,
    _get_groq_fallbacks,
)


# ── Confidence scoring suffix appended to every specialist prompt ────────
CONFIDENCE_SUFFIX = """

At the end of your response, on a new line, add: [CONFIDENCE: X.X] where X.X is 0.0-1.0.
- 0.9-1.0: Highly confident, verified with tools/sources
- 0.7-0.8: Fairly confident, based on strong reasoning
- 0.4-0.6: Moderate confidence, some uncertainty
- 0.1-0.3: Low confidence, speculative"""


# ── Specialist system prompts ─────────────────────────────────────────────

SPECIALIST_PROMPTS = {
    "researcher": """You are a Research Specialist agent. Your expertise is information gathering and synthesis.

Process:
1. THINK about what information is needed
2. Use your tools (web search, DuckDuckGo, Wikipedia, ArXiv, web scraper, YouTube transcripts) to gather comprehensive data
3. Cross-reference multiple sources for accuracy
4. Synthesize findings into a clear, well-sourced response

Always cite your sources with URLs when available. Be thorough but concise.""",

    "coder": """You are a Coding Specialist agent. Your expertise is writing and executing Python code.

Process:
1. THINK about the requirements and approach
2. Write clean, well-commented Python code
3. Execute the code using the Python REPL tool
4. Verify the output and fix any errors
5. Present the code and results clearly

Always use print() to display results. Handle errors gracefully. Include comments explaining your logic.""",

    "analyst": """You are an Analysis Specialist agent. Your expertise is financial data analysis and numerical reasoning.

Process:
1. THINK about what data and calculations are needed
2. Use finance tools to fetch real-time stock/market data
3. Use the calculator for precise mathematical operations
4. Use web search for supplementary financial news if needed
5. Present analysis with clear numbers, trends, and insights

IMPORTANT: Always present key data in a clear, structured format so other agents can use it easily.
For stock comparisons, always include a summary like:
- Company A: $X.XX (with key metrics)
- Company B: $Y.YY (with key metrics)

Always provide specific numbers and cite data sources. Be precise with financial figures.""",

    "writer": """You are a Writing Specialist agent. Your expertise is creating well-structured, engaging content.

Process:
1. THINK about the audience, tone, and format needed
2. If given research data or analysis from other agents, transform it into polished prose
3. Structure content with clear headings, sections, and flow
4. Use markdown formatting: headers, bullet points, bold, emphasis

You have NO tools — your power is pure language craft. Create content that is engaging, informative, and well-organized.""",

    "planner": """You are a Planning & Strategy Specialist agent. Your expertise is task decomposition and structured analysis.

Process:
1. THINK about the problem structure and decision factors
2. Research options using web search if needed
3. Create structured frameworks: comparison matrices, pros/cons lists, decision trees, step-by-step action plans
4. Use calculator for any quantitative comparisons
5. Provide clear recommendations with reasoning

Always organize output with clear structure. Use tables or lists for comparisons.""",

    "fact_checker": """You are a Fact-Checking Specialist agent. Your expertise is verifying claims and detecting misinformation.

Process:
1. Extract specific factual claims from the provided text
2. For EACH claim, search multiple sources (Tavily, DuckDuckGo, web scraper) to verify
3. Cross-reference at least 2 sources per major claim
4. Report findings using this format for each claim:
   - VERIFIED (with source URL) — if confirmed by reliable sources
   - DISPUTED (with counter-evidence) — if contradicted by sources
   - UNVERIFIED (no sources found) — if unable to confirm or deny

Be thorough and skeptical. Prioritize authoritative sources.""",

    "math_solver": """You are a Mathematics Specialist agent. Your expertise is step-by-step mathematical problem solving.

Process:
1. IDENTIFY the problem type and required approach
2. SOLVE step-by-step, showing ALL reasoning at each step
3. VERIFY your answer by executing Python code using the REPL
4. PRESENT the solution clearly:
   - Problem statement
   - Approach/method
   - Step-by-step solution
   - Code verification
   - Final answer

Always verify analytically derived answers with code. Use the calculator for individual calculations.""",

    "visualizer": """You are a Data Visualization Specialist agent. Your expertise is creating informative charts and graphs.

CRITICAL: You will receive data from another agent above your task. You MUST parse the ACTUAL numbers and labels from that data and use them in your chart. NEVER use placeholder/dummy/example data like "Stock A" or random numbers.

You do NOT have tools. Instead, write your complete Python code inside a ```python code block.
The code will be extracted and executed automatically.

Process:
1. READ the data provided above your task carefully. Extract the exact numbers, labels, and categories.
2. DETERMINE the best chart type (bar for comparisons, line for trends, pie for proportions, etc.)
3. Write a single, complete Python code block using matplotlib with the REAL data from step 1
4. The code MUST:
   - import matplotlib and set: matplotlib.use('Agg')
   - Prefer ONE simple figure (e.g. plt.figure(figsize=(9, 5))) and one primary chart (plt.bar or plt.plot). Avoid deeply nested parentheses and multi-subplot layouts unless necessary.
   - Syntax: double-check every opening '(' has a matching ')' before finishing. Invalid Python will produce no image.
   - Include: descriptive title, axis labels, legend when needed, clean formatting, appropriate colors
   - Save the chart: plt.savefig('/tmp/reasonflow_chart.png', dpi=150, bbox_inches='tight')
   - Call plt.close() after saving
   - Print a confirmation message like "Chart saved successfully"
5. After the code block, describe what the visualization shows and key insights from the ACTUAL data

Example: If given "NVIDIA: $175, AMD: $205", your code should be:
```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

stocks = ['NVIDIA', 'AMD']
prices = [175, 205]
colors = ['#76b900', '#ED1C24']

plt.figure(figsize=(8, 5))
plt.bar(stocks, prices, color=colors)
plt.title('NVIDIA vs AMD Stock Price Comparison')
plt.ylabel('Price (USD)')
plt.savefig('/tmp/reasonflow_chart.png', dpi=150, bbox_inches='tight')
plt.close()
print('Chart saved successfully')
```""",

    "critic": """You are a Quality Critic Specialist agent. Your expertise is reviewing and improving other agents' output.

Process:
1. Carefully READ the provided output from another agent
2. EVALUATE on these dimensions:
   - Completeness: Does it fully address the task?
   - Clarity: Is it well-structured and easy to understand?
   - Accuracy: Are claims and code correct?
   - Edge cases: Are important exceptions handled?
3. Rate overall quality (1-10) with justification
4. Provide SPECIFIC, ACTIONABLE improvement suggestions
5. If code: check for bugs, missing error handling, edge cases

Be constructive but thorough. Focus on what would make the output significantly better.""",
}

MAX_TOOL_ITERATIONS = 6   # Safety limit for ReAct loops
AGENT_TIMEOUT_SECS = 90   # Max execution time per agent


class SpecialistAgents:
    """
    Factory that runs specialist agents.
    Each specialist has its own tools and system prompt, and runs
    an internal ReAct loop (THINK -> ACT -> OBSERVE -> REPEAT -> ANSWER).

    9 agents: researcher, coder, analyst, writer, planner,
              fact_checker, math_solver, visualizer, critic

    Includes automatic model fallback: when the primary Groq model
    hits rate limits (429), tries smaller fallback models automatically.
    """

    def __init__(self, llm):
        self.llm = llm

    def run(self, state: State) -> dict:
        """
        Runs a specialist agent based on the current_agent and agent_task in state.
        Returns results in agent_results list for the reducer to merge.
        """
        agent_type = state.get("current_agent", "researcher")
        task = state.get("agent_task", "")
        start_time = time.time()

        if not task:
            return {
                "agent_results": [{
                    "agent": agent_type,
                    "result": "No task was assigned.",
                    "task": task,
                    "tools_used": [],
                    "confidence": 0.0,
                    "execution_time": 0.0,
                    "token_estimate": 0,
                    "model_note": "",
                }],
            }

        tools = get_specialist_tools(agent_type)
        system_prompt = SPECIALIST_PROMPTS.get(agent_type, SPECIALIST_PROMPTS["researcher"])
        system_prompt += CONFIDENCE_SUFFIX

        model_note = ""
        try:
            result, tools_used, model_note = self._react_loop(
                tools, system_prompt, task, agent_type
            )
            confidence = self._extract_confidence(result)
            # Visualizer: extract, execute, validate, and clean chart output
            if agent_type == "visualizer" and not tools_used:
                viz_meta = self._process_visualizer_output(task, result)
                result = viz_meta["result"]
                if viz_meta["tools_used"]:
                    tools_used = viz_meta["tools_used"]
                confidence = (
                    viz_meta["confidence"]
                    if confidence == 0.5
                    else min(confidence, viz_meta["confidence"])
                )
        except Exception as e:
            result = f"Agent encountered an issue: {self._friendly_error(e)}"
            tools_used = []
            confidence = 0.0

        execution_time = round(time.time() - start_time, 1)
        result = self._validate_result(result, agent_type)
        token_estimate = self._estimate_tokens(system_prompt + task + result)

        return {
            "agent_results": [{
                "agent": agent_type,
                "result": result,
                "task": task,
                "tools_used": tools_used,
                "confidence": confidence,
                "execution_time": execution_time,
                "token_estimate": token_estimate,
                "model_note": model_note,
            }],
        }

    def _react_loop(self, tools, system_prompt, task, agent_type="specialist"):
        """
        Internal ReAct loop for a specialist agent.
        Returns (final_response, list_of_tools_used, model_note).
        Respects AGENT_TIMEOUT_SECS and MAX_TOOL_ITERATIONS.
        Automatically falls back to smaller Groq models on rate limit.
        """
        start_time = time.time()
        label = agent_type
        model_note = ""

        # ── Tool-less agents (writer, critic) — single invocation ────────
        if not tools:
            print(f"[{label}] Starting (no tools)...")
            response, note = invoke_with_fallback(
                self.llm,
                [SystemMessage(content=system_prompt), HumanMessage(content=task)],
                label=label,
            )
            result = response.content if response.content else "No response generated."
            if note:
                print(f"[{label}] {note}")
            return result, [], note

        # ── Tool-equipped agents — ReAct loop ────────────────────────────
        print(f"[{label}] Starting with {len(tools)} tools...")
        llm_with_tools = self.llm.bind_tools(tools)
        tool_map = {t.name: t for t in tools}
        tools_used = []

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=task),
        ]

        last_response = None
        for iteration in range(MAX_TOOL_ITERATIONS):
            # Check timeout
            if time.time() - start_time > AGENT_TIMEOUT_SECS:
                partial = last_response.content if last_response and last_response.content else ""
                return (
                    f"Task timed out after {AGENT_TIMEOUT_SECS}s. Partial results: {partial}",
                    tools_used,
                    model_note,
                )

            # ── Invoke LLM with retry + fallback ─────────────────────
            response = None
            # Try primary model first (3 retries with backoff)
            for attempt in range(3):
                try:
                    response = llm_with_tools.invoke(messages)
                    break
                except Exception as e:
                    if is_rate_limit_error(e):
                        wait = 3 * (attempt + 1)
                        print(f"[{label}] Rate limited, waiting {wait}s (attempt {attempt+1}/3)")
                        time.sleep(wait)
                    else:
                        raise

            # Primary model failed — try fallback models
            if response is None:
                for fb_llm, fb_name in _get_groq_fallbacks():
                    try:
                        print(f"[{label}] Switching to fallback model: {fb_name}")
                        fb_bound = fb_llm.bind_tools(tools)
                        response = fb_bound.invoke(messages)
                        # Switch to fallback for all remaining iterations
                        llm_with_tools = fb_bound
                        model_note = f"Switched to {fb_name} (rate limit on primary model)"
                        print(f"[{label}] Fallback {fb_name} succeeded")
                        break
                    except Exception as fb_e:
                        if is_rate_limit_error(fb_e):
                            print(f"[{label}] Fallback {fb_name} also rate limited")
                            time.sleep(3)
                            continue
                        raise

            if response is None:
                return (
                    "Rate limit exceeded on all available models. "
                    "Please try again in a moment or switch to OpenAI in the sidebar.",
                    tools_used,
                    model_note or "All models rate limited",
                )

            messages.append(response)
            last_response = response

            # If no tool calls, the agent is done reasoning
            if not response.tool_calls:
                print(f"[{label}] Done after {iteration+1} iteration(s)")
                break

            # Execute all tool calls
            for tc in response.tool_calls:
                tc_name = tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
                tc_args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
                tc_id = tc.get("id", "") if isinstance(tc, dict) else getattr(tc, "id", "")

                tool = tool_map.get(tc_name)
                if tool:
                    if tc_name not in tools_used:
                        tools_used.append(tc_name)
                    try:
                        print(f"[{label}] Using tool: {tc_name}")
                        result = tool.invoke(tc_args)
                    except Exception as e:
                        result = f"Tool error ({tc_name}): {str(e)[:300]}"
                    messages.append(
                        ToolMessage(content=str(result)[:3000], tool_call_id=tc_id)
                    )
                else:
                    messages.append(
                        ToolMessage(
                            content=f"Unknown tool: {tc_name}",
                            tool_call_id=tc_id or "unknown",
                        )
                    )
        else:
            # Loop completed without break — max iterations reached
            print(f"[{label}] Hit max iterations ({MAX_TOOL_ITERATIONS})")
            if last_response and last_response.tool_calls:
                try:
                    summary, note = invoke_with_fallback(
                        self.llm, messages, label=f"{label}-summary"
                    )
                    if note:
                        model_note = model_note or note
                    if summary.content:
                        return summary.content, tools_used, model_note
                except Exception:
                    pass

        # Extract final answer
        if last_response and hasattr(last_response, "content") and last_response.content:
            return last_response.content, tools_used, model_note
        return "No response generated by specialist.", tools_used, model_note

    @staticmethod
    def _extract_code_block(text: str) -> str:
        """Extract the largest Python code block from a response."""
        code_blocks = re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if not code_blocks:
            code_blocks = re.findall(r"```\s*\n(.*?)```", text, re.DOTALL)
        if not code_blocks:
            return ""
        return max(code_blocks, key=len)

    @staticmethod
    def _execute_code(code: str) -> str:
        """Execute extracted Python and capture stdout/stderr-friendly output."""
        import sys
        import io

        old_stdout = sys.stdout
        redirected_output = io.StringIO()
        sys.stdout = redirected_output
        try:
            exec(code, {"__builtins__": __builtins__})
            output = redirected_output.getvalue()
            print(f"[visualizer] Code executed successfully")
            return output if output else "Code executed successfully."
        except Exception as e:
            print(f"[visualizer] Code execution error: {e}")
            return f"Code execution error: {type(e).__name__}: {str(e)[:200]}"
        finally:
            sys.stdout = old_stdout

    @staticmethod
    def _strip_code_blocks(text: str) -> str:
        """Remove Python code blocks and confidence suffixes from specialist prose."""
        cleaned = re.sub(r"```python\s*\n.*?```", "", text, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r"```\s*\n.*?```", "", cleaned, flags=re.DOTALL)
        cleaned = re.sub(r"\n?\[CONFIDENCE:\s*[\d.]+\]\s*$", "", cleaned.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        return cleaned

    @staticmethod
    def _latest_chart_path(since_ts: float) -> str:
        """Return the newest generated chart path after a given timestamp."""
        chart_paths = glob.glob("/tmp/reasonflow_chart*.png") + glob.glob("/tmp/reasonflow_chart*.jpg")
        fresh = [p for p in chart_paths if os.path.isfile(p) and os.path.getmtime(p) >= since_ts]
        if not fresh:
            return ""
        return max(fresh, key=os.path.getmtime)

    @staticmethod
    def _extract_reference_points(text: str) -> dict:
        """Extract simple label->value pairs from upstream analyst text when possible."""
        points = {}
        patterns = [
            r"(?im)^\s*[-*]?\s*([A-Z]{2,10})[^:\n]{0,40}:\s*\$?\s*([0-9]+(?:\.[0-9]+)?)",
            r"(?im)\b([A-Z]{2,10})\b[^\n]{0,40}?current price[^\d\n]{0,20}\$?\s*([0-9]+(?:\.[0-9]+)?)",
        ]
        for pattern in patterns:
            for label, value in re.findall(pattern, text):
                try:
                    points[label.strip().upper()] = float(value)
                except ValueError:
                    continue
        return points

    @staticmethod
    def _extract_chart_points_from_code(code: str, reference_points=None) -> dict:
        """Extract likely chart labels and plotted numeric values from simple list assignments."""
        reference_points = reference_points or {}
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return {}

        string_lists = []
        numeric_lists = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            value = node.value
            if not isinstance(value, (ast.List, ast.Tuple)) or not value.elts:
                continue

            if all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in value.elts):
                string_lists.append([str(e.value).strip() for e in value.elts])
            elif all(isinstance(e, ast.Constant) and isinstance(e.value, (int, float)) for e in value.elts):
                numeric_lists.append([float(e.value) for e in value.elts])

        best = {}
        best_overlap = -1
        ref_keys = {k.upper() for k in reference_points.keys()}
        for labels in string_lists:
            for values in numeric_lists:
                if len(labels) != len(values) or len(labels) < 2:
                    continue
                candidate = {label.strip().upper(): val for label, val in zip(labels, values) if label.strip()}
                overlap = len(ref_keys.intersection(candidate.keys())) if ref_keys else len(candidate)
                if overlap > best_overlap:
                    best = candidate
                    best_overlap = overlap
        return best

    @classmethod
    def _validate_chart_against_task(cls, task: str, code: str) -> tuple[str, str]:
        """Compare plotted label/value pairs against upstream data when available."""
        reference_points = cls._extract_reference_points(task)
        if not reference_points:
            return "unknown", "Automatic chart-data validation was inconclusive (no parsable upstream values found)."

        chart_points = cls._extract_chart_points_from_code(code, reference_points)
        if not chart_points:
            return "unknown", "Automatic chart-data validation was inconclusive (unable to parse plotted values)."

        overlap = sorted(set(reference_points).intersection(chart_points))
        if len(overlap) < 2:
            return "unknown", "Automatic chart-data validation was inconclusive (insufficient label overlap)."

        mismatches = []
        for label in overlap:
            ref = reference_points[label]
            got = chart_points[label]
            tolerance = max(0.5, abs(ref) * 0.02)
            if abs(ref - got) > tolerance:
                mismatches.append(f"{label}: expected {ref:.2f}, plotted {got:.2f}")

        if mismatches:
            return "mismatch", "Chart values did not match upstream data: " + "; ".join(mismatches[:4])

        return "ok", "Chart values matched the upstream data for " + ", ".join(overlap)

    @classmethod
    def _process_visualizer_output(cls, task: str, text: str) -> dict:
        """Execute and validate visualizer code, then return clean prose for downstream agents."""
        cleaned = cls._strip_code_blocks(text)
        code = cls._extract_code_block(text)
        if not code:
            return {
                "result": (cleaned + "\n\nValidation: No Python chart code block was found.").strip(),
                "tools_used": [],
                "confidence": 0.0,
            }

        started = time.time() - 0.01
        print(f"[visualizer] Executing extracted code ({len(code)} chars)...")
        exec_output = cls._execute_code(code)
        chart_path = cls._latest_chart_path(started)

        if exec_output.startswith("Code execution error:") or not chart_path:
            failure_note = exec_output if exec_output.startswith("Code execution error:") else "Chart file was not created."
            return {
                "result": (cleaned + f"\n\nValidation: Chart generation failed. {failure_note}").strip(),
                "tools_used": [],
                "confidence": 0.0,
            }

        status, validation_note = cls._validate_chart_against_task(task, code)
        confidence = 0.9 if status == "ok" else 0.6 if status == "unknown" else 0.0
        return {
            "result": (cleaned + f"\n\nValidation: {validation_note}").strip(),
            "tools_used": ["matplotlib"],
            "confidence": confidence,
        }

    @staticmethod
    def _extract_confidence(text: str) -> float:
        """Parse [CONFIDENCE: X.X] from agent response. Fallback: 0.5."""
        if not text:
            return 0.5
        match = re.search(r"\[CONFIDENCE:\s*([\d.]+)\]", text)
        if match:
            try:
                val = float(match.group(1))
                return max(0.0, min(1.0, val))
            except ValueError:
                pass
        return 0.5

    @staticmethod
    def _friendly_error(e: Exception) -> str:
        """Convert raw exceptions into user-friendly error messages."""
        msg = str(e)
        if "429" in msg or "rate_limit" in msg.lower():
            return (
                "Rate limited by the LLM provider. The system tried fallback models "
                "but all are currently limited. Try again shortly or switch to OpenAI."
            )
        if "timeout" in msg.lower():
            return "Request timed out. The task may be too complex for the current model."
        if "api_key" in msg.lower() or "authentication" in msg.lower():
            return "API key issue. Please check your API key in the sidebar."
        return f"Unexpected error: {msg[:200]}"

    @staticmethod
    def _validate_result(result: str, agent_type: str) -> str:
        """Ensure agent output is usable before synthesis."""
        if not result or len(result.strip()) < 10:
            return "Agent produced insufficient output for this task."
        result = re.sub(r"\n?\[CONFIDENCE:\s*[\d.]+\]\s*$", "", result.strip(), flags=re.IGNORECASE)
        # Strip raw tracebacks from non-code agents
        if agent_type not in ("coder", "math_solver", "visualizer"):
            if "Traceback (most recent call last)" in result and "Error" in result:
                return "Agent encountered a code error. " + result[:300]
        # Truncate extremely long outputs
        if len(result) > 5000:
            return result[:5000] + "\n\n[Output truncated at 5000 chars]"
        return result

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough estimate: ~4 chars per token for English."""
        return len(text) // 4 if text else 0
