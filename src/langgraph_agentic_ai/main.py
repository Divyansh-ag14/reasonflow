import os
import uuid
import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver

# Load .env file (if it exists) so API keys are auto-detected
load_dotenv()
from src.langgraph_agentic_ai.ui.streamlit_ui.load_ui import (
    LoadStreamlitUI,
    ARCHITECTURE_LAYERS,
)
from src.langgraph_agentic_ai.LLMS.groqllm import GroqLLM
from src.langgraph_agentic_ai.LLMS.openaillm import OpenAILLM
from src.langgraph_agentic_ai.graph.graph_builder import GraphBuilder
from src.langgraph_agentic_ai.ui.streamlit_ui.display_result import DisplayResultStreamlit


# ── Welcome: ReasonFlow Agent (default) ──────────────────────────────────
def _show_agent_welcome():
    st.markdown(
        '<div class="welcome-hero">'
        "<h1>ReasonFlow</h1>"
        '<p class="tagline">A <strong>multi-agent AI system</strong> where a Supervisor '
        "<strong>plans</strong>, dispatches <strong>specialist agents</strong> in parallel, "
        "and <strong>reflects</strong> on quality — with optional human approval gates.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Capability cards — updated for multi-agent
    capabilities = [
        ("🔀", "Smart Router", "Instant replies for casual chat, full pipeline for complex tasks"),
        ("📋", "Supervisor", "Plans tasks, delegates to specialists with sequential chains"),
        ("⚡", "9 Agents", "Researcher · Coder · Analyst · Writer · Planner · Fact-Checker · Math · Viz · Critic"),
        ("🔍", "Reflect + HITL", "Quality evaluation with retries, optional human approval"),
    ]
    cards = '<div class="feature-grid">'
    for icon, title, desc in capabilities:
        cards += (
            f'<div class="feature-card">'
            f'<div class="f-icon">{icon}</div>'
            f'<div class="f-title">{title}</div>'
            f'<div class="f-desc">{desc}</div>'
            f"</div>"
        )
    cards += "</div>"
    st.markdown(cards, unsafe_allow_html=True)

    st.markdown(
        '<div class="prompt-hint">'
        "💡 <strong>Try:</strong> "
        '"Research AI trends and write a blog post" · '
        '"Compare NVIDIA and AMD stocks and visualize the data" · '
        '"Solve the integral of x\u00b2\u00b7e\u02e3 step by step"'
        "</div>",
        unsafe_allow_html=True,
    )


# ── Welcome: Architecture Demos ──────────────────────────────────────────
def _show_architecture_welcome(selected_usecase):
    st.markdown(
        '<div class="welcome-hero">'
        "<h1>Architecture Demos</h1>"
        '<p class="tagline">See how ReasonFlow was built — layer by layer.<br>'
        "Each step adds a new capability on top of the previous one.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Find which layer is active
    active_idx = 0
    for i, layer in enumerate(ARCHITECTURE_LAYERS):
        if layer["usecase"] == selected_usecase:
            active_idx = i
            break

    # Build the evolution timeline
    timeline = '<div class="evo-timeline">'
    for i, layer in enumerate(ARCHITECTURE_LAYERS):
        if i < active_idx:
            step_cls = "completed"
            badge_cls = "completed"
            badge_content = "✓"
        elif i == active_idx:
            step_cls = "active"
            badge_cls = "active"
            badge_content = str(layer["num"])
        else:
            step_cls = ""
            badge_cls = "upcoming"
            badge_content = str(layer["num"])

        # Final layer (ReasonFlow Agent) shown differently
        is_final = i == 3
        opacity = "opacity:0.4;" if is_final and active_idx < 3 else ""
        label_extra = ' <span style="color:#6C63FF;font-size:0.72rem;">(Full Agent)</span>' if is_final else ""

        timeline += (
            f'<div class="evo-step {step_cls}" style="{opacity}">'
            f'<div class="evo-badge {badge_cls}">{badge_content}</div>'
            f'<div class="evo-body">'
            f'<div class="evo-title">{layer["title"]}{label_extra}</div>'
            f'<div class="evo-subtitle">{layer["subtitle"]}</div>'
            f'<div class="evo-added">{layer["added"]}</div>'
            f'<div class="evo-graph-label">{layer["graph"]}</div>'
            f"</div></div>"
        )

        # Connector between steps
        if i < len(ARCHITECTURE_LAYERS) - 1:
            conn_cls = "done" if i < active_idx else ""
            timeline += f'<div class="evo-connector {conn_cls}"></div>'

    timeline += "</div>"
    st.markdown(timeline, unsafe_allow_html=True)

    # Prompt hint
    hints = {
        "Basic Chatbot": '"Hi! Tell me about yourself" then ask "What did I just say?"',
        "Chatbot With Web": '"What is NVIDIA\'s stock price and latest AI news?"',
        "AI News": "Select a timeframe in the sidebar then click Fetch AI News",
    }
    hint = hints.get(selected_usecase, "")
    if hint:
        layer_name = ARCHITECTURE_LAYERS[active_idx]["title"]
        st.markdown(
            f'<div class="prompt-hint">'
            f"💡 <strong>Try {layer_name}:</strong> {hint}"
            f"</div>",
            unsafe_allow_html=True,
        )


# ── Footer ──────────────────────────────────────────────────────────────
def _render_footer():
    st.markdown(
        '<div class="app-footer">'
        'Built with <a href="https://github.com/langchain-ai/langgraph">LangGraph</a> · '
        '<a href="https://groq.com">Groq</a> · '
        '<a href="https://streamlit.io">Streamlit</a>'
        "</div>",
        unsafe_allow_html=True,
    )


# ── Helper: build graph + config ────────────────────────────────────────
def _build_graph(user_input, selected_usecase):
    """Creates the LLM model, builds the graph, and returns (graph, config)."""
    selected_llm = user_input.get("selected_llm", "Groq")
    if selected_llm == "OpenAI":
        obj_llm_config = OpenAILLM(user_contols_input=user_input)
    else:
        obj_llm_config = GroqLLM(user_contols_input=user_input)
    model = obj_llm_config.get_llm_model()
    if not model:
        return None, None

    graph_builder = GraphBuilder(model, st.session_state.checkpointer)
    graph = graph_builder.setup_graph(selected_usecase)
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    return graph, config


# ── Main app ─────────────────────────────────────────────────────────────
def load_langgraph_agentic_ai_app():
    """Loads and runs the ReasonFlow application."""

    # Session state init
    if "checkpointer" not in st.session_state:
        st.session_state.checkpointer = MemorySaver()
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())

    if st.session_state.get("new_conversation"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.new_conversation = False
        st.session_state.pop("hitl_pending", None)
        st.session_state.pop("hitl_resume", None)
        st.session_state.pop("hitl_action", None)
        st.session_state.pop("hitl_plan", None)
        st.session_state.pop("hitl_user_message", None)
        st.session_state.pop("processing", None)
        st.session_state.pop("pending_query", None)
        st.session_state.pop("pending_auto_approve", None)
        # Clear all per-usecase histories
        for key in list(st.session_state.keys()):
            if key.startswith("chat_history_"):
                st.session_state[key] = []

    # Load sidebar UI
    ui = LoadStreamlitUI()
    user_input = ui.load_streamlit_ui()

    if not user_input:
        st.error("Error: Failed to load user input from the UI.")
        return

    selected_usecase = user_input.get("selected_usecase", "")

    if selected_usecase == "ReasonFlow Agent":
        print(
            "[main] state snapshot | "
            f"processing={st.session_state.get('processing', False)} | "
            f"pending_query={bool(st.session_state.get('pending_query'))} | "
            f"hitl_pending={st.session_state.get('hitl_pending')} | "
            f"hitl_resume={st.session_state.get('hitl_resume')} | "
            f"hitl_action={st.session_state.get('hitl_action')} | "
            f"chat_history={len(st.session_state.get('chat_history', []))}"
        )

    # ── Per-usecase chat history ──────────────────────────────────────
    history_key = f"chat_history_{selected_usecase.replace(' ', '_')}"
    if history_key not in st.session_state:
        st.session_state[history_key] = []
    st.session_state.chat_history = st.session_state[history_key]

    # ── HITL Resume Path ─────────────────────────────────────────────
    # If the user clicked Approve/Reject on a HITL gate, resume the graph.
    # Input stays locked (processing=True from the original submission).
    if st.session_state.get("hitl_pending") and (
        st.session_state.get("hitl_resume") or st.session_state.get("hitl_action")
    ):
        # Render disabled chat input so user can't type during resume
        st.chat_input("Agents working — please wait...", disabled=True)

        resume_val = st.session_state.pop("hitl_resume", None) or st.session_state.pop("hitl_action", None)
        print(f"[main] HITL resume triggered with payload: {resume_val}")
        try:
            graph, config = _build_graph(user_input, selected_usecase)
            if graph:
                if st.session_state.get("hitl_user_message"):
                    with st.chat_message("user"):
                        st.write(st.session_state.get("hitl_user_message"))
                print("[main] Rebuilt graph for HITL resume")
                DisplayResultStreamlit(
                    selected_usecase, graph, None, config
                ).display_hitl_resume(resume_val)
            else:
                st.error("Error: LLM model could not be initialized for resume.")
        except Exception as e:
            st.error(f"Error resuming: {e}")

        # Pipeline done — unlock input (re-enables on next interaction)
        st.session_state.processing = False
        st.session_state.pop("hitl_pending", None)
        st.session_state.pop("hitl_action", None)
        st.session_state.pop("hitl_plan", None)
        st.session_state.pop("hitl_user_message", None)
        st.session_state.pop("pending_query", None)
        st.rerun()

    # ── Normal Path ──────────────────────────────────────────────────

    is_processing = st.session_state.get("processing", False)
    pending_query = st.session_state.get("pending_query", None)

    # Chat input — physically disabled while agents are working
    if st.session_state.IsFetchButtonClicked:
        new_message = st.session_state.timeframe
    else:
        new_message = st.chat_input(
            "Agents working — please wait..." if is_processing else "Type your message...",
            disabled=is_processing,
        )

    # Show any error from a previous run that was swallowed by st.rerun()
    if st.session_state.get("_last_error"):
        st.error(st.session_state.pop("_last_error"))

    # Welcome screen (only when truly idle)
    if (
        not is_processing
        and not st.session_state.get("hitl_pending")
        and not new_message
        and not pending_query
        and not st.session_state.chat_history
    ):
        if selected_usecase == "ReasonFlow Agent":
            _show_agent_welcome()
        else:
            _show_architecture_welcome(selected_usecase)
        _render_footer()
        return

    # Processing but nothing currently renderable: don't fall back to welcome.
    if is_processing and not pending_query and not st.session_state.chat_history:
        st.info("Resuming agents... please wait.")
        _render_footer()
        return

    # ── New message submitted: queue it and rerun to lock input ────
    if new_message and not is_processing:
        st.session_state.pop("hitl_plan", None)
        st.session_state.pop("hitl_pending", None)
        st.session_state.pop("hitl_resume", None)
        st.session_state.pop("hitl_action", None)
        st.session_state.pop("hitl_user_message", None)
        st.session_state.pending_query = new_message
        st.session_state.processing = True
        # Snapshot auto_approve at submission time (prevents mid-execution toggle bugs)
        st.session_state.pending_auto_approve = st.session_state.get("auto_approve", True)
        # Fresh thread for each ReasonFlow query (prevents state bleed)
        if selected_usecase == "ReasonFlow Agent":
            st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()  # <-- reruns immediately; chat_input renders as disabled

    # ── HITL approval UI replay (no re-processing) ───────────────────
    if (
        selected_usecase == "ReasonFlow Agent"
        and is_processing
        and pending_query
        and st.session_state.get("hitl_plan")
        and not st.session_state.get("hitl_pending")
    ):
        print("[main] Rendering saved HITL approval UI")
        DisplayResultStreamlit.render_hitl_plan_approval(
            st.session_state.get("hitl_user_message", pending_query),
            st.session_state.get("hitl_plan", []),
        )
        _render_footer()
        return

    # ── Process queued query (input is now locked) ────────────────
    if pending_query and is_processing:
        error_msg = None
        try:
            graph, config = _build_graph(user_input, selected_usecase)

            if not graph:
                error_msg = "Error: LLM model could not be initialized."
                st.session_state.processing = False
                st.session_state.pop("pending_query", None)
                st.session_state["_last_error"] = error_msg
                st.error(error_msg)
                return

            try:
                print(f"[main] Processing query: {pending_query[:60]}...")
                DisplayResultStreamlit(
                    selected_usecase, graph, pending_query, config
                ).display_result_on_ui()
                print(f"[main] Pipeline completed successfully")
            except Exception as e:
                error_msg = f"Error: Graph execution failed — {e}"
                print(f"[main] {error_msg}")
                st.session_state["_last_error"] = error_msg
        except Exception as e:
            error_msg = f"Error: {e}"
            print(f"[main] {error_msg}")
            st.session_state["_last_error"] = error_msg
        finally:
            # Use the snapshot taken at submission time (safe from mid-execution toggle)
            auto_approve = st.session_state.pop("pending_auto_approve", True)
            is_reasonflow = selected_usecase == "ReasonFlow Agent"

            if not is_reasonflow or auto_approve:
                # Pipeline done — unlock input and rerun so chat_input
                # re-renders as enabled (it was rendered disabled above).
                st.session_state.pop("pending_query", None)
                st.session_state.processing = False
                st.rerun()
            else:
                # HITL mode — keep input locked and preserve pending_query so the
                # approval UI is re-rendered on the next rerun/button click.
                print(
                    f"[main] HITL waiting for approval | "
                    f"pending_query={bool(st.session_state.get('pending_query'))}"
                )

    # ── Show existing chat history (after rerun, no pending work) ─
    elif st.session_state.chat_history and not is_processing:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                # Pipeline details AFTER the response text (so answer is visible first)
                if msg["role"] == "assistant" and msg.get("pipeline"):
                    st.markdown("---")
                    DisplayResultStreamlit.render_pipeline_replay(msg["pipeline"])

    _render_footer()
