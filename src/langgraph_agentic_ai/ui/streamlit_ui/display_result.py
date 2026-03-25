"""
display_result.py
Renders streaming results for every ReasonFlow use-case on the Streamlit UI.
Handles Basic Chatbot, Chatbot With Web, AI News, and the flagship
ReasonFlow multi-agent pipeline with live Mermaid graphs, phase bars,
agent cards, HITL approval, and self-reflection.
"""

import os
import glob
import re
import time
import streamlit as st
import streamlit.components.v1 as components
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

try:
    from langgraph.types import Command
except ImportError:
    Command = None


# ═══════════════════════════════════════════════════════════════════════════
#  STATIC DATA
# ═══════════════════════════════════════════════════════════════════════════

GRAPH_DEFINITIONS = {
    "Basic Chatbot": """
graph LR
    START([<b>START</b>]):::start --> chatbot[💬 Chatbot]:::agent --> END([<b>END</b>]):::finish
    classDef start fill:#0D1117,stroke:#6C63FF,stroke-width:2px,color:#6C63FF
    classDef finish fill:#0D1117,stroke:#10B981,stroke-width:2px,color:#10B981
    classDef agent fill:#161B22,stroke:#6C63FF,stroke-width:1.5px,color:#E6EDF3
""",
    "Chatbot With Web": """
graph LR
    START([<b>START</b>]):::start --> chatbot[🤖 Agent]:::agent
    chatbot <-->|ReAct| tools[🔧 Tools]:::tool
    chatbot --> END([<b>END</b>]):::finish
    classDef start fill:#0D1117,stroke:#6C63FF,stroke-width:2px,color:#6C63FF
    classDef finish fill:#0D1117,stroke:#10B981,stroke-width:2px,color:#10B981
    classDef agent fill:#161B22,stroke:#6C63FF,stroke-width:1.5px,color:#E6EDF3
    classDef tool fill:#161B22,stroke:#F59E0B,stroke-width:1.5px,color:#F59E0B
""",
    "AI News": """
graph LR
    START([<b>START</b>]):::start --> news_agent[📰 News Agent]:::agent
    news_agent <-->|ReAct| tools[🔧 Tools]:::tool
    news_agent --> save_result[💾 Save]:::save --> END([<b>END</b>]):::finish
    classDef start fill:#0D1117,stroke:#6C63FF,stroke-width:2px,color:#6C63FF
    classDef finish fill:#0D1117,stroke:#10B981,stroke-width:2px,color:#10B981
    classDef agent fill:#161B22,stroke:#6C63FF,stroke-width:1.5px,color:#E6EDF3
    classDef tool fill:#161B22,stroke:#F59E0B,stroke-width:1.5px,color:#F59E0B
    classDef save fill:#161B22,stroke:#10B981,stroke-width:1.5px,color:#10B981
""",
}

NODE_MAP = {
    "chatbot": "chatbot",
    "tools": "tools",
    "news_agent": "news_agent",
    "save_result": "save_result",
}

AGENT_META = {
    "researcher":   {"icon": "\U0001f52c", "color": "#48BFE3", "label": "Researcher"},
    "coder":        {"icon": "\U0001f4bb", "color": "#10B981", "label": "Coder"},
    "analyst":      {"icon": "\U0001f4ca", "color": "#F59E0B", "label": "Analyst"},
    "writer":       {"icon": "\u270d\ufe0f", "color": "#9B59B6", "label": "Writer"},
    "planner":      {"icon": "\U0001f4cb", "color": "#F39C12", "label": "Planner"},
    "fact_checker": {"icon": "\U0001f50e", "color": "#E74C3C", "label": "Fact-Checker"},
    "math_solver":  {"icon": "\U0001f522", "color": "#1ABC9C", "label": "Math Solver"},
    "visualizer":   {"icon": "\U0001f4c8", "color": "#3498DB", "label": "Visualizer"},
    "critic":       {"icon": "\U0001f3af", "color": "#E67E22", "label": "Critic"},
}

PHASE_DEFS = [
    ("\U0001f4cb", "Plan"),
    ("\u26a1", "Execute"),
    ("\U0001f9e0", "Synthesize"),
    ("\U0001f50d", "Reflect"),
]


# ═══════════════════════════════════════════════════════════════════════════
#  MERMAID HELPERS — ReasonFlow dynamic graph
# ═══════════════════════════════════════════════════════════════════════════

_MERMAID_CLASSDEFS = """
    classDef researcher fill:#0B2E3A,stroke:#48BFE3,stroke-width:1.5px,color:#48BFE3
    classDef coder fill:#062B1F,stroke:#10B981,stroke-width:1.5px,color:#10B981
    classDef analyst fill:#2B1D05,stroke:#F59E0B,stroke-width:1.5px,color:#F59E0B
    classDef writer fill:#1E0E2B,stroke:#9B59B6,stroke-width:1.5px,color:#9B59B6
    classDef planner fill:#2B1E05,stroke:#F39C12,stroke-width:1.5px,color:#F39C12
    classDef fact_checker fill:#2B0B08,stroke:#E74C3C,stroke-width:1.5px,color:#E74C3C
    classDef math_solver fill:#05241E,stroke:#1ABC9C,stroke-width:1.5px,color:#1ABC9C
    classDef visualizer fill:#071E2E,stroke:#3498DB,stroke-width:1.5px,color:#3498DB
    classDef critic fill:#2B1508,stroke:#E67E22,stroke-width:1.5px,color:#E67E22
    classDef router fill:#161B22,stroke:#6C63FF,stroke-width:2px,color:#6C63FF
    classDef active fill:#6C63FF,stroke:#6C63FF,stroke-width:2.5px,color:#FFFFFF
"""

_AGENT_IDS = {
    "researcher": "R", "coder": "C", "analyst": "A",
    "writer": "W", "planner": "PL", "fact_checker": "FC",
    "math_solver": "MS", "visualizer": "V", "critic": "CR",
}


def _build_reasonflow_mermaid(
    delegation_plan=None,
    route_type="PIPELINE",
    active_node=None,
    completed_nodes=None,
    retry_count=0,
    final_verdict=None,
):
    """Build a Mermaid graph that reflects the actual executed backend route."""
    delegation_plan = delegation_plan or []
    completed_nodes = completed_nodes or set()

    def _agent_id(agent_name: str) -> str:
        if agent_name in _AGENT_IDS:
            return _AGENT_IDS[agent_name]
        cleaned = re.sub(r"[^A-Za-z0-9]", "", agent_name or "AG")
        return (cleaned[:4] or "AG").upper()

    node_id_map = {
        "router": "ROUTER",
        "supervisor_plan": "SUP",
        "supervisor_synthesize": "SYNTH",
        "reflector": "REFL",
    }

    classdefs = "\n".join([
        '    classDef start fill:#0D1117,stroke:#6C63FF,stroke-width:2px,color:#6C63FF',
        '    classDef finish fill:#0D1117,stroke:#10B981,stroke-width:2px,color:#10B981',
        '    classDef node fill:#161B22,stroke:#6C63FF,stroke-width:1.5px,color:#E6EDF3',
        '    classDef completed fill:#1A2533,stroke:#10B981,stroke-width:2px,color:#E6EDF3',
        _MERMAID_CLASSDEFS,
    ])

    lines = ["graph TD"]
    lines.append('    START(["START"]):::start --> ROUTER{{"Router"}}:::router')

    if route_type == "DIRECT":
        lines.append('    ROUTER -->|"DIRECT"| END(["END"]):::finish')
    else:
        lines.append('    ROUTER -->|"PIPELINE"| SUP["Supervisor Plan"]:::router')

        if not delegation_plan:
            lines.append('    SUP -. planning .-> END(["END"]):::finish')
        else:
            for task in delegation_plan:
                agent = task.get("agent", "")
                if not agent:
                    continue
                aid = _agent_id(agent)
                node_id_map[agent] = aid
                label = AGENT_META.get(agent, {}).get("label", agent.replace("_", " ").title())
                acls = agent if agent in _AGENT_IDS else "node"
                lines.append(f'    {aid}["{label}"]:::{acls}')

            for task in delegation_plan:
                agent = task.get("agent", "")
                if not agent:
                    continue
                aid = node_id_map[agent]
                deps = task.get("depends_on", []) or []
                if not deps:
                    lines.append(f'    SUP -->|"dispatch"| {aid}')
                else:
                    for dep in deps:
                        dep_id = node_id_map.get(dep, _agent_id(dep))
                        lines.append(f'    {dep_id} -->|"depends_on"| {aid}')
                lines.append(f'    {aid} --> SYNTH["Synthesis"]:::router')

            if any((task.get("depends_on") or []) for task in delegation_plan):
                lines.append('    SYNTH -. next_phase .-> SUP')

            lines.append('    SYNTH -->|"reflect"| REFL["Reflector"]:::router')
            if retry_count:
                lines.append('    REFL -. retry .-> SUP')
            verdict_label = final_verdict if final_verdict in {"PASS", "RETRY"} else "done"
            lines.append(f'    REFL -->|"{verdict_label}"| END(["END"]):::finish')

    lines.append(classdefs)

    completed_ids = [node_id_map[n] for n in completed_nodes if n in node_id_map]
    if completed_ids:
        lines.append(f"    class {','.join(sorted(set(completed_ids)))} completed")

    if active_node and active_node in node_id_map:
        lines.append(f"    class {node_id_map[active_node]} active")

    return "\n".join(lines)


def _render_mermaid(usecase, active_node=None):
    """Returns Mermaid code for the demo use-cases with optional active-node highlight."""
    base = GRAPH_DEFINITIONS.get(usecase, "")
    if not base:
        return ""
    if active_node and active_node in NODE_MAP:
        mapped = NODE_MAP[active_node]
        base += f"\n    class {mapped} active"
        base += "\n    classDef active fill:#6C63FF,stroke:#6C63FF,stroke-width:2.5px,color:#FFFFFF"
    return base


def _mermaid_html(mermaid_code):
    """Wraps Mermaid code in a full HTML document that renders via CDN.

    SVG is fitted with a generous minimum size, then zoom buttons adjust pixel
    width/height (so scrollbars work). The scroll area allows panning large graphs.
    """
    return (
        '<!DOCTYPE html><html><head>'
        '<meta charset="utf-8"/>'
        '<meta name="viewport" content="width=device-width,initial-scale=1"/>'
        '<script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>'
        '<style>html,body{height:100%;margin:0;background:#0D1117;}'
        "#app{height:100%;display:flex;flex-direction:column;}"
        "#toolbar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;padding:8px 12px;"
        "background:#161B22;border-bottom:1px solid #30363D;color:#E6EDF3;"
        "font-family:system-ui,sans-serif;font-size:13px;}"
        "#toolbar button{background:#21262D;border:1px solid #30363D;color:#E6EDF3;"
        "border-radius:6px;padding:5px 12px;cursor:pointer;}"
        "#toolbar button:hover{background:#30363D}"
        "#toolbar .hint{font-size:11px;color:#8B949E;margin-left:4px;}"
        "#scrollwrap{flex:1;min-height:0;overflow:auto;-webkit-overflow-scrolling:touch;}"
        "#wrap{box-sizing:border-box;display:inline-flex;align-items:flex-start;justify-content:flex-start;"
        "padding:16px;min-width:min(100%,720px);}"
        ".mermaid{background:transparent;border-radius:12px;padding:4px;display:flex;"
        "align-items:center;justify-content:center;}"
        ".mermaid svg{display:block;overflow:visible!important;}"
        ".mermaid svg foreignObject{overflow:visible;}</style>"
        '</head><body>'
        '<div id="app">'
        '<div id="toolbar">'
        '<span style="opacity:.9">Pipeline graph</span>'
        '<button type="button" id="zin">Zoom +</button>'
        '<button type="button" id="zout">Zoom \u2212</button>'
        '<button type="button" id="zreset">Reset</button>'
        '<span class="hint">Scroll to pan &middot; Use zoom for small labels</span>'
        '</div>'
        '<div id="scrollwrap">'
        '<div id="wrap">'
        f'<div class="mermaid">{mermaid_code}</div>'
        '</div></div></div>'
        '<script>'
        "mermaid.initialize({"
        'startOnLoad:true,theme:"dark",'
        'themeVariables:{fontSize:"16px",primaryTextColor:"#E6EDF3",primaryColor:"#6C63FF",'
        'secondaryColor:"#161B22",tertiaryColor:"#0D1117",'
        'lineColor:"#8B949E",arrowheadColor:"#C9D1D9"},'
        "flowchart:{"
        'useMaxWidth:false,htmlLabels:true,nodeSpacing:64,rankSpacing:72,padding:26,'
        'curve:"basis",wrappingWidth:220'
        "}"
        "});"
        "function _fitMermaidSvg(){"
        "var sw=document.getElementById('scrollwrap');"
        "var wrap=document.getElementById('wrap');"
        "var svg=document.querySelector('.mermaid svg');"
        "if(!sw||!wrap||!svg)return;"
        "try{"
        "svg.style.overflow='visible';"
        "var bb=svg.getBBox();"
        "if(bb.width<0.5||bb.height<0.5)return;"
        "var margin=40;"
        "var vx=bb.x-margin;var vy=bb.y-margin;"
        "var vw=bb.width+2*margin;var vh=bb.height+2*margin;"
        "svg.setAttribute('viewBox',vx+' '+vy+' '+vw+' '+vh);"
        "var pad=32;var cw=Math.max(sw.clientWidth-pad,200);var ch=Math.max(sw.clientHeight-pad,200);"
        "var s=Math.min(cw/vw,ch/vh,1.35);"
        "var minW=Math.min(cw*0.92,920);"
        "if(vw*s<minW){s=Math.min(minW/vw,ch/vh,1.55);}"
        "s=Math.max(s,0.42);"
        "var W=Math.round(vw*s),H=Math.round(vh*s);"
        "svg.setAttribute('width',String(W));svg.setAttribute('height',String(H));"
        "svg.dataset.baseW=String(W);svg.dataset.baseH=String(H);"
        "svg.style.maxWidth='none';svg.style.maxHeight='none';"
        "svg.setAttribute('preserveAspectRatio','xMidYMid meet');"
        "}catch(e){}"
        "}"
        "function _zoom(factor){"
        "var svg=document.querySelector('.mermaid svg');"
        "if(!svg)return;var w=parseFloat(svg.getAttribute('width'))||100;"
        "var h=parseFloat(svg.getAttribute('height'))||100;"
        "svg.setAttribute('width',String(Math.round(w*factor)));"
        "svg.setAttribute('height',String(Math.round(h*factor)));"
        "}"
        "function _zoomReset(){"
        "var svg=document.querySelector('.mermaid svg');"
        "if(!svg||!svg.dataset.baseW)return;"
        "svg.setAttribute('width',svg.dataset.baseW);svg.setAttribute('height',svg.dataset.baseH);"
        "}"
        "document.getElementById('zin').onclick=function(){_zoom(1.2);};"
        "document.getElementById('zout').onclick=function(){_zoom(1/1.2);};"
        "document.getElementById('zreset').onclick=_zoomReset;"
        "setTimeout(_fitMermaidSvg,60);setTimeout(_fitMermaidSvg,250);"
        "setTimeout(_fitMermaidSvg,600);setTimeout(_fitMermaidSvg,950);"
        "window.addEventListener('resize',function(){setTimeout(_fitMermaidSvg,120);});"
        '</script>'
        '</body></html>'
    )


def _reasonflow_mermaid_iframe_height(agent_count: int) -> int:
    """Height for ReasonFlow TD Mermaid in ``components.html`` (scrollable)."""
    if agent_count <= 0:
        return 420
    # Wider/taller iframe so the initial fit stays readable; user can scroll inside.
    return min(520 + agent_count * 72, 920)


# ═══════════════════════════════════════════════════════════════════════════
#  HTML HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def _esc(text):
    """Escape HTML entities."""
    if not text:
        return ""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _plan_html(steps):
    """Render a numbered plan as HTML."""
    if not steps:
        return ""
    html = ""
    for i, step in enumerate(steps, 1):
        html += (
            f'<div class="plan-step">'
            f'<span class="num">{i}</span>'
            f'<span>{_esc(step)}</span>'
            f'</div>'
        )
    return html


def _tool_card_html(tool_name, content):
    """Render a tool result card."""
    return (
        f'<div class="tool-result-card">'
        f'<div class="tool-name">{_esc(tool_name)}</div>'
        f'<pre>{_esc(content[:500])}</pre>'
        f'</div>'
    )


def _reflection_html(verdict, feedback):
    """Render a reflection verdict card."""
    cls = "pass" if verdict == "PASS" else "retry"
    icon = "\u2705" if verdict == "PASS" else "\U0001f504"
    return (
        f'<div class="reflection-card {cls}">'
        f'<strong>{icon} Reflection: {_esc(verdict)}</strong><br>'
        f'<span style="font-size:0.82rem;">{_esc(feedback)}</span>'
        f'</div>'
    )


def _sanitize_response_markdown(text, has_chart=False):
    """Remove chart placeholders/artifacts; real charts are rendered separately by the UI."""
    if not text:
        return ""
    cleaned = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", str(text))
    cleaned = re.sub(r"<img\b[^>]*>", "", cleaned, flags=re.IGNORECASE)
    if has_chart:
        # Drop fenced pseudo-chart blocks (ASCII bars, labels, value callouts) when a real image is shown.
        cleaned = re.sub(
            r"\n{0,2}```\n(?=[\s\S]{0,400}?(SPY|QQQ|DIA|NVDA|AMD|INTC))(?=[\s\S]{0,400}?[|_/\-]{3,})[\s\S]*?```",
            "\n",
            cleaned,
            flags=re.IGNORECASE,
        )
        # Remove headings whose only purpose is to introduce the already-rendered chart.
        cleaned = re.sub(
            r"(?im)^#{1,6}\s*(bar chart|visualization|chart)\b[^\n]*\n(?:[^\n#].*\n){0,4}",
            "",
            cleaned,
        )
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def _delegation_plan_html(plan):
    """Render the delegation plan with agent icons and dependency arrows."""
    if not plan:
        return ""
    html = ""
    for task in plan:
        agent = task.get("agent", "unknown")
        meta = AGENT_META.get(agent, {"icon": "\U0001f916", "color": "#6C63FF", "label": agent.title()})
        task_text = task.get("task", "")
        deps = task.get("depends_on", [])
        deps_str = ""
        if deps:
            dep_labels = [AGENT_META.get(d, {}).get("label", d.title()) for d in deps]
            deps_str = (
                f'<span style="font-size:0.7rem;color:#8B949E;"> '
                f'\u2190 depends on: {", ".join(dep_labels)}</span>'
            )
        html += (
            f'<div class="deleg-step">'
            f'<div class="deleg-agent-icon {_esc(agent)}">{meta["icon"]}</div>'
            f'<div class="deleg-task">'
            f'<strong style="color:{meta["color"]}">{_esc(meta["label"])}</strong>{deps_str}<br>'
            f'<span style="font-size:0.78rem;color:#C9D1D9;">{_esc(task_text[:200])}</span>'
            f'</div>'
            f'</div>'
        )
    return html


def _phase_bar_html(phase_idx, status_text=""):
    """Render the 4-phase progress bar (Plan / Execute / Synthesize / Reflect)."""
    html = '<div class="phase-bar">'
    for i, (icon, label) in enumerate(PHASE_DEFS):
        if i < phase_idx:
            cls = "done"
        elif i == phase_idx:
            cls = "active"
        else:
            cls = "pending"
        html += (
            f'<div class="phase-step {cls}">'
            f'<div class="phase-icon">{icon}</div>'
            f'<span>{label}</span>'
            f'</div>'
        )
        if i < len(PHASE_DEFS) - 1:
            conn_cls = "done" if i < phase_idx else ""
            html += f'<div class="phase-connector {conn_cls}"></div>'
    if status_text:
        html += f'<div class="phase-status">{_esc(status_text)}</div>'
    html += '</div>'
    return html


def _agent_working_html(agent, task=""):
    """Render an animated 'working' card for a specialist agent."""
    meta = AGENT_META.get(agent, {"icon": "\U0001f916", "color": "#6C63FF", "label": agent.title()})
    return (
        f'<div class="agent-working-card {_esc(agent)}">'
        f'<div class="agent-header">'
        f'<span class="agent-icon">{meta["icon"]}</span>'
        f'<span class="agent-name {_esc(agent)}">{_esc(meta["label"])}</span>'
        f'<span class="working-badge">WORKING</span>'
        f'</div>'
        f'<div class="agent-task">{_esc(task[:150])}</div>'
        f'<div class="working-indicator">'
        f'<div class="working-dots"><span></span><span></span><span></span></div>'
        f'<span>Processing with tools...</span>'
        f'</div>'
        f'</div>'
    )


def _agent_done_card_html(agent, task="", result="", tools_used=None, confidence=0.5,
                           execution_time=0, error=False, model_note=""):
    """Render a completed agent result card."""
    meta = AGENT_META.get(agent, {"icon": "\U0001f916", "color": "#6C63FF", "label": agent.title()})
    tools_used = tools_used or []

    tools_html = ""
    if tools_used:
        tools_html = '<div class="agent-tools">'
        for t in tools_used:
            tools_html += f'<span class="agent-tool-tag">{_esc(t)}</span>'
        tools_html += '</div>'

    error_html = ""
    if error or confidence == 0.0:
        error_html = (
            '<div class="agent-error">'
            '\u26a0\ufe0f Agent encountered issues - results may be incomplete'
            '</div>'
        )

    # Model fallback notice
    model_note_html = ""
    if model_note:
        model_note_html = (
            f'<div style="font-size:0.7rem;color:#F59E0B;margin-bottom:4px;'
            f'padding:2px 6px;background:rgba(245,158,11,0.1);border-radius:4px;'
            f'display:inline-block;">'
            f'\u26a0\ufe0f {_esc(model_note)}'
            f'</div>'
        )

    conf_pct = f"{confidence:.0%}"
    time_str = f"{execution_time:.1f}s" if execution_time else ""
    meta_parts = []
    if conf_pct:
        meta_parts.append(f"Confidence: {conf_pct}")
    if time_str:
        meta_parts.append(time_str)
    meta_str = " \u00b7 ".join(meta_parts)

    # Truncate long results for display
    display_result = result[:800] if result else "No output."

    return (
        f'<div class="agent-done-card {_esc(agent)}">'
        f'<div class="agent-header">'
        f'<span class="agent-icon">{meta["icon"]}</span>'
        f'<span class="agent-name {_esc(agent)}">{_esc(meta["label"])}</span>'
        f'<span class="done-check">\u2713 DONE</span>'
        f'</div>'
        f'<div class="agent-task">{_esc(task[:150])}</div>'
        f'{model_note_html}'
        f'{error_html}'
        f'{tools_html}'
        f'<div style="font-size:0.7rem;color:#8B949E;margin-bottom:4px;">{meta_str}</div>'
        f'<div class="agent-output">{_esc(display_result)}</div>'
        f'</div>'
    )


def _completion_badge_html(agents_count=0, execution_time=0, verdict="PASS",
                           token_estimate=0):
    """Render a completion summary badge."""
    parts = ["\u2705 Complete"]
    if agents_count:
        parts.append(f"{agents_count} agents")
    if execution_time:
        parts.append(f"{execution_time:.1f}s")
    if verdict:
        parts.append(f"Verdict: {verdict}")
    if token_estimate:
        parts.append(f"~{token_estimate:,} tokens")
    return f'<div class="completion-badge">{" \u00b7 ".join(parts)}</div>'


def _section_header_html(kicker, title, subtitle=""):
    """Render a lightweight section header for result surfaces."""
    subtitle_html = (
        f'<p class="rf-section-subtitle">{_esc(subtitle)}</p>' if subtitle else ""
    )
    return (
        f'<div class="rf-section">'
        f'<div class="rf-section-kicker">{_esc(kicker)}</div>'
        f'<div class="rf-section-title">{_esc(title)}</div>'
        f'{subtitle_html}'
        f'</div>'
    )


def _render_chart_images():
    """Detect and display any chart images generated by the Visualizer agent.
    Charts are saved to /tmp/reasonflow_chart*.png by the Visualizer's Python REPL."""
    chart_patterns = ["/tmp/reasonflow_chart*.png", "/tmp/reasonflow_chart*.jpg"]
    found = []
    for pattern in chart_patterns:
        found.extend(glob.glob(pattern))
    if not found:
        return False

    displayed = False
    for chart_path in sorted(found, key=os.path.getmtime, reverse=True):
        try:
            st.image(chart_path, caption="Generated Chart", width="stretch")
            displayed = True
        except Exception:
            pass
    return displayed


def _has_chart_images():
    """Check whether a generated chart image currently exists in /tmp."""
    for pattern in ["/tmp/reasonflow_chart*.png", "/tmp/reasonflow_chart*.jpg"]:
        if glob.glob(pattern):
            return True
    return False


def _capture_chart_blobs():
    """Capture generated chart images as in-memory blobs for chat-history replay."""
    chart_patterns = ["/tmp/reasonflow_chart*.png", "/tmp/reasonflow_chart*.jpg"]
    found = []
    for pattern in chart_patterns:
        found.extend(glob.glob(pattern))

    blobs = []
    for chart_path in sorted(found, key=os.path.getmtime, reverse=True):
        try:
            with open(chart_path, "rb") as f:
                blobs.append({
                    "name": os.path.basename(chart_path),
                    "bytes": f.read(),
                })
        except OSError:
            continue
    return blobs


def _cleanup_chart_images():
    """Remove chart images from /tmp after displaying them, to avoid stale charts on next query."""
    for pattern in ["/tmp/reasonflow_chart*.png", "/tmp/reasonflow_chart*.jpg"]:
        for f in glob.glob(pattern):
            try:
                os.remove(f)
            except OSError:
                pass


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN CLASS
# ═══════════════════════════════════════════════════════════════════════════

class DisplayResultStreamlit:
    """Renders the result of graph invocations on the Streamlit UI."""

    def __init__(self, usecase, graph, user_message, config=None):
        self.usecase = usecase
        self.graph = graph
        self.user_message = user_message
        self.config = config

    # ── Chat history ────────────────────────────────────────────────────
    def _render_chat_history(self):
        """Display stored chat history from session state."""
        for msg in st.session_state.get("chat_history", []):
            role = msg.get("role", "assistant")
            with st.chat_message(role):
                st.markdown(msg.get("content", ""), unsafe_allow_html=True)
                if role == "assistant" and msg.get("pipeline"):
                    self.render_pipeline_replay(msg.get("pipeline"))

    @staticmethod
    def render_hitl_plan_approval(user_message, plan):
        """Render a saved HITL approval card without re-running the graph."""
        with st.chat_message("user"):
            st.write(user_message)

        with st.chat_message("assistant"):
            st.markdown(
                '<div class="hitl-card">'
                '<div class="hitl-title">\U0001f6d1 Plan Approval Required</div>'
                '<div class="hitl-subtitle">Review the delegation plan before agents execute.</div>'
                '</div>',
                unsafe_allow_html=True,
            )
            st.markdown(_delegation_plan_html(plan), unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                if st.button("\u2705 Approve Plan", width="stretch", type="primary", key="hitl_approve_saved"):
                    st.session_state["hitl_pending"] = True
                    st.session_state["hitl_resume"] = {"action": "approve"}
                    st.session_state["hitl_action"] = {"action": "approve"}
                    st.session_state["hitl_plan"] = plan
                    print("[hitl] Approve clicked from saved plan UI; session keys set for resume")
                    st.rerun()
            with col2:
                if st.button("\u274c Reject Plan", width="stretch", key="hitl_reject_saved"):
                    st.session_state["hitl_pending"] = True
                    st.session_state["hitl_resume"] = {"action": "reject"}
                    st.session_state["hitl_action"] = {"action": "reject"}
                    st.session_state["hitl_plan"] = plan
                    print("[hitl] Reject clicked from saved plan UI; session keys set for resume")
                    st.rerun()

    # ── Graph rendering (demo use-cases) ────────────────────────────────
    def _show_graph(self, active_node=None):
        """Render the Mermaid graph for demo use-cases using components.html."""
        mermaid_code = _render_mermaid(self.usecase, active_node)
        if mermaid_code:
            html = _mermaid_html(mermaid_code)
            components.html(html, height=420, scrolling=True)

    # ── Graph rendering (ReasonFlow) ────────────────────────────────────
    def _show_reasonflow_graph(
        self,
        delegation_plan=None,
        route_type="PIPELINE",
        active_node=None,
        completed_nodes=None,
        retry_count=0,
        final_verdict=None,
    ):
        """Render the Mermaid graph for the ReasonFlow pipeline using components.html."""
        mermaid_code = _build_reasonflow_mermaid(
            delegation_plan=delegation_plan,
            route_type=route_type,
            active_node=active_node,
            completed_nodes=completed_nodes,
            retry_count=retry_count,
            final_verdict=final_verdict,
        )
        n = len(delegation_plan) if delegation_plan else 0
        height = _reasonflow_mermaid_iframe_height(n)
        html = _mermaid_html(mermaid_code)
        components.html(html, height=height, scrolling=True)

    # ── Multi-agent streaming ───────────────────────────────────────────
    def _stream_multiagent(self, stream_input, config):
        """
        Streams the ReasonFlow multi-agent pipeline and renders live UI.
        Handles events: router, supervisor_plan, specialist,
        supervisor_synthesize, reflector.
        """
        graph = self.graph

        # --- Placeholders: start EMPTY to avoid flash on DIRECT responses ---
        phase_ph = st.empty()
        graph_container = st.empty()
        activity_container = st.empty()
        activity_expander = None  # Created only for PIPELINE

        # Tracking state
        delegation_plan = []
        agent_names = []
        agent_results_map = {}   # agent_name -> result dict
        working_placeholders = {}  # agent_name -> st placeholder
        completed_agents = set()
        total_start = time.time()
        final_response = None
        verdict = "PASS"
        reflection_feedback = ""
        total_tokens = 0
        retry_count = 0
        route_type = "PIPELINE"
        executed_nodes = set()

        def _render_runtime_graph(active=None):
            with graph_container.container():
                with st.expander("\U0001f5fa\ufe0f Pipeline Graph", expanded=True):
                    self._show_reasonflow_graph(
                        delegation_plan=delegation_plan,
                        route_type=route_type,
                        active_node=active,
                        completed_nodes=executed_nodes,
                        retry_count=retry_count,
                        final_verdict=verdict,
                    )

        def _bootstrap_from_graph_state():
            """Rebuild UI when resuming after HITL approval."""
            nonlocal delegation_plan, agent_names, agent_results_map, completed_agents
            nonlocal verdict, reflection_feedback, retry_count, route_type, activity_expander
            try:
                state_snapshot = graph.get_state(config)
                values = getattr(state_snapshot, "values", {}) or {}
                print(f"[hitl] bootstrap graph state keys: {list(values.keys())}")
            except Exception:
                print("[hitl] bootstrap graph state unavailable")
                return

            route_type = values.get("route_type", route_type) or route_type
            delegation_plan = values.get("delegation_plan") or delegation_plan
            agent_names = [t.get("agent", "") for t in delegation_plan if t.get("agent")]

            prior_results = values.get("agent_results") or []
            for res in prior_results:
                agent = res.get("agent", "")
                if not agent:
                    continue
                agent_results_map[agent] = res
                completed_agents.add(agent)
                executed_nodes.add(agent)

            if values.get("verdict"):
                verdict = values.get("verdict") or verdict
            if values.get("reflection_feedback"):
                reflection_feedback = values.get("reflection_feedback") or reflection_feedback
            retry_count = values.get("reflection_count") or retry_count

            executed_nodes.update({"router", "supervisor_plan"})
            phase_ph.markdown(
                _phase_bar_html(1, f"{len(completed_agents)}/{len(agent_names)} agents done"),
                unsafe_allow_html=True,
            )
            _render_runtime_graph(active="supervisor_plan")
            activity_expander = activity_container.expander(
                "\U0001f916 Agent Activity", expanded=True
            )
            with activity_expander:
                st.markdown(
                    f'<div style="font-size:0.82rem;font-weight:600;color:#6C63FF;margin-bottom:4px;">'
                    f'\U0001f4cb Delegation Plan</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(_delegation_plan_html(delegation_plan), unsafe_allow_html=True)

                for task in delegation_plan:
                    agent = task.get("agent", "")
                    if not agent or agent in working_placeholders:
                        continue
                    working_placeholders[agent] = st.empty()
                    if agent in agent_results_map:
                        res = agent_results_map[agent]
                        working_placeholders[agent].markdown(
                            _agent_done_card_html(
                                agent=agent,
                                task=res.get("task", ""),
                                result=res.get("result", ""),
                                tools_used=res.get("tools_used", []),
                                confidence=res.get("confidence", 0.5),
                                execution_time=res.get("execution_time", 0),
                                error=(res.get("confidence", 0.5) == 0.0),
                                model_note=res.get("model_note", ""),
                            ),
                            unsafe_allow_html=True,
                        )
                    else:
                        working_placeholders[agent].markdown(
                            _agent_working_html(agent, task.get("task", "")),
                            unsafe_allow_html=True,
                        )

        if isinstance(stream_input, Command):
            _bootstrap_from_graph_state()

        # --- Stream events ---
        for event in graph.stream(stream_input, config, stream_mode="updates"):
            for node_name, node_data in event.items():

                # ── ROUTER ──────────────────────────────────────────
                if node_name == "router":
                    route_type = node_data.get("route_type", "PIPELINE")
                    total_tokens += node_data.get("token_usage", 0)
                    executed_nodes.add("router")

                    if route_type == "DIRECT":
                        # Direct response — no pipeline UI at all
                        messages = node_data.get("messages", [])
                        if messages:
                            resp = messages[-1]
                            content = resp.content if hasattr(resp, "content") else str(resp)
                            final_response = _sanitize_response_markdown(content)
                        return final_response

                    # PIPELINE — NOW render the pipeline UI elements
                    phase_ph.markdown(
                        _phase_bar_html(0, "Planning delegation..."),
                        unsafe_allow_html=True,
                    )
                    _render_runtime_graph(active="router")
                    activity_expander = activity_container.expander(
                        "\U0001f916 Agent Activity", expanded=True
                    )

                # ── SUPERVISOR PLAN ─────────────────────────────────
                elif node_name == "supervisor_plan":
                    executed_nodes.add("supervisor_plan")
                    delegation_plan = node_data.get("delegation_plan", [])
                    agent_names = [t.get("agent", "") for t in delegation_plan]

                    # Update phase bar
                    phase_ph.markdown(
                        _phase_bar_html(0, f"Delegating to {len(agent_names)} agent(s)..."),
                        unsafe_allow_html=True,
                    )

                    # Update graph with actual agents (collapsible)
                    _render_runtime_graph(active="supervisor_plan")

                    # Show delegation plan
                    with activity_expander:
                        st.markdown(
                            f'<div style="font-size:0.82rem;font-weight:600;color:#6C63FF;margin-bottom:4px;">'
                            f'\U0001f4cb Delegation Plan</div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(_delegation_plan_html(delegation_plan), unsafe_allow_html=True)

                        # Parallel badge
                        phase0 = [t for t in delegation_plan if not t.get("depends_on")]
                        if len(phase0) > 1:
                            st.markdown(
                                f'<div class="parallel-badge">\u26a1 {len(phase0)} agents running in parallel</div>',
                                unsafe_allow_html=True,
                            )

                        # Create working placeholders for each agent
                        for t in delegation_plan:
                            a = t.get("agent", "")
                            if a and a not in working_placeholders:
                                working_placeholders[a] = st.empty()
                                working_placeholders[a].markdown(
                                    _agent_working_html(a, t.get("task", "")),
                                    unsafe_allow_html=True,
                                )

                    # Update phase to Execute
                    phase_ph.markdown(
                        _phase_bar_html(1, f"0/{len(agent_names)} agents done"),
                        unsafe_allow_html=True,
                    )

                # ── SPECIALIST ──────────────────────────────────────
                elif node_name == "specialist":
                    results_list = node_data.get("agent_results", [])
                    last_agent = None
                    for res in results_list:
                        agent = res.get("agent", "")
                        if not agent:
                            continue

                        agent_results_map[agent] = res
                        completed_agents.add(agent)
                        executed_nodes.add(agent)
                        last_agent = agent
                        total_tokens += res.get("token_estimate", 0)

                        # Replace working card with done card
                        if agent in working_placeholders:
                            working_placeholders[agent].markdown(
                                _agent_done_card_html(
                                    agent=agent,
                                    task=res.get("task", ""),
                                    result=res.get("result", ""),
                                    tools_used=res.get("tools_used", []),
                                    confidence=res.get("confidence", 0.5),
                                    execution_time=res.get("execution_time", 0),
                                    error=(res.get("confidence", 0.5) == 0.0),
                                    model_note=res.get("model_note", ""),
                                ),
                                unsafe_allow_html=True,
                            )

                    # Update phase bar progress
                    done_count = len(completed_agents)
                    total_count = len(agent_names) if agent_names else len(completed_agents)
                    phase_ph.markdown(
                        _phase_bar_html(1, f"{done_count}/{total_count} agents done"),
                        unsafe_allow_html=True,
                    )
                    _render_runtime_graph(active=last_agent)

                # ── SUPERVISOR SYNTHESIZE ───────────────────────────
                elif node_name == "supervisor_synthesize":
                    executed_nodes.add("supervisor_synthesize")
                    phase_ph.markdown(
                        _phase_bar_html(2, "Synthesizing results..."),
                        unsafe_allow_html=True,
                    )

                    # Check for messages (final synthesis)
                    messages = node_data.get("messages", [])
                    if messages:
                        resp = messages[-1]
                        content = resp.content if hasattr(resp, "content") else str(resp)
                        final_response = _sanitize_response_markdown(content)
                    _render_runtime_graph(active="supervisor_synthesize")

                    with activity_expander:
                        st.markdown(
                            '<div class="synthesis-indicator">'
                            '<span class="synth-icon">\U0001f9e0</span>'
                            'Synthesizing agent results into a unified response...'
                            '</div>',
                            unsafe_allow_html=True,
                        )

                # ── REFLECTOR ───────────────────────────────────────
                elif node_name == "reflector":
                    executed_nodes.add("reflector")
                    verdict = node_data.get("verdict", "PASS")
                    reflection_feedback = node_data.get("reflection_feedback", "")

                    phase_ph.markdown(
                        _phase_bar_html(3, f"Verdict: {verdict}"),
                        unsafe_allow_html=True,
                    )

                    with activity_expander:
                        st.markdown(
                            _reflection_html(verdict, reflection_feedback),
                            unsafe_allow_html=True,
                        )
                    _render_runtime_graph(active="reflector")

                    if verdict == "RETRY":
                        retry_count += 1
                        with activity_expander:
                            st.markdown(
                                f'<div class="retry-separator">'
                                f'\U0001f504 Retry #{retry_count} - Reflector requested improvements'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                        # Reset UI placeholders for next loop
                        # NOTE: Do NOT clear agent_results_map — retry may not
                        # re-dispatch agents (they're already completed in state),
                        # so we keep the results for the final pipeline data.
                        working_placeholders.clear()
                        completed_agents.clear()
                        phase_ph.markdown(
                            _phase_bar_html(0, "Re-planning..."),
                            unsafe_allow_html=True,
                        )
                        _render_runtime_graph(active="supervisor_plan")

        # --- Final summary ---
        total_time = time.time() - total_start
        phase_ph.markdown(
            _phase_bar_html(4, "Done"),  # All phases complete
            unsafe_allow_html=True,
        )
        st.markdown(
            _completion_badge_html(
                agents_count=len(agent_results_map),
                execution_time=total_time,
                verdict=verdict,
                token_estimate=total_tokens,
            ),
            unsafe_allow_html=True,
        )
        _render_runtime_graph()

        # ── Save pipeline data for replay after st.rerun() ──────────
        st.session_state["_last_pipeline"] = {
            "delegation_plan": delegation_plan,
            "agent_results": list(agent_results_map.values()),
            "agents": agent_names,
            "verdict": verdict,
            "reflection_feedback": reflection_feedback,
            "total_time": total_time,
            "total_tokens": total_tokens,
            "retry_count": retry_count,
            "route_type": route_type,
            "executed_nodes": sorted(executed_nodes),
        }

        return final_response

    # ── HITL resume ─────────────────────────────────────────────────────
    def display_hitl_resume(self, resume_value):
        """Resume a HITL-interrupted graph with the user's approval/rejection."""
        if Command is None:
            st.error("HITL requires `langgraph.types.Command`. Please upgrade langgraph.")
            return

        config = self.config or {"configurable": {"thread_id": st.session_state.get("thread_id", "default")}}
        cmd = Command(resume=resume_value)
        print(f"[hitl] display_hitl_resume called with: {resume_value}")

        with st.chat_message("assistant"):
            response = self._stream_multiagent(cmd, config)
            print(f"[hitl] resume stream returned response={bool(response)}")
            if response:
                st.markdown("---")
                has_chart = _has_chart_images()
                if has_chart:
                    st.markdown(
                        _section_header_html(
                            "Visualization",
                            "Generated chart",
                            "The chart is rendered separately from the written answer for cleaner presentation.",
                        ),
                        unsafe_allow_html=True,
                    )
                    _render_chart_images()
                response = _sanitize_response_markdown(response, has_chart=has_chart)
                st.markdown(
                    _section_header_html(
                        "Final Answer",
                        "Response",
                        "Synthesized output from the executed agent pipeline.",
                    ),
                    unsafe_allow_html=True,
                )
                st.markdown(response, unsafe_allow_html=True)

        if response:
            history = st.session_state.setdefault("chat_history", [])
            user_message = st.session_state.get("hitl_user_message")
            if user_message:
                history.append({"role": "user", "content": user_message})
            pipeline_data = st.session_state.pop("_last_pipeline", None)
            if pipeline_data:
                pipeline_data["chart_blobs"] = _capture_chart_blobs()
            history.append({
                "role": "assistant",
                "content": response,
                "pipeline": pipeline_data,
            })

    # ── Main dispatch ───────────────────────────────────────────────────
    def display_result_on_ui(self):
        """Primary method: dispatches to the correct handler for each use-case."""
        usecase = self.usecase
        graph = self.graph
        user_message = self.user_message

        # ── Basic Chatbot ───────────────────────────────────────────
        if usecase == "Basic Chatbot":
            self._show_graph()
            with st.chat_message("user"):
                st.write(user_message)

            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                full_response = ""
                for event in graph.stream({"messages": ("user", user_message)}):
                    for value in event.values():
                        msg = value.get("messages")
                        if msg:
                            content = msg.content if hasattr(msg, "content") else str(msg)
                            full_response = content
                response_placeholder.markdown(full_response)

            # Save to chat history
            history = st.session_state.setdefault("chat_history", [])
            history.append({"role": "user", "content": user_message})
            history.append({"role": "assistant", "content": full_response})

        # ── Chatbot With Web ────────────────────────────────────────
        elif usecase == "Chatbot With Web":
            self._show_graph()
            initial_state = {"messages": [HumanMessage(content=user_message)]}
            with st.spinner("Agent is thinking..."):
                res = graph.invoke(initial_state)

            for message in res.get("messages", []):
                if isinstance(message, HumanMessage):
                    with st.chat_message("user"):
                        st.write(message.content)
                elif isinstance(message, ToolMessage):
                    with st.chat_message("ai"):
                        st.markdown(
                            _tool_card_html(
                                message.name if hasattr(message, "name") else "Tool",
                                message.content,
                            ),
                            unsafe_allow_html=True,
                        )
                elif isinstance(message, AIMessage) and message.content:
                    with st.chat_message("assistant"):
                        st.markdown(message.content)

        # ── AI News ─────────────────────────────────────────────────
        elif usecase == "AI News":
            self._show_graph()
            frequency = self.user_message
            with st.spinner("Fetching and summarizing news... \u23f3"):
                result = graph.invoke({"messages": frequency})
                try:
                    AI_NEWS_PATH = f"./AINews/{frequency.lower()}_summary.md"
                    with open(AI_NEWS_PATH, "r") as file:
                        markdown_content = file.read()
                    st.markdown(markdown_content, unsafe_allow_html=True)
                except FileNotFoundError:
                    st.error(f"News Not Generated or File not found: {AI_NEWS_PATH}")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

        # ── ReasonFlow Agent ────────────────────────────────────────
        elif usecase == "ReasonFlow Agent":
            auto_approve = st.session_state.get("auto_approve", True)
            config = self.config or {
                "configurable": {
                    "thread_id": st.session_state.get("thread_id", "default"),
                }
            }

            with st.chat_message("user"):
                st.write(user_message)

            with st.chat_message("assistant"):
                if auto_approve:
                    self._run_auto_approve(graph, user_message, config, auto_approve)
                else:
                    self._run_with_hitl(graph, user_message, config)

    # ── Auto-approve flow ───────────────────────────────────────────────
    def _run_auto_approve(self, graph, user_message, config, auto_approve):
        """Run the ReasonFlow pipeline with auto-approval (no HITL interrupts)."""
        stream_input = {
            "messages": [HumanMessage(content=user_message)],
            "auto_approve": auto_approve,
        }

        # Clean stale charts before running pipeline
        _cleanup_chart_images()

        response = self._stream_multiagent(stream_input, config)

        if response:
            st.markdown("---")
            # Show any chart images generated by the Visualizer agent
            has_chart = _has_chart_images()
            if has_chart:
                st.markdown(
                    _section_header_html(
                        "Visualization",
                        "Generated chart",
                        "Rendered from the visualizer output and kept separate from the prose summary.",
                    ),
                    unsafe_allow_html=True,
                )
                _render_chart_images()
            response = _sanitize_response_markdown(response, has_chart=has_chart)
            st.markdown(
                _section_header_html(
                    "Final Answer",
                    "Response",
                    "A cleaned synthesis of the pipeline results.",
                ),
                unsafe_allow_html=True,
            )
            st.markdown(response, unsafe_allow_html=True)

            # Save to history (include pipeline data for replay after st.rerun)
            pipeline_data = st.session_state.pop("_last_pipeline", None)
            # Track chart paths for replay
            if has_chart:
                if pipeline_data:
                    pipeline_data["chart_blobs"] = _capture_chart_blobs()
            history = st.session_state.setdefault("chat_history", [])
            history.append({"role": "user", "content": user_message})
            history.append({
                "role": "assistant",
                "content": response,
                "pipeline": pipeline_data,
            })
        else:
            st.info("No response was generated. Try rephrasing your query.")

    # ── HITL flow ───────────────────────────────────────────────────────
    def _run_with_hitl(self, graph, user_message, config):
        """
        Run the ReasonFlow pipeline with HITL approval gate.
        The graph will interrupt at supervisor_plan for human approval.
        """
        stream_input = {
            "messages": [HumanMessage(content=user_message)],
            "auto_approve": False,
        }

        # Initial run — will pause at interrupt()
        try:
            response = self._stream_multiagent(stream_input, config)
        except Exception as e:
            err_str = str(e)
            # Check if this is a HITL interrupt
            if "interrupt" in err_str.lower():
                response = None
            else:
                st.error(f"Error: {err_str}")
                return

        # Check graph state for pending interrupt
        try:
            state = graph.get_state(config)
            pending = state.tasks if hasattr(state, "tasks") else []

            if pending:
                # There is a pending interrupt — show approval UI
                interrupt_data = None
                for task in pending:
                    if hasattr(task, "interrupts") and task.interrupts:
                        interrupt_data = task.interrupts[0].value
                        break

                if interrupt_data and interrupt_data.get("type") == "plan_approval":
                    plan = interrupt_data.get("delegation_plan", [])
                    st.session_state["hitl_user_message"] = user_message
                    st.session_state["hitl_plan"] = plan

                    st.markdown(
                        '<div class="hitl-card">'
                        '<div class="hitl-title">\U0001f6d1 Plan Approval Required</div>'
                        '<div class="hitl-subtitle">Review the delegation plan before agents execute.</div>'
                        '</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(_delegation_plan_html(plan), unsafe_allow_html=True)

                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("\u2705 Approve Plan", width="stretch", type="primary"):
                            st.session_state["hitl_pending"] = True
                            st.session_state["hitl_resume"] = {"action": "approve"}
                            st.session_state["hitl_action"] = {"action": "approve"}
                            st.session_state["hitl_plan"] = plan
                            print("[hitl] Approve clicked; session keys set for resume")
                            st.rerun()
                    with col2:
                        if st.button("\u274c Reject Plan", width="stretch"):
                            st.session_state["hitl_pending"] = True
                            st.session_state["hitl_resume"] = {"action": "reject"}
                            st.session_state["hitl_action"] = {"action": "reject"}
                            st.session_state["hitl_plan"] = plan
                            print("[hitl] Reject clicked; session keys set for resume")
                            st.rerun()
                    return

        except Exception:
            # No interrupt pending, graph completed normally
            pass

        if response:
            st.markdown("---")
            has_chart = _has_chart_images()
            if has_chart:
                st.markdown(
                    _section_header_html(
                        "Visualization",
                        "Generated chart",
                        "Rendered separately so the response can stay concise and readable.",
                    ),
                    unsafe_allow_html=True,
                )
                _render_chart_images()
            response = _sanitize_response_markdown(response, has_chart=has_chart)
            st.markdown(
                _section_header_html(
                    "Final Answer",
                    "Response",
                    "The approved plan has completed and the final synthesis is shown below.",
                ),
                unsafe_allow_html=True,
            )
            st.markdown(response, unsafe_allow_html=True)

            history = st.session_state.setdefault("chat_history", [])
            history.append({"role": "user", "content": user_message})
            pipeline_data = st.session_state.pop("_last_pipeline", None)
            if pipeline_data:
                pipeline_data["chart_blobs"] = _capture_chart_blobs()
            history.append({
                "role": "assistant",
                "content": response,
                "pipeline": pipeline_data,
            })
        else:
            st.info("Waiting for pipeline to complete...")

    # ── Pipeline replay (static, for chat history after st.rerun) ────
    @staticmethod
    def render_pipeline_replay(pipeline_data):
        """
        Render a compact, static pipeline summary from saved metadata.
        Called when replaying chat history after st.rerun() so the
        pipeline visualization persists instead of vanishing.
        """
        if not pipeline_data:
            return

        results = pipeline_data.get("agent_results", [])
        agents = pipeline_data.get("agents", [])
        delegation_plan = pipeline_data.get("delegation_plan", [])
        verdict = pipeline_data.get("verdict", "PASS")
        total_time = pipeline_data.get("total_time", 0)
        total_tokens = pipeline_data.get("total_tokens", 0)
        route_type = pipeline_data.get("route_type", "PIPELINE")
        executed_nodes = set(pipeline_data.get("executed_nodes", []))
        retry_count = pipeline_data.get("retry_count", 0)

        # Completion badge at top
        st.markdown(
            _completion_badge_html(
                agents_count=len(results),
                execution_time=total_time,
                verdict=verdict,
                token_estimate=total_tokens,
            ),
            unsafe_allow_html=True,
        )

        # Chart images from Visualizer agent (replay from saved paths)
        chart_blobs = pipeline_data.get("chart_blobs", [])
        if chart_blobs:
            st.markdown(
                _section_header_html("Visualization", "Generated chart"),
                unsafe_allow_html=True,
            )
            for blob in chart_blobs:
                data = blob.get("bytes")
                if not data:
                    continue
                try:
                    st.image(data, caption="Generated Chart", width="stretch")
                except Exception:
                    pass
        else:
            chart_paths = pipeline_data.get("chart_paths", [])
            for cp in chart_paths:
                if os.path.isfile(cp):
                    st.markdown(
                        _section_header_html("Visualization", "Generated chart"),
                        unsafe_allow_html=True,
                    )
                    try:
                        st.image(cp, caption="Generated Chart", width="stretch")
                    except Exception:
                        pass

        # Mermaid graph — collapsible so it can take full space when expanded
        if agents or route_type == "PIPELINE":
            st.markdown(
                _section_header_html(
                    "Execution Trace",
                    "Pipeline details",
                    "Collapsed panels below preserve the technical path the system actually executed.",
                ),
                unsafe_allow_html=True,
            )
            with st.expander("\U0001f5fa\ufe0f Pipeline Graph", expanded=True):
                mermaid_code = _build_reasonflow_mermaid(
                    delegation_plan=delegation_plan,
                    route_type=route_type,
                    active_node=None,
                    completed_nodes=executed_nodes,
                    retry_count=retry_count,
                    final_verdict=verdict,
                )
                html = _mermaid_html(mermaid_code)
                graph_height = _reasonflow_mermaid_iframe_height(len(delegation_plan or agents))
                components.html(html, height=graph_height, scrolling=True)

        # Agent details in collapsible panel
        if results:
            with st.expander("\U0001f916 Agent Details", expanded=False):
                # Delegation plan
                if delegation_plan:
                    st.markdown(
                        f'<div style="font-size:0.82rem;font-weight:600;color:#6C63FF;'
                        f'margin-bottom:4px;">'
                        f'\U0001f4cb Delegation Plan</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        _delegation_plan_html(delegation_plan),
                        unsafe_allow_html=True,
                    )

                # Agent result cards
                for res in results:
                    st.markdown(
                        _agent_done_card_html(
                            agent=res.get("agent", ""),
                            task=res.get("task", ""),
                            result=res.get("result", ""),
                            tools_used=res.get("tools_used", []),
                            confidence=res.get("confidence", 0.5),
                            execution_time=res.get("execution_time", 0),
                            error=(res.get("confidence", 0.5) == 0.0),
                            model_note=res.get("model_note", ""),
                        ),
                        unsafe_allow_html=True,
                    )

                # Reflection verdict
                feedback = pipeline_data.get("reflection_feedback", "")
                if feedback or verdict:
                    st.markdown(
                        _reflection_html(verdict, feedback),
                        unsafe_allow_html=True,
                    )
