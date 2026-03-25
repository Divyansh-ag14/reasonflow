import streamlit as st
import os
from src.langgraph_agentic_ai.ui.ui_config import Config

# ── Custom CSS ───────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
/* ── Global ───────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Sidebar ──────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #161B22 0%, #0E1117 100%);
    border-right: 1px solid rgba(108,99,255,0.15);
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stTextInput label {
    font-size: 0.8rem; font-weight: 600; letter-spacing: 0.02em;
    color: #8B949E; text-transform: uppercase;
}

/* ── Brand ────────────────────────────────────────────────────────────── */
.sidebar-brand { text-align: center; padding: 1rem 0 0.4rem; }
.sidebar-brand h2 {
    margin: 0; font-size: 1.3rem; font-weight: 800;
    background: linear-gradient(135deg, #6C63FF 0%, #48BFE3 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.sidebar-brand p {
    margin: 0.15rem 0 0; font-size: 0.68rem; color: #8B949E;
    letter-spacing: 0.08em; text-transform: uppercase;
}

/* ── Section label ────────────────────────────────────────────────────── */
.sidebar-section {
    font-size: 0.68rem; font-weight: 700; color: #6C63FF;
    letter-spacing: 0.1em; text-transform: uppercase;
    margin: 0.7rem 0 0.25rem; padding-bottom: 0.15rem;
    border-bottom: 1px solid rgba(108,99,255,0.2);
}

/* ── Mode toggle container ────────────────────────────────────────────── */
.mode-toggle {
    display: flex; gap: 6px; margin: 0.4rem 0 0.5rem;
}
.mode-btn {
    flex: 1; text-align: center; padding: 8px 6px; border-radius: 10px;
    font-size: 0.78rem; font-weight: 600; cursor: pointer;
    border: 1px solid #30363D; color: #8B949E; background: #0D1117;
    transition: all 0.2s;
}
.mode-btn.active {
    background: rgba(108,99,255,0.15); border-color: #6C63FF; color: #E6EDF3;
}

/* ── Architecture layer cards (sidebar) ───────────────────────────────── */
.layer-card {
    background: #0D1117; border: 1px solid #30363D; border-radius: 10px;
    padding: 8px 12px; margin: 4px 0; transition: all 0.2s;
}
.layer-card.active { border-color: #6C63FF; background: rgba(108,99,255,0.08); }
.layer-card .layer-num {
    font-size: 0.65rem; font-weight: 700; color: #6C63FF;
    letter-spacing: 0.08em; text-transform: uppercase;
}
.layer-card .layer-title {
    font-size: 0.85rem; font-weight: 600; color: #E6EDF3; margin: 1px 0;
}
.layer-card .layer-added {
    font-size: 0.72rem; color: #48BFE3; font-style: italic;
}

/* ── Tool pills ───────────────────────────────────────────────────────── */
.tool-pills { display: flex; flex-wrap: wrap; gap: 5px; margin: 0.3rem 0; }
.tool-pill {
    display: inline-flex; align-items: center; gap: 3px;
    background: rgba(108,99,255,0.1); border: 1px solid rgba(108,99,255,0.2);
    border-radius: 20px; padding: 3px 10px;
    font-size: 0.72rem; font-weight: 500; color: #C9D1D9;
}

/* ── Session badge ────────────────────────────────────────────────────── */
.session-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(108,99,255,0.06); border: 1px solid rgba(108,99,255,0.15);
    border-radius: 8px; padding: 5px 10px; margin: 0.2rem 0;
    font-size: 0.75rem; color: #8B949E; width: 100%;
}
.session-badge code { color: #6C63FF; font-weight: 600; font-size: 0.74rem; }

/* ── Main content ─────────────────────────────────────────────────────── */
.main .block-container { padding-top: 1.5rem; max-width: 920px; }

/* ── Welcome hero ─────────────────────────────────────────────────────── */
.welcome-hero { text-align: center; padding: 2rem 1rem 1rem; }
.welcome-hero h1 {
    font-size: 2.6rem; font-weight: 800; margin-bottom: 0.2rem;
    background: linear-gradient(135deg, #6C63FF 0%, #48BFE3 50%, #6C63FF 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.welcome-hero .tagline {
    font-size: 1.05rem; color: #8B949E; max-width: 540px; margin: 0 auto; line-height: 1.6;
}

/* ── Architecture evolution timeline ──────────────────────────────────── */
.evo-timeline { max-width: 780px; margin: 1.5rem auto 0; }

.evo-step {
    display: flex; gap: 16px; align-items: flex-start;
    padding: 14px 18px; margin: 0;
    border-left: 3px solid #30363D; position: relative;
    transition: all 0.2s;
}
.evo-step.active { border-left-color: #6C63FF; background: rgba(108,99,255,0.04); border-radius: 0 12px 12px 0; }
.evo-step.completed { border-left-color: #10B981; }

.evo-badge {
    min-width: 36px; height: 36px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.9rem; font-weight: 700; flex-shrink: 0;
    position: relative; left: -28px; margin-right: -12px;
}
.evo-badge.completed { background: #10B981; color: #fff; }
.evo-badge.active { background: #6C63FF; color: #fff; box-shadow: 0 0 12px rgba(108,99,255,0.4); }
.evo-badge.upcoming { background: #21262D; color: #8B949E; border: 2px solid #30363D; }

.evo-body { flex: 1; }
.evo-title { font-size: 1rem; font-weight: 700; color: #E6EDF3; }
.evo-subtitle { font-size: 0.78rem; color: #8B949E; margin: 2px 0; }
.evo-added {
    display: inline-flex; align-items: center; gap: 4px;
    font-size: 0.73rem; color: #48BFE3; font-weight: 600;
    background: rgba(72,191,227,0.08); border: 1px solid rgba(72,191,227,0.15);
    border-radius: 6px; padding: 2px 8px; margin-top: 4px;
}

.evo-graph-label {
    font-size: 0.72rem; color: #6C63FF; margin-top: 6px; font-weight: 600;
    font-family: 'SF Mono', 'Fira Code', monospace; letter-spacing: 0.02em;
}

/* ── Connector arrow between steps ────────────────────────────────────── */
.evo-connector {
    border-left: 3px solid #30363D; margin-left: 0;
    padding: 0 0 0 24px; font-size: 0.72rem; color: #484F58;
    font-style: italic; height: 8px;
}
.evo-connector.done { border-left-color: #10B981; }

/* ── Feature grid (for ReasonFlow welcome) ────────────────────────────── */
.feature-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 1.5rem auto; max-width: 780px; }
.feature-card {
    background: #161B22; border: 1px solid #30363D; border-radius: 12px;
    padding: 1rem; text-align: center; transition: border-color 0.2s, transform 0.2s;
}
.feature-card:hover { border-color: #6C63FF; transform: translateY(-2px); }
.feature-card .f-icon { font-size: 1.8rem; margin-bottom: 0.3rem; }
.feature-card .f-title { font-size: 0.82rem; font-weight: 700; color: #E6EDF3; }
.feature-card .f-desc { font-size: 0.72rem; color: #8B949E; margin-top: 0.15rem; line-height: 1.4; }

/* ── Chat bubbles ─────────────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    border-radius: 14px; margin-bottom: 0.8rem;
    border: 1px solid #30363D; padding: 1rem 1.2rem;
}

/* ── Expander ─────────────────────────────────────────────────────────── */
details[data-testid="stExpander"] {
    border: 1px solid #30363D; border-radius: 12px;
    background: #161B22; margin-bottom: 0.6rem;
}
details[data-testid="stExpander"] summary span { font-weight: 600; font-size: 0.86rem; }

/* ── Status pill ──────────────────────────────────────────────────────── */
.status-pill {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 5px 14px; border-radius: 20px; font-size: 0.76rem; font-weight: 600;
}
.status-pill.running { background: rgba(245,158,11,0.12); color: #F59E0B; border: 1px solid rgba(245,158,11,0.25); }
.status-pill.done    { background: rgba(16,185,129,0.12); color: #10B981; border: 1px solid rgba(16,185,129,0.25); }

/* ── Tool result card ─────────────────────────────────────────────────── */
.tool-result-card {
    background: #0D1117; border: 1px solid #30363D; border-radius: 10px;
    padding: 0.7rem 1rem; margin: 0.4rem 0;
}
.tool-result-card .tool-name {
    font-size: 0.75rem; font-weight: 600; color: #6C63FF;
    margin-bottom: 0.2rem; text-transform: uppercase; letter-spacing: 0.03em;
}
.tool-result-card pre {
    font-size: 0.78rem; color: #C9D1D9; margin: 0;
    white-space: pre-wrap; word-break: break-word;
}

/* ── Plan steps ───────────────────────────────────────────────────────── */
.plan-step {
    display: flex; align-items: flex-start; gap: 10px;
    padding: 5px 0; font-size: 0.86rem; color: #C9D1D9;
}
.plan-step .num {
    min-width: 22px; height: 22px; border-radius: 50%;
    background: #6C63FF; color: #fff; display: flex;
    align-items: center; justify-content: center;
    font-size: 0.7rem; font-weight: 700; flex-shrink: 0;
}

/* ── Reflection card ──────────────────────────────────────────────────── */
.reflection-card {
    border-left: 3px solid; border-radius: 8px;
    padding: 0.6rem 1rem; margin: 0.5rem 0; font-size: 0.86rem;
}
.reflection-card.pass { border-color: #10B981; background: rgba(16,185,129,0.06); color: #A7F3D0; }
.reflection-card.retry { border-color: #F59E0B; background: rgba(245,158,11,0.06); color: #FDE68A; }

/* ── Prompt hint ──────────────────────────────────────────────────────── */
.prompt-hint {
    text-align: center; margin-top: 1rem; padding: 12px 20px;
    background: rgba(108,99,255,0.06); border: 1px solid rgba(108,99,255,0.15);
    border-radius: 12px; font-size: 0.88rem; color: #C9D1D9;
    max-width: 600px; margin-left: auto; margin-right: auto;
}
.prompt-hint strong { color: #6C63FF; }

/* ── Agent identity badges ────────────────────────────────────────────── */
.agent-badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 6px 14px; border-radius: 10px; margin: 4px 0;
    font-size: 0.8rem; font-weight: 600;
    border: 1px solid;
}
.agent-badge.researcher { background: rgba(72,191,227,0.08); border-color: rgba(72,191,227,0.25); color: #48BFE3; }
.agent-badge.coder { background: rgba(16,185,129,0.08); border-color: rgba(16,185,129,0.25); color: #10B981; }
.agent-badge.analyst { background: rgba(245,158,11,0.08); border-color: rgba(245,158,11,0.25); color: #F59E0B; }
.agent-badge.writer { background: rgba(155,89,182,0.08); border-color: rgba(155,89,182,0.25); color: #9B59B6; }
.agent-badge.planner { background: rgba(243,156,18,0.08); border-color: rgba(243,156,18,0.25); color: #F39C12; }
.agent-badge.fact_checker { background: rgba(231,76,60,0.08); border-color: rgba(231,76,60,0.25); color: #E74C3C; }
.agent-badge.math_solver { background: rgba(26,188,156,0.08); border-color: rgba(26,188,156,0.25); color: #1ABC9C; }
.agent-badge.visualizer { background: rgba(52,152,219,0.08); border-color: rgba(52,152,219,0.25); color: #3498DB; }
.agent-badge.critic { background: rgba(230,126,34,0.08); border-color: rgba(230,126,34,0.25); color: #E67E22; }

/* ── Agent result card ───────────────────────────────────────────────── */
.agent-result-card {
    background: #0D1117; border: 1px solid #30363D; border-radius: 12px;
    padding: 1rem 1.2rem; margin: 0.5rem 0;
}
.agent-result-card .agent-header {
    display: flex; align-items: center; gap: 8px; margin-bottom: 0.5rem;
}
.agent-result-card .agent-icon { font-size: 1.2rem; }
.agent-result-card .agent-name {
    font-size: 0.82rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.04em;
}
.agent-result-card .agent-name.researcher { color: #48BFE3; }
.agent-result-card .agent-name.coder { color: #10B981; }
.agent-result-card .agent-name.analyst { color: #F59E0B; }
.agent-result-card .agent-name.writer { color: #9B59B6; }
.agent-result-card .agent-name.planner { color: #F39C12; }
.agent-result-card .agent-name.fact_checker { color: #E74C3C; }
.agent-result-card .agent-name.math_solver { color: #1ABC9C; }
.agent-result-card .agent-name.visualizer { color: #3498DB; }
.agent-result-card .agent-name.critic { color: #E67E22; }
.agent-result-card .agent-task {
    font-size: 0.76rem; color: #8B949E; font-style: italic; margin-bottom: 0.4rem;
}
.agent-result-card .agent-tools {
    display: flex; gap: 5px; flex-wrap: wrap; margin-bottom: 0.4rem;
}
.agent-result-card .agent-tool-tag {
    font-size: 0.68rem; padding: 2px 8px; border-radius: 4px;
    background: rgba(108,99,255,0.1); color: #6C63FF; font-weight: 500;
}
.agent-result-card .agent-output {
    font-size: 0.82rem; color: #C9D1D9; line-height: 1.5;
    max-height: 200px; overflow-y: auto;
    white-space: pre-wrap; word-break: break-word;
}

/* ── HITL approval card ──────────────────────────────────────────────── */
.hitl-card {
    background: rgba(108,99,255,0.04); border: 2px solid #6C63FF;
    border-radius: 14px; padding: 1.2rem 1.5rem; margin: 1rem 0;
}
.hitl-card .hitl-title {
    font-size: 0.9rem; font-weight: 700; color: #E6EDF3; margin-bottom: 0.4rem;
}
.hitl-card .hitl-subtitle {
    font-size: 0.78rem; color: #8B949E; margin-bottom: 0.8rem;
}

/* ── Delegation plan display ─────────────────────────────────────────── */
.deleg-step {
    display: flex; align-items: flex-start; gap: 10px;
    padding: 6px 0; font-size: 0.84rem; color: #C9D1D9;
}
.deleg-agent-icon {
    min-width: 28px; height: 28px; border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.85rem; flex-shrink: 0;
}
.deleg-agent-icon.researcher { background: rgba(72,191,227,0.15); }
.deleg-agent-icon.coder { background: rgba(16,185,129,0.15); }
.deleg-agent-icon.analyst { background: rgba(245,158,11,0.15); }
.deleg-agent-icon.writer { background: rgba(155,89,182,0.15); }
.deleg-agent-icon.planner { background: rgba(243,156,18,0.15); }
.deleg-agent-icon.fact_checker { background: rgba(231,76,60,0.15); }
.deleg-agent-icon.math_solver { background: rgba(26,188,156,0.15); }
.deleg-agent-icon.visualizer { background: rgba(52,152,219,0.15); }
.deleg-agent-icon.critic { background: rgba(230,126,34,0.15); }
.deleg-task { flex: 1; }
.deleg-task strong { font-size: 0.78rem; text-transform: capitalize; }

/* ── Parallel dispatch indicator ─────────────────────────────────────── */
.parallel-badge {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 4px 12px; border-radius: 6px; margin: 0.3rem 0;
    font-size: 0.72rem; font-weight: 600;
    background: rgba(108,99,255,0.1); border: 1px solid rgba(108,99,255,0.2);
    color: #6C63FF;
}

/* ── Phase Progress Bar ──────────────────────────────────────────────── */
.phase-bar {
    display: flex; align-items: center;
    background: #0D1117; border: 1px solid #30363D; border-radius: 14px;
    padding: 14px 20px; margin: 0.8rem 0; gap: 0;
}
.phase-step {
    display: flex; align-items: center; gap: 8px;
    padding: 6px 14px; border-radius: 10px;
    font-size: 0.8rem; font-weight: 600; white-space: nowrap;
    transition: all 0.3s ease;
}
.phase-step.pending { color: #484F58; }
.phase-step.active {
    color: #E6EDF3;
    background: rgba(108,99,255,0.12);
    animation: phase-glow 2s ease-in-out infinite;
}
.phase-step.done { color: #10B981; }
.phase-step .phase-icon {
    width: 26px; height: 26px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.75rem; font-weight: 700;
}
.phase-step.pending .phase-icon {
    background: #21262D; border: 2px solid #30363D; color: #484F58;
}
.phase-step.active .phase-icon {
    background: #6C63FF; color: #fff;
    box-shadow: 0 0 14px rgba(108,99,255,0.5);
}
.phase-step.done .phase-icon {
    background: #10B981; color: #fff;
}
.phase-connector {
    flex: 1; height: 2px; background: #30363D;
    margin: 0 6px; min-width: 16px;
    transition: background 0.3s;
}
.phase-connector.done { background: #10B981; }
.phase-status {
    margin-left: auto; padding-left: 14px;
    font-size: 0.72rem; color: #8B949E; white-space: nowrap;
    font-style: italic;
}
@keyframes phase-glow {
    0%, 100% { box-shadow: none; }
    50% { box-shadow: 0 0 20px rgba(108,99,255,0.15); }
}

/* ── Agent Working Card (animated) ───────────────────────────────────── */
.agent-working-card {
    background: #0D1117; border: 1px solid #30363D; border-radius: 12px;
    padding: 1rem 1.2rem; margin: 0.5rem 0;
    border-left: 3px solid #30363D;
    position: relative; overflow: hidden;
}
.agent-working-card.researcher { border-left-color: #48BFE3; }
.agent-working-card.coder { border-left-color: #10B981; }
.agent-working-card.analyst { border-left-color: #F59E0B; }
.agent-working-card.writer { border-left-color: #9B59B6; }
.agent-working-card.planner { border-left-color: #F39C12; }
.agent-working-card.fact_checker { border-left-color: #E74C3C; }
.agent-working-card.math_solver { border-left-color: #1ABC9C; }
.agent-working-card.visualizer { border-left-color: #3498DB; }
.agent-working-card.critic { border-left-color: #E67E22; }
.agent-working-card::after {
    content: ''; position: absolute; top: 0; left: -100%;
    width: 100%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(108,99,255,0.05), transparent);
    animation: shimmer 2.5s ease-in-out infinite;
}
.agent-working-card .agent-header {
    display: flex; align-items: center; gap: 8px; margin-bottom: 0.5rem;
}
.agent-working-card .agent-icon { font-size: 1.2rem; }
.agent-working-card .agent-name {
    font-size: 0.82rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.04em;
}
.agent-working-card .agent-name.researcher { color: #48BFE3; }
.agent-working-card .agent-name.coder { color: #10B981; }
.agent-working-card .agent-name.analyst { color: #F59E0B; }
.agent-working-card .agent-name.writer { color: #9B59B6; }
.agent-working-card .agent-name.planner { color: #F39C12; }
.agent-working-card .agent-name.fact_checker { color: #E74C3C; }
.agent-working-card .agent-name.math_solver { color: #1ABC9C; }
.agent-working-card .agent-name.visualizer { color: #3498DB; }
.agent-working-card .agent-name.critic { color: #E67E22; }
.agent-working-card .agent-task {
    font-size: 0.76rem; color: #8B949E; font-style: italic; margin-bottom: 0.4rem;
}
.agent-working-card .working-badge {
    font-size: 0.6rem; font-weight: 700; letter-spacing: 0.08em;
    padding: 2px 10px; border-radius: 4px; margin-left: auto;
    background: rgba(108,99,255,0.15); color: #6C63FF;
    animation: badge-pulse 1.5s ease-in-out infinite;
}
.working-indicator {
    display: flex; align-items: center; gap: 10px;
    font-size: 0.76rem; color: #8B949E; margin-top: 0.4rem;
}
.working-dots {
    display: inline-flex; gap: 4px;
}
.working-dots span {
    width: 5px; height: 5px; border-radius: 50%; background: #6C63FF;
    animation: dot-bounce 1.4s ease-in-out infinite;
}
.working-dots span:nth-child(2) { animation-delay: 0.2s; }
.working-dots span:nth-child(3) { animation-delay: 0.4s; }
@keyframes shimmer {
    0% { left: -100%; }
    100% { left: 100%; }
}
@keyframes dot-bounce {
    0%, 80%, 100% { opacity: 0.3; transform: scale(0.8); }
    40% { opacity: 1; transform: scale(1.3); }
}
@keyframes badge-pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

/* ── Agent Done Card ─────────────────────────────────────────────────── */
.agent-done-card {
    background: #0D1117; border: 1px solid #30363D; border-radius: 12px;
    padding: 1rem 1.2rem; margin: 0.5rem 0;
    border-left: 3px solid #30363D;
    animation: card-appear 0.5s ease-out;
}
.agent-done-card.researcher { border-left-color: #48BFE3; }
.agent-done-card.coder { border-left-color: #10B981; }
.agent-done-card.analyst { border-left-color: #F59E0B; }
.agent-done-card.writer { border-left-color: #9B59B6; }
.agent-done-card.planner { border-left-color: #F39C12; }
.agent-done-card.fact_checker { border-left-color: #E74C3C; }
.agent-done-card.math_solver { border-left-color: #1ABC9C; }
.agent-done-card.visualizer { border-left-color: #3498DB; }
.agent-done-card.critic { border-left-color: #E67E22; }
.agent-done-card .agent-header {
    display: flex; align-items: center; gap: 8px; margin-bottom: 0.5rem;
}
.agent-done-card .agent-icon { font-size: 1.2rem; }
.agent-done-card .agent-name {
    font-size: 0.82rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.04em;
}
.agent-done-card .agent-name.researcher { color: #48BFE3; }
.agent-done-card .agent-name.coder { color: #10B981; }
.agent-done-card .agent-name.analyst { color: #F59E0B; }
.agent-done-card .agent-name.writer { color: #9B59B6; }
.agent-done-card .agent-name.planner { color: #F39C12; }
.agent-done-card .agent-name.fact_checker { color: #E74C3C; }
.agent-done-card .agent-name.math_solver { color: #1ABC9C; }
.agent-done-card .agent-name.visualizer { color: #3498DB; }
.agent-done-card .agent-name.critic { color: #E67E22; }
.agent-done-card .done-check {
    margin-left: auto; font-size: 0.72rem; font-weight: 600;
    color: #10B981; background: rgba(16,185,129,0.1);
    padding: 2px 10px; border-radius: 4px;
}
.agent-done-card .agent-task {
    font-size: 0.76rem; color: #8B949E; font-style: italic; margin-bottom: 0.4rem;
}
.agent-done-card .agent-tools {
    display: flex; gap: 5px; flex-wrap: wrap; margin-bottom: 0.4rem;
}
.agent-done-card .agent-tool-tag {
    font-size: 0.68rem; padding: 2px 8px; border-radius: 4px;
    background: rgba(108,99,255,0.1); color: #6C63FF; font-weight: 500;
}
.agent-done-card .agent-output {
    font-size: 0.78rem; color: #C9D1D9; line-height: 1.5;
    max-height: 150px; overflow-y: auto;
    white-space: pre-wrap; word-break: break-word;
}
.agent-done-card .agent-error {
    padding: 8px 12px; margin-bottom: 0.4rem; border-radius: 6px;
    font-size: 0.76rem; font-weight: 600;
    background: rgba(245,158,11,0.08); border: 1px solid rgba(245,158,11,0.2);
    color: #F59E0B;
}
@keyframes card-appear {
    0% { opacity: 0; transform: translateY(10px); }
    100% { opacity: 1; transform: translateY(0); }
}

/* ── Synthesis indicator ─────────────────────────────────────────────── */
.synthesis-indicator {
    display: flex; align-items: center; gap: 10px;
    padding: 12px 18px; margin: 0.5rem 0;
    background: rgba(108,99,255,0.04); border: 1px solid rgba(108,99,255,0.15);
    border-radius: 10px; font-size: 0.82rem; color: #C9D1D9;
    animation: phase-glow 2s ease-in-out infinite;
}
.synthesis-indicator .synth-icon { font-size: 1.2rem; }

/* ── Retry separator ─────────────────────────────────────────────────── */
.retry-separator {
    text-align: center; padding: 10px 0; margin: 0.8rem 0;
    font-size: 0.78rem; font-weight: 600; color: #F59E0B;
    border-top: 1px dashed rgba(245,158,11,0.3);
    border-bottom: 1px dashed rgba(245,158,11,0.3);
}

/* ── Completion summary badge ─────────────────────────────────────────── */
.completion-badge {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 6px 16px; border-radius: 20px; margin: 0.5rem 0;
    font-size: 0.76rem; font-weight: 600;
    background: rgba(16,185,129,0.08); border: 1px solid rgba(16,185,129,0.2);
    color: #10B981;
    animation: card-appear 0.5s ease-out;
}

/* ── Agent activity expander scroll ──────────────────────────────────── */
details[data-testid="stExpander"] [data-testid="stExpanderContent"] {
    max-height: 520px;
    overflow-y: auto;
    scrollbar-width: thin;
    scrollbar-color: #30363D transparent;
}
details[data-testid="stExpander"] [data-testid="stExpanderContent"]::-webkit-scrollbar {
    width: 5px;
}
details[data-testid="stExpander"] [data-testid="stExpanderContent"]::-webkit-scrollbar-thumb {
    background: #30363D; border-radius: 4px;
}

/* ── Mermaid pending node style ──────────────────────────────────────── */
.mermaid .pending text { fill: #484F58 !important; }

/* ── Footer ───────────────────────────────────────────────────────────── */
.app-footer {
    text-align: center; padding: 2rem 0 1rem; font-size: 0.73rem; color: #484F58;
}
.app-footer a { color: #6C63FF; text-decoration: none; }

/* ── 2026 UI refresh overrides ────────────────────────────────────────── */
:root {
    --rf-bg: #080b12;
    --rf-surface: rgba(14, 18, 28, 0.82);
    --rf-surface-strong: #111827;
    --rf-surface-soft: rgba(17, 24, 39, 0.68);
    --rf-border: rgba(148, 163, 184, 0.16);
    --rf-border-strong: rgba(108, 99, 255, 0.24);
    --rf-text: #e5eefb;
    --rf-muted: #94a3b8;
    --rf-accent: #7c6cff;
    --rf-accent-soft: rgba(124, 108, 255, 0.16);
    --rf-teal: #4cc9f0;
    --rf-shadow: 0 20px 45px rgba(0, 0, 0, 0.28);
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background:
        radial-gradient(circle at top center, rgba(76, 201, 240, 0.08), transparent 28%),
        radial-gradient(circle at top left, rgba(124, 108, 255, 0.12), transparent 24%),
        linear-gradient(180deg, #0a0f18 0%, #080b12 100%);
    color: var(--rf-text);
}

[data-testid="stAppViewContainer"] > .main {
    background: transparent;
}

.main .block-container {
    padding-top: 1.2rem;
    max-width: 1120px;
    padding-left: 1.4rem;
    padding-right: 1.4rem;
}

section[data-testid="stSidebar"] {
    background:
        radial-gradient(circle at top, rgba(124, 108, 255, 0.14), transparent 22%),
        linear-gradient(180deg, rgba(14, 18, 28, 0.98) 0%, rgba(9, 12, 18, 0.98) 100%);
    border-right: 1px solid rgba(124, 108, 255, 0.18);
    box-shadow: inset -1px 0 0 rgba(255, 255, 255, 0.02);
}

[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-testid="stTextInputRootElement"],
[data-testid="stSidebar"] [data-testid="stNumberInputContainer"] {
    background: rgba(15, 23, 42, 0.72);
    border: 1px solid var(--rf-border);
    border-radius: 14px;
    box-shadow: none;
}

[data-testid="stSidebar"] [data-baseweb="select"] input,
[data-testid="stSidebar"] input {
    color: var(--rf-text);
}

[data-testid="stSidebar"] .stCaptionContainer,
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] p {
    color: var(--rf-muted);
}

.sidebar-brand {
    text-align: left;
    padding: 0.4rem 0 0.8rem;
}

.sidebar-brand h2 {
    font-size: 1.45rem;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #9d91ff 0%, #4cc9f0 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.sidebar-section {
    font-size: 0.66rem;
    color: #9d91ff;
    letter-spacing: 0.14em;
    margin: 0.95rem 0 0.45rem;
    border-bottom: 1px solid rgba(157, 145, 255, 0.18);
}

.tool-pill,
.session-badge,
.layer-card,
.feature-card,
.prompt-hint,
.tool-result-card,
.agent-result-card,
.agent-working-card,
.agent-done-card,
.synthesis-indicator,
.hitl-card,
.phase-bar,
details[data-testid="stExpander"] {
    background: var(--rf-surface);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border: 1px solid var(--rf-border);
    box-shadow: var(--rf-shadow);
}

.tool-pill {
    border-radius: 999px;
    background: rgba(124, 108, 255, 0.1);
    border-color: rgba(124, 108, 255, 0.18);
    color: #dbe7ff;
    padding: 4px 11px;
}

.session-badge {
    border-radius: 14px;
    padding: 0.65rem 0.8rem;
    background: rgba(124, 108, 255, 0.08);
    border-color: rgba(124, 108, 255, 0.18);
}

.welcome-hero {
    padding: 1.1rem 0 0.8rem;
}

.welcome-hero h1 {
    font-size: clamp(2.5rem, 4vw, 3.5rem);
    letter-spacing: -0.05em;
    margin-bottom: 0.45rem;
    background: linear-gradient(135deg, #f5f7ff 0%, #9d91ff 45%, #4cc9f0 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.welcome-hero .tagline {
    color: #a8b6cc;
    max-width: 700px;
    font-size: 1.04rem;
    line-height: 1.75;
}

.feature-grid {
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 16px;
    max-width: 900px;
}

.feature-card {
    border-radius: 18px;
    padding: 1.15rem 1rem;
    background: linear-gradient(180deg, rgba(18, 25, 39, 0.9) 0%, rgba(12, 17, 28, 0.86) 100%);
    transition: transform 0.22s ease, border-color 0.22s ease, box-shadow 0.22s ease;
}

.feature-card:hover {
    transform: translateY(-4px);
    border-color: rgba(124, 108, 255, 0.32);
    box-shadow: 0 24px 44px rgba(0, 0, 0, 0.35);
}

.feature-card .f-title {
    font-size: 0.9rem;
    margin-top: 0.1rem;
}

.feature-card .f-desc {
    font-size: 0.76rem;
    color: #9fb0c7;
}

.prompt-hint {
    border-radius: 18px;
    padding: 14px 18px;
    max-width: 720px;
    background: linear-gradient(180deg, rgba(124, 108, 255, 0.08) 0%, rgba(76, 201, 240, 0.05) 100%);
    border-color: rgba(124, 108, 255, 0.18);
}

[data-testid="stChatMessage"] {
    border-radius: 20px;
    padding: 1.15rem 1.25rem;
    background: linear-gradient(180deg, rgba(15, 23, 42, 0.78) 0%, rgba(10, 15, 24, 0.84) 100%);
    border: 1px solid rgba(148, 163, 184, 0.12);
    box-shadow: 0 18px 36px rgba(0, 0, 0, 0.22);
}

[data-testid="stChatMessageContent"] p,
[data-testid="stChatMessageContent"] li {
    color: #dbe7ff;
    line-height: 1.72;
}

[data-testid="stChatMessageContent"] h1,
[data-testid="stChatMessageContent"] h2,
[data-testid="stChatMessageContent"] h3 {
    letter-spacing: -0.02em;
    margin-top: 0.2rem;
}

details[data-testid="stExpander"] {
    border-radius: 18px;
    margin: 0.85rem 0;
    overflow: hidden;
}

details[data-testid="stExpander"] summary {
    padding-top: 0.2rem;
    padding-bottom: 0.2rem;
}

details[data-testid="stExpander"] summary span {
    font-size: 0.92rem;
    font-weight: 700;
}

.app-footer {
    color: #6f819d;
    padding-top: 1.4rem;
}

.app-footer a {
    color: #9d91ff;
}

.rf-section {
    margin: 0.9rem 0 0.45rem;
}

.rf-section-kicker {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 10px;
    border-radius: 999px;
    background: rgba(124, 108, 255, 0.12);
    border: 1px solid rgba(124, 108, 255, 0.18);
    color: #b8acff;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

.rf-section-title {
    margin: 0.5rem 0 0.1rem;
    font-size: 1.15rem;
    font-weight: 700;
    color: #f2f6ff;
    letter-spacing: -0.03em;
}

.rf-section-subtitle {
    margin: 0;
    font-size: 0.84rem;
    line-height: 1.6;
    color: #8fa2bf;
}

@media (max-width: 980px) {
    .feature-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}

@media (max-width: 640px) {
    .main .block-container {
        padding-left: 0.8rem;
        padding-right: 0.8rem;
    }

    .feature-grid {
        grid-template-columns: 1fr;
    }

    .welcome-hero h1 {
        font-size: 2.2rem;
    }

    [data-testid="stChatMessage"] {
        padding: 0.95rem 1rem;
    }
}
</style>
"""

# ── Architecture layers metadata ─────────────────────────────────────────
ARCHITECTURE_LAYERS = [
    {
        "num": 1,
        "usecase": "Basic Chatbot",
        "title": "Stateful Chat",
        "subtitle": "LLM with persistent memory",
        "added": "+ MemorySaver checkpointing",
        "graph": "START → 💬 Chatbot → END",
    },
    {
        "num": 2,
        "usecase": "Chatbot With Web",
        "title": "Tool-Augmented Agent",
        "subtitle": "ReAct loop with 5 tools",
        "added": "+ Web · Wikipedia · ArXiv · Finance · Calculator",
        "graph": "START → 🤖 Agent ⇔ 🔧 Tools → END",
    },
    {
        "num": 3,
        "usecase": "AI News",
        "title": "Domain Pipeline",
        "subtitle": "Autonomous news research & export",
        "added": "+ Autonomous queries · Markdown export",
        "graph": "START → 📰 Agent ⇔ 🔧 Tools → 💾 Save → END",
    },
    {
        "num": 4,
        "usecase": "ReasonFlow Agent",
        "title": "Multi-Agent System",
        "subtitle": "Supervisor + Parallel Specialists + HITL",
        "added": "+ Router · Supervisor · 9 Agents · depends_on chains · HITL · Reflection",
        "graph": "START → 🔀 Router → 📋 Supervisor → [9 Specialists] → 🧠 Synthesize → 🔍 Reflect → END",
    },
]

TOOL_PILLS = [
    ("🔍", "Web"), ("🦆", "DuckDuckGo"), ("📖", "Wiki"), ("📰", "ArXiv"),
    ("💰", "Finance"), ("🐍", "Python"), ("🔢", "Calc"),
    ("🌐", "Scraper"), ("🎬", "YouTube"), ("📊", "Charts"),
]


class LoadStreamlitUI:
    def __init__(self):
        self.config = Config()
        self.user_controls = {}

    def load_streamlit_ui(self):
        st.set_page_config(
            page_title="ReasonFlow — Agentic AI",
            page_icon="🧠",
            layout="wide",
            initial_sidebar_state="expanded",
        )
        st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
        st.session_state.IsFetchButtonClicked = False

        with st.sidebar:
            # ── Brand ──
            st.markdown(
                '<div class="sidebar-brand">'
                "<h2>🧠 ReasonFlow</h2>"
                "<p>Stateful Agentic AI</p>"
                "</div>",
                unsafe_allow_html=True,
            )
            st.markdown("---")

            # ── Mode selector ──
            st.markdown('<div class="sidebar-section">Mode</div>', unsafe_allow_html=True)
            mode = st.radio(
                "mode_radio",
                ["🧠 ReasonFlow Agent", "🔬 Architecture Demos"],
                index=0,
                label_visibility="collapsed",
                horizontal=True,
            )

            is_demo_mode = mode == "🔬 Architecture Demos"

            if is_demo_mode:
                # ── Architecture layer selector ──
                st.markdown('<div class="sidebar-section">Build Layer</div>', unsafe_allow_html=True)

                layer_options = [
                    "Layer 1 · Stateful Chat",
                    "Layer 2 · Tool Agent",
                    "Layer 3 · News Pipeline",
                ]
                selected_layer = st.radio(
                    "layer_radio", layer_options, index=0, label_visibility="collapsed",
                )

                layer_to_usecase = {
                    "Layer 1 · Stateful Chat": "Basic Chatbot",
                    "Layer 2 · Tool Agent": "Chatbot With Web",
                    "Layer 3 · News Pipeline": "AI News",
                }
                selected_usecase = layer_to_usecase[selected_layer]

                # Show what this layer adds
                layer_idx = layer_options.index(selected_layer)
                layer_info = ARCHITECTURE_LAYERS[layer_idx]
                st.markdown(
                    f'<div class="layer-card active">'
                    f'<div class="layer-num">LAYER {layer_info["num"]}</div>'
                    f'<div class="layer-title">{layer_info["title"]}</div>'
                    f'<div class="layer-added">{layer_info["added"]}</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                selected_usecase = "ReasonFlow Agent"

            self.user_controls["selected_usecase"] = selected_usecase

            # ── LLM Configuration ──
            st.markdown('<div class="sidebar-section">LLM</div>', unsafe_allow_html=True)
            llm_options = self.config.get_llm_options()
            self.user_controls["selected_llm"] = st.selectbox(
                "Provider", llm_options, label_visibility="collapsed",
            )

            # Read .env defaults (set by load_dotenv() in main.py)
            env_groq = os.environ.get("GROQ_API_KEY", "")
            env_openai = os.environ.get("OPENAI_API_KEY", "")
            env_tavily = os.environ.get("TAVILY_API_KEY", "")

            if self.user_controls["selected_llm"] == "Groq":
                model_options = self.config.get_groq_model_options()
                self.user_controls["selected_groq_model"] = st.selectbox("Model", model_options)
                groq_key = st.text_input(
                    "Groq API Key", value=env_groq, type="password", placeholder="gsk_...",
                )
                self.user_controls["GROQ_API_KEY"] = st.session_state["GROQ_API_KEY"] = groq_key
                if groq_key and not env_groq:
                    os.environ["GROQ_API_KEY"] = groq_key
                if not groq_key:
                    st.warning("⚠️ [Get a Groq key →](https://console.groq.com/keys)")
                elif env_groq:
                    st.caption("✅ Loaded from .env")

            elif self.user_controls["selected_llm"] == "OpenAI":
                model_options = self.config.get_openai_model_options()
                self.user_controls["selected_openai_model"] = st.selectbox("Model", model_options)
                openai_key = st.text_input(
                    "OpenAI API Key", value=env_openai, type="password", placeholder="sk-...",
                )
                self.user_controls["OPENAI_API_KEY"] = st.session_state["OPENAI_API_KEY"] = openai_key
                if openai_key and not env_openai:
                    os.environ["OPENAI_API_KEY"] = openai_key
                if not openai_key:
                    st.warning("⚠️ [Get an OpenAI key →](https://platform.openai.com/api-keys)")
                elif env_openai:
                    st.caption("✅ Loaded from .env")

            # ── Tavily key for tool use cases ──
            if selected_usecase in ("Chatbot With Web", "AI News", "ReasonFlow Agent"):
                tavily_key = st.text_input(
                    "Tavily API Key", value=env_tavily, type="password", placeholder="tvly-...",
                )
                os.environ["TAVILY_API_KEY"] = tavily_key
                self.user_controls["TAVILY_API_KEY"] = st.session_state["TAVILY_API_KEY"] = tavily_key
                if not tavily_key:
                    st.warning("⚠️ [Get a Tavily key →](https://app.tavily.com/home)")
                elif env_tavily:
                    st.caption("✅ Loaded from .env")

            # ── Tool pills ──
            if selected_usecase in ("ReasonFlow Agent", "Chatbot With Web"):
                st.markdown('<div class="sidebar-section">Tools</div>', unsafe_allow_html=True)
                pills = TOOL_PILLS if selected_usecase == "ReasonFlow Agent" else TOOL_PILLS[:4]
                html = '<div class="tool-pills">'
                for icon, name in pills:
                    html += f'<span class="tool-pill">{icon} {name}</span>'
                html += "</div>"
                st.markdown(html, unsafe_allow_html=True)

            # ── Multi-agent settings (ReasonFlow only) ──
            if selected_usecase == "ReasonFlow Agent":
                st.markdown('<div class="sidebar-section">Agents</div>', unsafe_allow_html=True)
                agent_pills_html = (
                    '<div class="tool-pills">'
                    '<span class="tool-pill" style="border-color:rgba(72,191,227,0.3);color:#48BFE3;">🔬 Researcher</span>'
                    '<span class="tool-pill" style="border-color:rgba(16,185,129,0.3);color:#10B981;">💻 Coder</span>'
                    '<span class="tool-pill" style="border-color:rgba(245,158,11,0.3);color:#F59E0B;">📊 Analyst</span>'
                    '<span class="tool-pill" style="border-color:rgba(155,89,182,0.3);color:#9B59B6;">✍️ Writer</span>'
                    '<span class="tool-pill" style="border-color:rgba(243,156,18,0.3);color:#F39C12;">📋 Planner</span>'
                    '<span class="tool-pill" style="border-color:rgba(231,76,60,0.3);color:#E74C3C;">🔎 Fact-Checker</span>'
                    '<span class="tool-pill" style="border-color:rgba(26,188,156,0.3);color:#1ABC9C;">🔢 Math Solver</span>'
                    '<span class="tool-pill" style="border-color:rgba(52,152,219,0.3);color:#3498DB;">📈 Visualizer</span>'
                    '<span class="tool-pill" style="border-color:rgba(230,126,34,0.3);color:#E67E22;">🎯 Critic</span>'
                    '</div>'
                )
                st.markdown(agent_pills_html, unsafe_allow_html=True)

                st.markdown('<div class="sidebar-section">Human-in-the-Loop</div>', unsafe_allow_html=True)
                auto_approve = st.toggle("Auto-approve plans", value=True, help="When OFF, you must approve the delegation plan before agents execute.")
                st.session_state.auto_approve = auto_approve
                if not auto_approve:
                    st.caption("You will review agent plans before execution.")

            # ── AI News settings ──
            if selected_usecase == "AI News":
                st.markdown('<div class="sidebar-section">News</div>', unsafe_allow_html=True)
                time_frame = st.selectbox("Time Frame", ["Daily", "Weekly", "Monthly"], index=0)
                if st.button("🔍 Fetch AI News", width="stretch", type="primary"):
                    st.session_state.IsFetchButtonClicked = True
                    st.session_state.timeframe = time_frame

            # ── Session ──
            st.markdown("---")
            tid = st.session_state.get("thread_id", "N/A")
            st.markdown(
                f'<div class="session-badge">🧵 Session <code>{tid[:8]}</code></div>',
                unsafe_allow_html=True,
            )
            if st.button("🔄  New Conversation", width="stretch"):
                st.session_state.new_conversation = True
                st.rerun()

        return self.user_controls
