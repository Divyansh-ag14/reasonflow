"""
LLM Fallback Utility
When a Groq model hits rate limits (429), automatically try smaller/faster models.
Provides visible logging via print() and returns model switch notes for UI display.
"""

import os
import time


# Ordered from most capable to lightest — tried in order after primary fails
GROQ_FALLBACK_MODELS = [
    "llama-3.1-8b-instant",
    "gemma2-9b-it",
]


def is_rate_limit_error(e: Exception) -> bool:
    """Check if an exception is a rate limit (429) error."""
    msg = str(e)
    return "429" in msg or "rate_limit" in msg.lower() or "rate limit" in msg.lower()


def _get_groq_fallbacks():
    """Create (ChatGroq, model_name) pairs for each fallback model."""
    try:
        from langchain_groq import ChatGroq
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            return []
        fallbacks = []
        for model_name in GROQ_FALLBACK_MODELS:
            try:
                fallbacks.append((ChatGroq(api_key=api_key, model=model_name), model_name))
            except Exception:
                continue
        return fallbacks
    except ImportError:
        return []


def invoke_with_fallback(primary_llm, messages, *, label="LLM", retries=2):
    """
    Invoke an LLM with retry + automatic model fallback on rate limit.

    Returns:
        (response, model_note)
        model_note is "" if primary succeeded, or describes the fallback used.
    """
    last_error = None

    # ── Try primary model with retries ────────────────────────────────
    for attempt in range(retries):
        try:
            response = primary_llm.invoke(messages)
            return response, ""
        except Exception as e:
            if is_rate_limit_error(e):
                wait = 2 * (attempt + 1)
                print(f"[{label}] Rate limited, retrying in {wait}s (attempt {attempt+1}/{retries})")
                time.sleep(wait)
                last_error = e
            else:
                raise

    # ── Primary exhausted — try fallback models ──────────────────────
    for fb_llm, fb_name in _get_groq_fallbacks():
        try:
            print(f"[{label}] Switching to fallback model: {fb_name}")
            response = fb_llm.invoke(messages)
            print(f"[{label}] Fallback {fb_name} succeeded")
            return response, f"Switched to {fb_name} (rate limit on primary model)"
        except Exception as e:
            if is_rate_limit_error(e):
                print(f"[{label}] Fallback {fb_name} also rate limited, trying next...")
                time.sleep(3)
                continue
            raise

    # ── All models exhausted ─────────────────────────────────────────
    raise last_error or Exception(
        f"All models rate limited for [{label}]. Please wait a moment and try again."
    )


def invoke_tools_with_fallback(primary_llm, tools, messages, *, label="LLM", retries=2):
    """
    Like invoke_with_fallback but for tool-bound LLMs.

    Returns:
        (response, new_llm_with_tools_or_None, model_note)
        new_llm_with_tools_or_None is set if a fallback was used (caller should
        switch to it for subsequent calls). None means primary succeeded.
    """
    llm_with_tools = primary_llm.bind_tools(tools) if tools else primary_llm
    last_error = None

    # ── Try primary model ─────────────────────────────────────────────
    for attempt in range(retries):
        try:
            response = llm_with_tools.invoke(messages)
            return response, None, ""
        except Exception as e:
            if is_rate_limit_error(e):
                wait = 2 * (attempt + 1)
                print(f"[{label}] Rate limited, retrying in {wait}s (attempt {attempt+1}/{retries})")
                time.sleep(wait)
                last_error = e
            else:
                raise

    # ── Fallback models ───────────────────────────────────────────────
    for fb_llm, fb_name in _get_groq_fallbacks():
        try:
            print(f"[{label}] Switching to fallback model: {fb_name}")
            fb_with_tools = fb_llm.bind_tools(tools) if tools else fb_llm
            response = fb_with_tools.invoke(messages)
            print(f"[{label}] Fallback {fb_name} succeeded")
            return response, fb_with_tools, f"Switched to {fb_name} (rate limit on primary model)"
        except Exception as e:
            if is_rate_limit_error(e):
                print(f"[{label}] Fallback {fb_name} also rate limited, trying next...")
                time.sleep(3)
                continue
            raise

    raise last_error or Exception(
        f"All models rate limited for [{label}]. Please wait a moment and try again."
    )
