import os
import streamlit as st
from langchain_openai import ChatOpenAI


class OpenAILLM:
    """OpenAI LLM provider — mirrors GroqLLM interface."""

    def __init__(self, user_contols_input):
        self.user_controls_input = user_contols_input

    def get_llm_model(self):
        try:
            api_key = self.user_controls_input.get("OPENAI_API_KEY", "")
            model = self.user_controls_input.get("selected_openai_model", "gpt-4o-mini")

            if not api_key and not os.environ.get("OPENAI_API_KEY", ""):
                st.error("Please enter your OpenAI API key.")
                return None

            llm = ChatOpenAI(api_key=api_key, model=model)
        except Exception as e:
            raise ValueError(f"Error creating OpenAI model: {e}")
        return llm
