"""Groq LLM client for ultra-low latency inference."""

from langchain_groq import ChatGroq
import config


def get_llm():
    """Get the Groq LLM model and return it."""
    return ChatGroq(
        model=config.LLM_MODEL_NAME,
        temperature=config.LLM_TEMPERATURE,
    )
