"""Groq LLM client for ultra-low latency inference."""

from langchain_groq import ChatGroq
import config
from utils.logger import get_logger

logger = get_logger(__name__)


def get_llm():
    """Get the Groq LLM model and return it."""
    logger.info("Initializing LLM model '%s'", config.LLM_MODEL_NAME)
    return ChatGroq(
        model=config.LLM_MODEL_NAME,
        temperature=config.LLM_TEMPERATURE,
    )
