"""LangSmith observability and tracing setup.

Configures LangChain tracing environment variables so every LLM call,
retriever query, and agent action is tracked and inspectable in LangSmith.
"""

import os
import config
from utils.logger import get_logger

logger = get_logger(__name__)


def setup_langsmith_tracing() -> bool:
    """Configure LangChain environment variables to enable LangSmith tracing.

    Returns:
        bool: True if tracing was successfully configured, False otherwise.
    """
    if not config.LANGSMITH_TRACING:
        logger.info("LangSmith tracing is disabled (LANGSMITH_TRACING is not set to true).")
        return False

    if not config.LANGSMITH_API_KEY:
        logger.warning(
            "LANGSMITH_TRACING is true, but LANGSMITH_API_KEY is missing. Tracing skipped."
        )
        return False

    # Set the standard LangChain tracing environment variables
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = config.LANGSMITH_ENDPOINT
    os.environ["LANGCHAIN_API_KEY"] = config.LANGSMITH_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = config.LANGSMITH_PROJECT

    logger.info(
        "LangSmith tracing enabled successfully! Project: '%s' | Dashboard: %s",
        config.LANGSMITH_PROJECT,
        config.LANGSMITH_ENDPOINT,
    )
    return True


def is_tracing_enabled() -> bool:
    """Check if LangChain tracing is currently active in the environment."""
    return os.environ.get("LANGCHAIN_TRACING_V2", "").lower() == "true"
