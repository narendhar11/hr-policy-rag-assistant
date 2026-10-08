"""Portkey AI Gateway integration for resilient LLM routing and automatic fallback.

Provides gateway-wrapped ChatOpenAI clients for both the primary LLM
and the guardrail model using Portkey saved config slugs.
"""

from langchain_openai import ChatOpenAI
from portkey_ai import createHeaders, PORTKEY_GATEWAY_URL

import config
from utils.logger import get_logger

logger = get_logger(__name__)


def get_gateway_llm() -> ChatOpenAI:
    """Return a chat model routed through Portkey using a saved config slug."""
    logger.info("Routing LLM calls through Portkey with config '%s'", config.PORTKEY_CONFIG_SLUG)
    headers = createHeaders(
        api_key=config.PORTKEY_API_KEY,
        config=config.PORTKEY_CONFIG_SLUG,
    )
    return ChatOpenAI(
        api_key="portkey",
        base_url=PORTKEY_GATEWAY_URL,
        default_headers=headers,
    )


def get_gateway_guard_llm() -> ChatOpenAI:
    """Return a guard model routed through Portkey using a saved config slug."""
    logger.info("Routing Guard LLM calls through Portkey with config '%s'", config.PORTKEY_GUARD_CONFIG_SLUG)
    headers = createHeaders(
        api_key=config.PORTKEY_API_KEY,
        config=config.PORTKEY_GUARD_CONFIG_SLUG,
    )
    return ChatOpenAI(
        api_key="portkey",
        base_url=PORTKEY_GATEWAY_URL,
        default_headers=headers,
        model_kwargs={"response_format": {"type": "json_object"}},
        max_tokens=256,
    )
