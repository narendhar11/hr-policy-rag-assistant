"""LLM client routed through Portkey AI Gateway."""

import config
from llm_gateway.gateway import get_gateway_llm
from utils.logger import get_logger

logger = get_logger(__name__)


def get_llm():
    """Get the LLM model routed through Portkey gateway."""
    logger.info("Initializing LLM client via Portkey gateway (config: '%s')", config.PORTKEY_CONFIG_SLUG)
    return get_gateway_llm()
