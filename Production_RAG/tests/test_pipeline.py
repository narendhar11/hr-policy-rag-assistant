"""Basic smoke tests for the RAG pipeline components."""

import config

def test_config_loads_api_keys():
    """Verify that configuration loads without crashing (assumes .env is present)."""
    assert hasattr(config, "GROQ_API_KEY")
    assert hasattr(config, "JINA_API_KEY")

def test_chunking_config():
    """Verify that chunking settings are positive integers."""
    assert isinstance(config.CHUNK_SIZE, int)
    assert config.CHUNK_SIZE > 0
    assert isinstance(config.CHUNK_OVERLAP, int)
    assert config.CHUNK_OVERLAP > 0

def test_system_prompt_exists():
    """Verify that the system prompt is defined."""
    assert isinstance(config.SYSTEM_PROMPT, str)
    assert len(config.SYSTEM_PROMPT) > 10
