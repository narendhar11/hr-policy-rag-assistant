"""Unit tests for utils/tracing.py"""

import os
from unittest.mock import patch
from utils.tracing import setup_langsmith_tracing, is_tracing_enabled


def test_setup_tracing_disabled_when_flag_false():
    """When LANGSMITH_TRACING is False, setup_langsmith_tracing returns False."""
    with patch("config.LANGSMITH_TRACING", False):
        result = setup_langsmith_tracing()
        assert result is False


def test_setup_tracing_disabled_when_no_api_key():
    """When LANGSMITH_TRACING is True but API key is missing, setup returns False."""
    with patch("config.LANGSMITH_TRACING", True), \
         patch("config.LANGSMITH_API_KEY", None):
        result = setup_langsmith_tracing()
        assert result is False


def test_setup_tracing_sets_env_vars_when_valid():
    """When enabled and API key is present, correct environment variables are set."""
    with patch("config.LANGSMITH_TRACING", True), \
         patch("config.LANGSMITH_API_KEY", "test-key-123"), \
         patch("config.LANGSMITH_PROJECT", "test-project"), \
         patch("config.LANGSMITH_ENDPOINT", "https://api.smith.test"):
        
        result = setup_langsmith_tracing()
        assert result is True
        assert os.environ.get("LANGCHAIN_TRACING_V2") == "true"
        assert os.environ.get("LANGCHAIN_API_KEY") == "test-key-123"
        assert os.environ.get("LANGCHAIN_PROJECT") == "test-project"
        assert os.environ.get("LANGCHAIN_ENDPOINT") == "https://api.smith.test"
        assert is_tracing_enabled() is True
