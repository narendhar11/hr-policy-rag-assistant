"""Unit tests for retrieval/retriever.py"""

from unittest.mock import MagicMock
from retrieval.retriever import get_retriever
import config


def test_get_retriever_returns_retriever():
    """get_retriever should return the retriever from vector_store.as_retriever."""
    mock_vs = MagicMock()
    mock_retriever = MagicMock()
    mock_vs.as_retriever.return_value = mock_retriever

    result = get_retriever(mock_vs)
    assert result == mock_retriever


def test_get_retriever_uses_default_top_k():
    """get_retriever should call as_retriever with the configured top_k value."""
    mock_vs = MagicMock()
    get_retriever(mock_vs)
    mock_vs.as_retriever.assert_called_once_with(
        search_kwargs={"k": config.TOP_K_RESULTS}
    )


def test_get_retriever_uses_custom_top_k():
    """get_retriever should respect a custom top_k argument."""
    mock_vs = MagicMock()
    get_retriever(mock_vs, top_k=5)
    mock_vs.as_retriever.assert_called_once_with(search_kwargs={"k": 5})
