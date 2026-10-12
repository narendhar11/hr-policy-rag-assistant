"""Unit tests for reranking/reranker.py (function-based and wrapper)."""

from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from reranking.reranker import rerank, rerank_documents, JinaReranker, get_reranker


def test_rerank_empty_documents_function():
    """rerank() function should return an empty list when document list is empty."""
    assert rerank("test query", [], api_key="test_key") == []


def test_rerank_no_api_key_fallback_function():
    """rerank() function should fallback gracefully when no API key is provided."""
    docs = ["doc 1", "doc 2"]
    result = rerank("test query", docs, top_n=2, api_key="")
    assert len(result) == 2
    assert result[0]["index"] == 0


@patch("requests.post")
def test_rerank_function_success(mock_post):
    """rerank() function should parse API response and return reordered scores."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "results": [
            {"index": 1, "relevance_score": 0.95},
            {"index": 0, "relevance_score": 0.45},
        ]
    }
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    result = rerank("leave policy", ["doc A", "doc B"], top_n=2, api_key="test_key")

    assert len(result) == 2
    assert result[0]["index"] == 1
    assert result[0]["relevance_score"] == 0.95
    assert result[1]["index"] == 0
    assert result[1]["relevance_score"] == 0.45


@patch("requests.post")
def test_rerank_documents_function_reorders(mock_post):
    """rerank_documents() function should reorder Document objects and add rerank_score metadata."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "results": [
            {"index": 1, "relevance_score": 0.88},
            {"index": 0, "relevance_score": 0.22},
        ]
    }
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    doc_a = Document(page_content="Policy A", metadata={"id": "a"})
    doc_b = Document(page_content="Policy B", metadata={"id": "b"})

    reranked = rerank_documents("query", [doc_a, doc_b], top_n=2, api_key="test_key")

    assert len(reranked) == 2
    assert reranked[0].page_content == "Policy B"
    assert reranked[0].metadata["rerank_score"] == 0.88
    assert reranked[1].page_content == "Policy A"
    assert reranked[1].metadata["rerank_score"] == 0.22


@patch("requests.post")
def test_rerank_documents_function_error_fallback(mock_post):
    """rerank_documents() function should catch exceptions and fall back gracefully."""
    mock_post.side_effect = Exception("Network timeout")

    docs = [Document(page_content="Doc 1"), Document(page_content="Doc 2")]
    result = rerank_documents("query", docs, top_n=2, api_key="test_key")
    assert len(result) <= 2


def test_get_reranker_factory():
    """get_reranker factory function should return a JinaReranker instance."""
    reranker = get_reranker()
    assert isinstance(reranker, JinaReranker)


def test_reranker_init_and_custom_url():
    """JinaReranker wrapper should support custom configuration."""
    reranker = JinaReranker(api_key="test_key", model="jina-reranker-v3.5", rerank_url="https://custom.jina/v1/rerank")
    assert reranker.api_key == "test_key"
    assert reranker.model == "jina-reranker-v3.5"
    assert reranker.rerank_url == "https://custom.jina/v1/rerank"
