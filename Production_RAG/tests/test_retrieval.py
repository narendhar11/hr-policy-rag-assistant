"""Unit tests for retrieval/retriever.py"""

from unittest.mock import MagicMock
from retrieval.retriever import get_retriever, get_reranked_retriever, RerankedRetriever
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


def test_get_reranked_retriever_creation():
    """get_reranked_retriever should return a RerankedRetriever instance with configured parameters."""
    mock_vs = MagicMock()
    mock_reranker = MagicMock()
    retriever = get_reranked_retriever(mock_vs, top_k=3, candidate_k=10, reranker=mock_reranker)
    assert isinstance(retriever, RerankedRetriever)
    assert retriever.top_k == 3
    assert retriever.candidate_k == 10
    assert retriever.reranker == mock_reranker


def test_reranked_retriever_invokes_and_reranks():
    """RerankedRetriever should fetch candidates from vector_store and pass them to reranker."""
    from langchain_core.documents import Document

    mock_vs = MagicMock()
    doc1 = Document(page_content="Candidate 1")
    doc2 = Document(page_content="Candidate 2")
    mock_vs.similarity_search.return_value = [doc1, doc2]

    mock_reranker = MagicMock()
    mock_reranker.rerank_documents.return_value = [doc2, doc1]

    retriever = RerankedRetriever(
        vector_store=mock_vs,
        reranker=mock_reranker,
        candidate_k=10,
        top_k=2,
    )

    results = retriever.invoke("What is the probation rule?")
    mock_vs.similarity_search.assert_called_once_with("What is the probation rule?", k=10)
    mock_reranker.rerank_documents.assert_called_once_with(
        "What is the probation rule?", [doc1, doc2], top_n=2
    )
    assert results == [doc2, doc1]


def test_reranked_retriever_handles_empty_candidates():
    """RerankedRetriever should return empty list if vector store returns nothing."""
    mock_vs = MagicMock()
    mock_vs.similarity_search.return_value = []
    mock_reranker = MagicMock()

    retriever = RerankedRetriever(vector_store=mock_vs, reranker=mock_reranker)
    results = retriever.invoke("nonexistent topic")

    assert results == []
    mock_reranker.rerank_documents.assert_not_called()
