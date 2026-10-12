"""Create a retriever from the Qdrant vector store with optional neural reranking."""
from __future__ import annotations

from typing import Any, Callable
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

import config
from reranking.reranker import rerank_documents, JinaReranker, get_reranker
from utils.logger import get_logger

logger = get_logger(__name__)


class _RerankedRetriever(BaseRetriever):
    """Two-stage retriever engine: fetches candidates from vector store, then reranks with Jina."""

    vector_store: Any
    reranker: Any = None
    rerank_fn: Callable[..., list[Document]] | None = None
    candidate_k: int = config.CANDIDATE_K
    top_k: int = config.TOP_K_RESULTS

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        logger.info(
            "RerankedRetriever querying vector store with candidate_k=%d for query: '%s'",
            self.candidate_k,
            query,
        )
        candidates = []
        try:
            if hasattr(self.vector_store, "similarity_search"):
                candidates = self.vector_store.similarity_search(query, k=self.candidate_k)
            elif hasattr(self.vector_store, "as_retriever"):
                sub_retriever = self.vector_store.as_retriever(search_kwargs={"k": self.candidate_k})
                candidates = sub_retriever.invoke(query)
        except Exception as e:
            logger.error("Error retrieving candidates from vector store: %s", e)
            return []

        if not candidates:
            logger.info("No candidates found in vector store for query: '%s'", query)
            return []

        # Rerank via function or reranker object
        try:
            if self.rerank_fn:
                return self.rerank_fn(query, candidates, top_n=self.top_k)
            if self.reranker:
                if hasattr(self.reranker, "rerank_documents"):
                    return self.reranker.rerank_documents(query, candidates, top_n=self.top_k)
                if callable(self.reranker):
                    return self.reranker(query, candidates, top_n=self.top_k)
            # Default to pure function
            return rerank_documents(query, candidates, top_n=self.top_k)
        except Exception as e:
            logger.error("Reranking failed: %s. Falling back to top %d vector results.", e, self.top_k)

        return candidates[: self.top_k]


# Alias for backward-compatibility
RerankedRetriever = _RerankedRetriever


def get_retriever(vector_store, top_k: int = config.TOP_K_RESULTS):
    """Get the standard vector retriever (without reranking)."""
    logger.info("Creating standard vector retriever with top_k=%d", top_k)
    return vector_store.as_retriever(search_kwargs={"k": top_k})


def get_reranked_retriever(
    vector_store,
    top_k: int = config.TOP_K_RESULTS,
    candidate_k: int = config.CANDIDATE_K,
    reranker: Any = None,
    rerank_fn: Callable[..., list[Document]] | None = None,
) -> _RerankedRetriever:
    """Get a two-stage retriever with Jina neural reranking."""
    logger.info("Creating reranked retriever (candidate_k=%d, top_k=%d)", candidate_k, top_k)
    return _RerankedRetriever(
        vector_store=vector_store,
        reranker=reranker,
        rerank_fn=rerank_fn or (reranker.rerank_documents if hasattr(reranker, "rerank_documents") else (reranker if callable(reranker) else rerank_documents)),
        candidate_k=candidate_k,
        top_k=top_k,
    )
