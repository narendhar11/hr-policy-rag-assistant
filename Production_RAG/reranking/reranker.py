"""Jina AI reranker model for neural cross-encoder candidate scoring."""

from langchain_core.documents import Document
import requests

import config
from utils.logger import get_logger

logger = get_logger(__name__)


def rerank(
    query: str,
    documents: list[str],
    top_n: int = None,
    api_key: str = None,
    model: str = None,
    rerank_url: str = None,
    timeout: int = 60,
) -> list[dict]:
    """Rerank text candidates for a query using Jina AI Reranker API and return scored indices.

    Args:
        query: The search question.
        documents: List of text strings to rank.
        top_n: Number of top results to return.
        api_key: Optional Jina API key (defaults to config.JINA_API_KEY).
        model: Optional model name (defaults to config.JINA_RERANKER_MODEL).
        rerank_url: Optional endpoint URL (defaults to config.JINA_RERANK_URL).
        timeout: Request timeout in seconds.

    Returns:
        List of dicts with keys: 'index' (original list position), 'relevance_score' (float).
    """
    if not documents:
        return []

    active_key = api_key or config.JINA_API_KEY
    active_model = model or config.JINA_RERANKER_MODEL
    active_url = rerank_url or config.JINA_RERANK_URL
    target_n = top_n or len(documents)

    if not active_key:
        logger.warning("No JINA_API_KEY configured; returning candidates in original order.")
        return [{"index": i, "relevance_score": 0.0} for i in range(min(target_n, len(documents)))]

    try:
        logger.info("Calling Jina Reranker (model: %s, candidates: %d, top_n: %d)", active_model, len(documents), target_n)
        response = requests.post(
            active_url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {active_key}",
            },
            json={
                "model": active_model,
                "query": query,
                "documents": documents,
                "top_n": target_n,
                "return_documents": False,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        results = [
            {"index": item["index"], "relevance_score": float(item["relevance_score"])}
            for item in payload.get("results", [])
        ]
        logger.info("Successfully reranked %d candidates", len(results))
        return results
    except Exception as e:
        logger.error("Jina rerank failed: %s. Falling back to original document order.", e)
        return [{"index": i, "relevance_score": 0.0} for i in range(min(target_n, len(documents)))]


def rerank_documents(
    query: str,
    documents: list[Document],
    top_n: int = None,
    api_key: str = None,
    model: str = None,
    rerank_url: str = None,
    timeout: int = 60,
) -> list[Document]:
    """Rerank LangChain Document objects and return top_n re-ordered documents."""
    if not documents:
        return []

    doc_texts = [doc.page_content for doc in documents]
    target_n = top_n or len(documents)
    scored = rerank(
        query=query,
        documents=doc_texts,
        top_n=target_n,
        api_key=api_key,
        model=model,
        rerank_url=rerank_url,
        timeout=timeout,
    )

    reranked_docs = []
    for item in scored:
        idx = item["index"]
        if 0 <= idx < len(documents):
            doc = documents[idx]
            doc.metadata["rerank_score"] = item["relevance_score"]
            reranked_docs.append(doc)

    return reranked_docs


# Backward-compatible class wrapper
class JinaReranker:
    """Wrapper class providing backward-compatibility for class-based consumers."""

    def __init__(
        self,
        api_key: str = None,
        model: str = None,
        rerank_url: str = None,
        timeout: int = 60,
    ):
        self.api_key = api_key or config.JINA_API_KEY
        self.model = model or config.JINA_RERANKER_MODEL
        self.rerank_url = rerank_url or config.JINA_RERANK_URL
        self.timeout = timeout

    def rerank(self, query: str, documents: list[str], top_n: int = None) -> list[dict]:
        return rerank(
            query=query,
            documents=documents,
            top_n=top_n,
            api_key=self.api_key,
            model=self.model,
            rerank_url=self.rerank_url,
            timeout=self.timeout,
        )

    def rerank_documents(
        self,
        query: str,
        documents: list[Document],
        top_n: int = None,
    ) -> list[Document]:
        return rerank_documents(
            query=query,
            documents=documents,
            top_n=top_n,
            api_key=self.api_key,
            model=self.model,
            rerank_url=self.rerank_url,
            timeout=self.timeout,
        )


def get_reranker() -> JinaReranker:
    """Get the Jina reranker model client and return it."""
    logger.info("Initializing Jina reranker with model '%s'", config.JINA_RERANKER_MODEL)
    return JinaReranker()
