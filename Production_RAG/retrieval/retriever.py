"""Create a retriever from the Qdrant vector store."""

import config
from utils.logger import get_logger

logger = get_logger(__name__)


def get_retriever(vector_store, top_k: int = config.TOP_K_RESULTS):
    """Get the retriever from the vector store. Returns top_k results for a given query."""
    logger.info("Creating retriever with top_k=%d", top_k)
    return vector_store.as_retriever(search_kwargs={"k": top_k})
