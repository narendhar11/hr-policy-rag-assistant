"""Create a retriever from the FAISS vector store."""

import config


def get_retriever(vector_store, top_k: int = config.TOP_K_RESULTS):
    """Get the retriever from the vector store. Returns top_k results for a given query."""
    return vector_store.as_retriever(search_kwargs={"k": top_k})
