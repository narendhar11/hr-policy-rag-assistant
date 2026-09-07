"""Jina AI embeddings model for semantic vector representations."""

from langchain_community.embeddings import JinaEmbeddings
import config


def get_embeddings_model():
    """Get the Jina embeddings model and return it."""
    return JinaEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)  # type: ignore
