"""Jina AI embeddings model for semantic vector representations."""

from langchain_community.embeddings import JinaEmbeddings
import config
from utils.logger import get_logger

logger = get_logger(__name__)


def get_embeddings_model():
    """Get the Jina embeddings model and return it."""
    logger.info("Initializing embeddings model '%s'", config.EMBEDDING_MODEL_NAME)
    return JinaEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)  # type: ignore
