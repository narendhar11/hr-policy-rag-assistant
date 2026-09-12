"""Qdrant cloud vector store — build, load, and existence check."""


import config
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from embeddings.embedder import get_embeddings_model
from utils.logger import get_logger

logger = get_logger(__name__)


def build_vector_store(chunks):
    """Build and persist a Qdrant vector store from document chunks."""
    logger.info("Initializing Qdrant cloud vector store")
    embeddings_model = get_embeddings_model()
    vector_store = QdrantVectorStore.from_documents(
        chunks,
        embedding= embeddings_model,
        url = config.QDRANT_URL,
        api_key = config.QDRANT_API_KEY,
        collection_name= config.QDRANT_COLLECTION_NAME
    )
    logger.info("Qdrant vector store built and persisted with %d chunks", len(chunks))
    return vector_store

def load_vector_store():
    """Load the vector store from the existing Qdrant cloud collection."""
    logger.info("Loading vector store from Qdrant collection '%s'", config.QDRANT_COLLECTION_NAME)
    embeddings_model = get_embeddings_model()
    return QdrantVectorStore.from_existing_collection(
        embedding=embeddings_model,
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
        collection_name=config.QDRANT_COLLECTION_NAME,
    )


def vector_store_exists() -> bool:
    """Check if the Qdrant collection exists in the cloud."""
    client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
    return client.collection_exists(config.QDRANT_COLLECTION_NAME)
