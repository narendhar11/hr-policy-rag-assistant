"""FAISS vector store — build, save, load, and existence check."""

import os
from langchain_community.vectorstores import FAISS
import config
from embeddings.embedder import get_embeddings_model


def build_vector_store(chunks):
    """Build the vector store using FAISS from document chunks."""
    embeddings_model = get_embeddings_model()
    return FAISS.from_documents(chunks, embeddings_model)


def save_vector_store(vector_store, path: str = config.VECTOR_STORE_PATH) -> None:
    """Save the vector store to the specified path."""
    vector_store.save_local(path)


def load_vector_store(path: str = config.VECTOR_STORE_PATH):
    """Load the vector store from the specified path."""
    embeddings_model = get_embeddings_model()
    # Safely loading a local FAISS index that you created using the same embeddings model.
    return FAISS.load_local(path, embeddings_model, allow_dangerous_deserialization=True)


def vector_store_exists(path: str = config.VECTOR_STORE_PATH) -> bool:
    """Check if the vector store exists at the specified path."""
    return os.path.exists(os.path.join(path, "index.faiss"))
