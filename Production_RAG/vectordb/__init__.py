"""Vector database module — FAISS build, save, and load operations."""
from .vector_store import (
    build_vector_store,
    save_vector_store,
    load_vector_store,
    vector_store_exists,
)

__all__ = [
    "build_vector_store",
    "save_vector_store",
    "load_vector_store",
    "vector_store_exists",
]
