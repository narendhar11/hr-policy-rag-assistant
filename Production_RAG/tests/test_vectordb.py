"""Unit tests for vectordb/vector_store.py — Qdrant cloud backend."""

from unittest.mock import patch, MagicMock


def test_vector_store_exists_returns_true():
    """vector_store_exists should return True when the Qdrant collection exists."""
    with patch("vectordb.vector_store.QdrantClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = True
        mock_client_cls.return_value = mock_client

        from vectordb.vector_store import vector_store_exists
        assert vector_store_exists() is True
        mock_client.collection_exists.assert_called_once()


def test_vector_store_exists_returns_false():
    """vector_store_exists should return False when the Qdrant collection does not exist."""
    with patch("vectordb.vector_store.QdrantClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = False
        mock_client_cls.return_value = mock_client

        from vectordb.vector_store import vector_store_exists
        assert vector_store_exists() is False


def test_load_vector_store_calls_from_existing_collection():
    """load_vector_store should call QdrantVectorStore.from_existing_collection."""
    with patch("vectordb.vector_store.QdrantVectorStore") as mock_qdrant, \
         patch("vectordb.vector_store.get_embeddings_model") as mock_embeddings:
        mock_embeddings.return_value = MagicMock()

        from vectordb.vector_store import load_vector_store
        load_vector_store()

        mock_qdrant.from_existing_collection.assert_called_once()
        _, kwargs = mock_qdrant.from_existing_collection.call_args
        assert "embedding" in kwargs
        assert "url" in kwargs
        assert "collection_name" in kwargs


def test_build_vector_store_calls_from_documents():
    """build_vector_store should call QdrantVectorStore.from_documents with chunks."""
    with patch("vectordb.vector_store.QdrantVectorStore") as mock_qdrant, \
         patch("vectordb.vector_store.get_embeddings_model") as mock_embeddings:
        mock_embeddings.return_value = MagicMock()
        mock_chunks = [MagicMock(), MagicMock()]

        from vectordb.vector_store import build_vector_store
        build_vector_store(mock_chunks)

        mock_qdrant.from_documents.assert_called_once()
        args, kwargs = mock_qdrant.from_documents.call_args
        assert args[0] == mock_chunks
        assert "embedding" in kwargs
        assert "url" in kwargs
        assert "collection_name" in kwargs
