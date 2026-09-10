"""Unit tests for vectordb/vector_store.py"""

import os
import tempfile
from unittest.mock import patch, MagicMock
from vectordb.vector_store import vector_store_exists, save_vector_store, load_vector_store


def test_vector_store_exists_returns_false_when_missing():
    """vector_store_exists should return False when the index file does not exist."""
    result = vector_store_exists("/non/existent/path")
    assert result is False


def test_vector_store_exists_returns_true_when_present():
    """vector_store_exists should return True when index.faiss file is present."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        open(os.path.join(tmp_dir, "index.faiss"), "w").close()
        assert vector_store_exists(tmp_dir) is True


def test_save_vector_store_calls_save_local():
    """save_vector_store should call vector_store.save_local with the correct path."""
    mock_vs = MagicMock()
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_vector_store(mock_vs, path=tmp_dir)
        mock_vs.save_local.assert_called_once_with(tmp_dir)


def test_load_vector_store_calls_load_local():
    """load_vector_store should call FAISS.load_local with the correct arguments."""
    with patch("vectordb.vector_store.FAISS") as mock_faiss, \
         patch("vectordb.vector_store.get_embeddings_model") as mock_embeddings:
        mock_embeddings.return_value = MagicMock()
        load_vector_store("/fake/path")
        mock_faiss.load_local.assert_called_once()
        args, kwargs = mock_faiss.load_local.call_args
        assert args[0] == "/fake/path"
        assert kwargs.get("allow_dangerous_deserialization") is True
