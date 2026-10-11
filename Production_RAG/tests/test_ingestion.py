"""Unit tests for ingestion/loader.py"""

import os
import tempfile
import pytest
from langchain_core.documents import Document
from ingestion.loader import load_documents


def test_load_documents_returns_list():
    """load_documents should return a list."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("Sample HR policy content.")
        tmp_path = f.name
    try:
        docs = load_documents(tmp_path)
        assert isinstance(docs, list)
    finally:
        os.unlink(tmp_path)


def test_load_documents_returns_document_objects():
    """Each item in the returned list should be a LangChain Document."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("Leave policy: 20 days per year.")
        tmp_path = f.name
    try:
        docs = load_documents(tmp_path)
        assert len(docs) == 1
        assert isinstance(docs[0], Document)
    finally:
        os.unlink(tmp_path)


def test_load_documents_content_matches_file():
    """Document page_content should match the file contents exactly."""
    content = "Remote work is allowed up to 3 days per week."
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(content)
        tmp_path = f.name
    try:
        docs = load_documents(tmp_path)
        assert docs[0].page_content == content
    finally:
        os.unlink(tmp_path)


def test_load_documents_metadata_has_source():
    """Document metadata should include the source file path."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("Benefits policy.")
        tmp_path = f.name
    try:
        docs = load_documents(tmp_path)
        assert "source" in docs[0].metadata
        assert docs[0].metadata["source"] == tmp_path
    finally:
        os.unlink(tmp_path)


def test_load_documents_raises_for_missing_file():
    """load_documents should raise FileNotFoundError for a missing file."""
    with pytest.raises(FileNotFoundError):
        load_documents("/non/existent/path/hr_policies.txt")
