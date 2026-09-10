"""Unit tests for chunking/chunker.py"""

import pytest
from langchain_core.documents import Document
from chunking.chunker import split_into_chunks


def _make_docs(text: str):
    """Helper: wrap a string into a list of Documents."""
    return [Document(page_content=text, metadata={"source": "test"})]


def test_split_returns_list():
    """split_into_chunks should return a list."""
    docs = _make_docs("Hello world. " * 100)
    chunks = split_into_chunks(docs)
    assert isinstance(chunks, list)


def test_split_produces_chunks():
    """A long document should be split into more than one chunk."""
    long_text = "This is a sentence about HR policies. " * 200
    docs = _make_docs(long_text)
    chunks = split_into_chunks(docs)
    assert len(chunks) > 1


def test_each_chunk_is_document():
    """Every chunk should be a LangChain Document."""
    docs = _make_docs("Remote work policy. " * 100)
    chunks = split_into_chunks(docs)
    for chunk in chunks:
        assert isinstance(chunk, Document)


def test_chunk_content_is_non_empty():
    """No chunk should have empty page_content."""
    docs = _make_docs("Leave policy details. " * 100)
    chunks = split_into_chunks(docs)
    for chunk in chunks:
        assert chunk.page_content.strip() != ""


def test_chunk_size_respects_config():
    """Each chunk's content should not exceed chunk_size by a large margin."""
    import config
    docs = _make_docs("A" * 5000)
    chunks = split_into_chunks(docs)
    for chunk in chunks:
        assert len(chunk.page_content) <= config.CHUNK_SIZE + config.CHUNK_OVERLAP


def test_split_empty_document():
    """Splitting an empty document should return an empty list."""
    docs = _make_docs("")
    chunks = split_into_chunks(docs)
    assert chunks == []
