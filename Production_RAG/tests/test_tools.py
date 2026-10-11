"""Unit tests for tools/tools.py"""

from unittest.mock import MagicMock
from langchain_core.documents import Document
from tools.tools import search_hr_policy_tool


def test_search_hr_policy_tool_returns_callable():
    """search_hr_policy_tool should return an invokable LangChain tool."""
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = []
    tool = search_hr_policy_tool(mock_retriever)
    assert hasattr(tool, "invoke")


def test_search_hr_policy_tool_calls_retriever():
    """The tool should invoke the retriever with the question."""
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = []
    tool = search_hr_policy_tool(mock_retriever)
    tool.invoke("What is the leave policy?")
    mock_retriever.invoke.assert_called_once_with("What is the leave policy?")


def test_search_hr_policy_tool_joins_chunks():
    """The tool should join all chunk page_content with double newlines."""
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [
        Document(page_content="Chunk one."),
        Document(page_content="Chunk two."),
    ]
    tool = search_hr_policy_tool(mock_retriever)
    result = tool.invoke("remote work")
    assert "Chunk one." in result
    assert "Chunk two." in result
    assert "\n\n" in result


def test_search_hr_policy_tool_empty_results():
    """The tool should return an empty string when retriever finds nothing."""
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = []
    tool = search_hr_policy_tool(mock_retriever)
    result = tool.invoke("unknown query")
    assert result == ""
