"""Unit tests for the evaluation module."""

from unittest.mock import MagicMock, patch
import pytest

from evaluation.evaluations import (
    DATASET_NAME,
    JUDGE_MODEL_NAME,
    JUDGE_PROVIDER,
    TEST_CASES,
    _ensure_dataset,
    _get_judge_llm,
)


def test_test_cases_integrity():
    """Verify that test cases are well-formed and cover company policies."""
    assert len(TEST_CASES) == 10
    for case in TEST_CASES:
        assert "question" in case and isinstance(case["question"], str)
        assert "answer" in case and isinstance(case["answer"], str)
        assert len(case["question"].strip()) > 0
        assert len(case["answer"].strip()) > 0


def test_ensure_dataset_reuses_existing():
    """Verify that _ensure_dataset reuses an existing dataset when present."""
    mock_client = MagicMock()
    mock_client.has_dataset.return_value = True
    mock_existing_dataset = MagicMock()
    mock_client.read_dataset.return_value = mock_existing_dataset

    dataset = _ensure_dataset(mock_client)

    mock_client.has_dataset.assert_called_once_with(dataset_name=DATASET_NAME)
    mock_client.read_dataset.assert_called_once_with(dataset_name=DATASET_NAME)
    mock_client.create_dataset.assert_not_called()
    mock_client.create_examples.assert_not_called()
    assert dataset == mock_existing_dataset


def test_ensure_dataset_creates_new():
    """Verify that _ensure_dataset creates a new dataset and uploads examples if missing."""
    mock_client = MagicMock()
    mock_client.has_dataset.return_value = False
    mock_created_dataset = MagicMock()
    mock_created_dataset.id = "mock-dataset-id"
    mock_client.create_dataset.return_value = mock_created_dataset

    dataset = _ensure_dataset(mock_client)

    mock_client.has_dataset.assert_called_once_with(dataset_name=DATASET_NAME)
    mock_client.create_dataset.assert_called_once()
    mock_client.create_examples.assert_called_once()
    call_kwargs = mock_client.create_examples.call_args[1]
    assert call_kwargs["dataset_id"] == "mock-dataset-id"
    assert len(call_kwargs["examples"]) == len(TEST_CASES)
    assert dataset == mock_created_dataset


@patch("evaluation.evaluations.createHeaders")
@patch("evaluation.evaluations.ChatOpenAI")
def test_get_judge_llm(mock_chat_openai, mock_create_headers):
    """Verify that judge LLM is configured with Portkey fallback provider and judge model."""
    mock_create_headers.return_value = {"x-portkey-provider": JUDGE_PROVIDER}
    mock_chat_openai.return_value = MagicMock()

    judge = _get_judge_llm()

    mock_create_headers.assert_called_once()
    create_headers_kwargs = mock_create_headers.call_args[1]
    assert create_headers_kwargs["provider"] == "@hrpolicyfallback"
    mock_chat_openai.assert_called_once()
    kwargs = mock_chat_openai.call_args[1]
    assert kwargs["model"] == JUDGE_MODEL_NAME
    assert kwargs["temperature"] == 0.0
