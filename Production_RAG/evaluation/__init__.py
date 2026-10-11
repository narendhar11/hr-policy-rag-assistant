"""Evaluation module for HR Policy RAG Assistant.

Provides dataset management and LLM-as-judge evaluation workflows using
LangSmith and openevals to evaluate correctness and groundedness.
"""

from evaluation.evaluations import (
    DATASET_NAME,
    JUDGE_MODEL_NAME,
    JUDGE_PROVIDER,
    TEST_CASES,
    run_evaluation,
)

__all__ = [
    "DATASET_NAME",
    "JUDGE_MODEL_NAME",
    "JUDGE_PROVIDER",
    "TEST_CASES",
    "run_evaluation",
]
