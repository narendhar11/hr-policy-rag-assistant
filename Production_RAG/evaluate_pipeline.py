"""Run the HR policy assistant evaluation and upload results to LangSmith.

Usage:
    python evaluate_pipeline.py
    # or with uv:
    uv run python evaluate_pipeline.py
"""

from evaluation.evaluations import run_evaluation
from utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """Execute the evaluation pipeline and display summary results."""
    logger.info("Running HR policy assistant evaluation pipeline...")
    print("Running HR policy assistant evaluation...")
    results = run_evaluation()
    print("Done. Open your LangSmith project to see the experiment.")
    print(results)
    logger.info("Evaluation pipeline completed successfully.")


if __name__ == "__main__":
    main()