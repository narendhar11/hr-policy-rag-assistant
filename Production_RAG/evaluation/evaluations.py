"""Evaluation suite for HR Policy RAG Assistant.

Evaluates answer quality against a fixed set of test questions covering
Vertexon Solutions HR policies using LangSmith Datasets and Experiments.

Unlike tracing (which passively records runtime telemetry for live queries),
evaluation runs the agent against a curated set of question and reference-answer
pairs (a Dataset) and scores each response using a second LLM as a judge.
Results are saved as an Experiment in LangSmith for regression testing and
quality benchmarking across prompt, model, or guardrail iterations.

Two evaluators are used from openevals (LLM-as-judge):
1. Correctness: Checks if the generated answer accurately reflects the ground-truth
   reference answer from company HR policy.
2. Groundedness: Verifies that every claim in the answer is strictly supported by
   the context chunks retrieved from Qdrant, catching hallucinations that correctness
   might miss.

Quota Isolation Architecture:
To prevent '429 Too Many Requests' rate limit errors during evaluations, requests
are split across two Groq accounts via Portkey AI Gateway:
- Primary Account (@hrpolicy): Dedicated to the agent answering questions & guardrails.
- Secondary Account (@hrpolicyfallback): Dedicated exclusively to the judge LLM
  (e.g., openai/gpt-oss-120b) for scoring correctness and groundedness.
"""

import os
import sys
from pathlib import Path

# Ensure Production_RAG root directory is on the module search path
_BASE_DIR = Path(__file__).resolve().parent.parent
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

from langchain_openai import ChatOpenAI
from langsmith import Client
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT, RAG_GROUNDEDNESS_PROMPT
from portkey_ai import createHeaders, PORTKEY_GATEWAY_URL

import config
from utils.logger import get_logger
from pipeline import build_hr_policy_assistant, ask_hr_policy_question
from vectordb.vector_store import load_vector_store
from retrieval.retriever import get_retriever

logger = get_logger(__name__)

# ── Evaluation Dataset Configuration ──────────────────────────────────────────
DATASET_NAME = os.getenv("EVAL_DATASET_NAME", "hr-policy-evaluations")
JUDGE_MODEL_NAME = os.getenv("JUDGE_MODEL_NAME", "openai/gpt-oss-120b")

# Dedicated Portkey virtual key provider for the evaluation judge.
# Using @hrpolicyfallback isolates rate limits between the agent (Account 1: @hrpolicy)
# and the judge (Account 2: @hrpolicyfallback), avoiding 429 Too Many Requests errors.
JUDGE_PROVIDER = os.getenv("JUDGE_PORTKEY_PROVIDER", "@hrpolicyfallback")

# 10 diverse test cases covering all 8 sections of Vertexon Solutions HR policy:
# 1. Leave Policy (annual entitlement & carry forward)
# 2. Leave Policy (sick leave & medical certificate requirement)
# 3. Work From Home Policy (allowed days & core working hours)
# 4. Probation Period (duration & review milestones)
# 5. Notice Period (probation vs standard resignation notice)
# 6. Reimbursement Policy (submission deadline & processing turnaround)
# 7. Code of Conduct (mandatory training timeline)
# 8. Holidays (public & floating holidays allowance)
# 9. Holidays (compensation for working on public holidays)
# 10. Exit Policy (department clearance & final settlement timeline)
TEST_CASES = [
    {
        "question": "How many days of paid annual leave do full-time employees receive, and how many unused days can be carried forward?",
        "answer": "18 days of paid annual leave per calendar year, and up to a maximum of 8 unused days can be carried forward to the next year.",
    },
    {
        "question": "When is a medical certificate required if an employee takes sick leave?",
        "answer": "A medical certificate is required for sick leave longer than 3 consecutive days.",
    },
    {
        "question": "How many days per week can employees work from home, and what are the core working hours?",
        "answer": "Employees may work from home up to 3 days per week with manager approval, and must be reachable during core hours from 11 AM to 5 PM.",
    },
    {
        "question": "How long is the probation period for new joiners, and when are performance reviews held?",
        "answer": "The probation period is 6 months from the date of joining, with performance reviews at the 3-month and 6-month marks.",
    },
    {
        "question": "What is the resignation notice period during probation compared to after confirmation?",
        "answer": "The notice period is 30 days during probation and 60 days for confirmed employees.",
    },
    {
        "question": "What is the deadline for submitting expense reimbursement claims, and how long does processing take?",
        "answer": "Claims must be submitted with valid bills within 15 days of the expense, and are processed within 5 working days after manager approval.",
    },
    {
        "question": "Within what timeframe must new employees complete Code of Conduct and Data Privacy training?",
        "answer": "Within their first 30 days of joining, with a refresher required every year.",
    },
    {
        "question": "How many public and floating holidays does the company provide annually?",
        "answer": "10 public holidays per year as per the official holiday calendar, plus 2 floating holidays.",
    },
    {
        "question": "What compensation or benefit do employees get if they work on a public holiday?",
        "answer": "Employees working on a public holiday are eligible for compensatory leave or 2x pay.",
    },
    {
        "question": "Which departments are involved in exit clearance, and when is the full and final settlement processed?",
        "answer": "Exit clearance involves IT, Finance, Admin, and HR departments, and full and final settlement is processed within 30 days of the last working day.",
    },
]


def _get_judge_llm() -> ChatOpenAI:
    """Return an evaluator judge LLM routed through Portkey AI Gateway.

    Routes to the dedicated fallback Groq account (@hrpolicyfallback) to achieve
    Quota Isolation: the main agent pipeline uses the primary account (@hrpolicy),
    while the judge uses the secondary account, preventing '429 Too Many Requests'
    rate limit collisions during intensive evaluation runs.

    Uses an independent, higher-capacity judge model (JUDGE_MODEL_NAME, e.g.
    'openai/gpt-oss-120b') so the system does not grade its own outputs verbatim.

    Returns:
        ChatOpenAI: A LangChain ChatOpenAI client instance pointing to Portkey.
    """
    logger.info(
        "Initializing evaluation judge LLM via Portkey (provider: '%s', model: '%s') for rate limit isolation",
        JUDGE_PROVIDER,
        JUDGE_MODEL_NAME,
    )
    headers = createHeaders(
        api_key=config.PORTKEY_API_KEY,
        provider=JUDGE_PROVIDER,
    )
    return ChatOpenAI(
        api_key="portkey",
        base_url=PORTKEY_GATEWAY_URL,
        default_headers=headers,
        model=JUDGE_MODEL_NAME,
        temperature=0.0,
    )


def _ensure_dataset(client: Client):
    """Retrieve existing evaluation dataset from LangSmith or create it if not present.

    Checks if a dataset matching DATASET_NAME already exists in LangSmith.
    If found, reuses it to keep historical benchmarks aligned. If not found,
    creates the dataset and registers the curated test cases as examples.

    Args:
        client (Client): An authenticated LangSmith Client instance.

    Returns:
        Dataset: The LangSmith Dataset object.
    """
    logger.info("Checking LangSmith for dataset '%s'...", DATASET_NAME)
    if client.has_dataset(dataset_name=DATASET_NAME):
        logger.info("Dataset '%s' already exists in LangSmith; reusing it.", DATASET_NAME)
        return client.read_dataset(dataset_name=DATASET_NAME)

    logger.info(
        "Dataset '%s' not found. Creating new dataset with %d test cases...",
        DATASET_NAME,
        len(TEST_CASES),
    )
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Benchmark QA dataset for Vertexon Solutions HR Policy RAG Assistant",
    )
    client.create_examples(
        dataset_id=dataset.id,
        examples=[
            {
                "inputs": {"question": case["question"]},
                "outputs": {"answer": case["answer"]},
            }
            for case in TEST_CASES
        ],
    )
    logger.info(
        "Successfully created dataset '%s' (ID: %s) with %d examples.",
        DATASET_NAME,
        dataset.id,
        len(TEST_CASES),
    )
    return dataset


def run_evaluation():
    """Upload dataset (if needed) and run correctness + groundedness evaluations.

    Executes the full evaluation lifecycle:
    1. Authenticates with LangSmith and ensures the evaluation dataset is ready.
    2. Builds the HR assistant agent and loads the Qdrant retriever once for reuse.
    3. Configures LLM-as-judge evaluators for correctness and groundedness.
    4. Evaluates the agent against all test cases and uploads results to LangSmith.

    Returns:
        ExperimentResults: Summary of the evaluation experiment uploaded to LangSmith.
    """
    logger.info("Starting HR Policy evaluation workflow...")
    client = Client()
    dataset = _ensure_dataset(client)

    # Agent and retriever are built once and reused across all test cases,
    # avoiding redundant connections to Qdrant and LLM initialization.
    logger.info("Building HR Policy Assistant agent and Qdrant retriever...")
    agent = build_hr_policy_assistant()
    retriever = get_retriever(load_vector_store())

    def target(inputs: dict) -> dict:
        """Run a test question through the agent and retrieve context chunks.

        Invokes the real HR assistant pipeline (including guardrails and tools),
        and also queries the retriever directly to capture the exact context
        chunks needed for groundedness evaluation.

        Args:
            inputs (dict): Input dictionary containing the 'question' string.

        Returns:
            dict: Dictionary with 'answer' (generated string) and 'context' (retrieved chunks).
        """
        question = inputs["question"]
        logger.info("Evaluating test question: '%s'", question)
        try:
            answer = ask_hr_policy_question(agent, question)
        except Exception as e:
            logger.error("Failed to generate answer for '%s': %s", question, e)
            answer = f"Error during generation: {e}"

        try:
            chunks = retriever.invoke(question)
            context = "\n\n".join(chunk.page_content for chunk in chunks)
        except Exception as e:
            logger.error("Failed to retrieve context for '%s': %s", question, e)
            context = ""

        logger.debug(
            "Question: '%s' | Generated Answer length: %d chars | Retrieved %d chunk(s)",
            question,
            len(answer),
            len(chunks) if 'chunks' in locals() else 0,
        )
        return {"answer": answer, "context": context}

    # Setup LLM-as-judge evaluators
    judge_llm = _get_judge_llm()

    logger.info("Configuring correctness and groundedness evaluators via openevals...")
    correctness_judge = create_llm_as_judge(
        prompt=CORRECTNESS_PROMPT,
        feedback_key="correctness",
        judge=judge_llm,
    )

    groundedness_judge = create_llm_as_judge(
        prompt=RAG_GROUNDEDNESS_PROMPT,
        feedback_key="groundedness",
        judge=judge_llm,
    )

    def correctness_evaluator(inputs: dict, outputs: dict, reference_outputs: dict, **kwargs) -> dict:
        """Evaluate if the generated answer factually matches the reference answer."""
        if not outputs or not isinstance(outputs, dict) or "answer" not in outputs:
            return {"score": 0, "reasoning": "No valid answer generated by agent."}
        return correctness_judge(
            inputs=inputs,
            outputs={"answer": outputs["answer"]},
            reference_outputs=reference_outputs,
        )

    def groundedness_evaluator(outputs: dict, **kwargs) -> dict:
        """Evaluate if all claims in the generated answer are supported by retrieved context."""
        if not outputs or not isinstance(outputs, dict) or "answer" not in outputs:
            return {"score": 0, "reasoning": "No valid answer generated by agent."}
        return groundedness_judge(
            outputs={"answer": outputs["answer"]},
            context=outputs.get("context", ""),
        )

    experiment_prefix = "hr-policy-evals"
    logger.info(
        "Running LangSmith evaluation against dataset '%s' (experiment prefix: '%s')...",
        dataset.name,
        experiment_prefix,
    )

    results = client.evaluate(
        target,
        data=dataset.name,
        evaluators=[correctness_evaluator, groundedness_evaluator],
        experiment_prefix=experiment_prefix,
        description="Vertexon Solutions HR policy correctness and groundedness evaluation",
    )

    logger.info("Evaluation finished successfully. Results uploaded to LangSmith.")
    return results


if __name__ == "__main__":
    logger.info("Executing evaluation suite from CLI...")
    run_evaluation()