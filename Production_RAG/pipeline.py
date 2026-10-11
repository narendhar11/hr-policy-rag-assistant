"""End-to-end HR Policy assistant pipeline — wires all components together."""

import config
from utils.tracing import setup_langsmith_tracing
from ingestion.loader import load_documents
from chunking.chunker import split_into_chunks
from vectordb.vector_store import (
    build_vector_store,
    load_vector_store,
    vector_store_exists,
)
from retrieval.retriever import get_retriever
from llm.llm_client import get_llm
from tools.tools import search_hr_policy_tool
from agent.agent import create_hr_policy_agent
from llm_guard.guardrails import REFUSAL_MESSAGE, check_input_safety, check_output_safety
from utils.logger import get_logger

logger = get_logger(__name__)


def build_vector_store_for_documents(file_path: str = config.DATA_FILE_PATH):
    """Build or load the Qdrant cloud vector store. Loads documents, splits into chunks, and embeds into Qdrant."""
    if vector_store_exists():
        print("Qdrant collection already exists. Loading the existing vector store.")
        logger.info("Qdrant collection already exists. Loading existing vector store.")
        return load_vector_store()

    # Load documents
    logger.info("Qdrant collection not found. Building new vector store from documents.")
    documents = load_documents(file_path)
    # Split documents into chunks
    chunks = split_into_chunks(documents)
    logger.info("Number of chunks created: %d from %s", len(chunks), file_path)
    # Build and persist vector store in Qdrant
    vector_store = build_vector_store(chunks)
    logger.info("Qdrant vector store built and persisted to collection '%s'", config.QDRANT_COLLECTION_NAME)
    return vector_store


def build_hr_policy_assistant(file_path: str = config.DATA_FILE_PATH):
    """Build the HR policy assistant by creating the vector store, retriever, tools, and agent. ready to use."""
    setup_langsmith_tracing()     # Enable LangSmith tracing before any LangChain objects are created
    config.check_api_keys()       # Check if API keys are set

    # Build or load the vector store
    vector_store = build_vector_store_for_documents(file_path)

    # Create a retriever from the vector store
    retriever = get_retriever(vector_store)

    # Get the LLM model
    llm = get_llm()

    # Create tools for HR policy retrieval
    tools = [search_hr_policy_tool(retriever)]

    # Create an agent for HR policy retrieval
    agent = create_hr_policy_agent(llm, tools)
    logger.info("HR policy assistant built successfully")
    return agent


def ask_hr_policy_question(agent, question: str) -> str:
    """Ask a question to the HR policy assistant agent and return the answer."""
    logger.info("Asking HR policy question: '%s'", question)

    # Input safe guard - safe input check
    # safe or not, reason
    input_is_safe, _ = check_input_safety(question)
    if not input_is_safe:
        return REFUSAL_MESSAGE

    response = agent.invoke({"messages": [{"role": "user", "content": question}]})
    answer = response["messages"][-1].content
    # Output safe guard - safe output check
    output_is_safe,_ = check_output_safety(answer)
    if not output_is_safe:
        return REFUSAL_MESSAGE

    logger.info("Received answer:'%s", answer)
    return answer

