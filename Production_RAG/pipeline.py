"""End-to-end HR Policy assistant pipeline — wires all components together."""

import config
from ingestion.loader import load_documents
from chunking.chunker import split_into_chunks
from vectordb.vector_store import (
    build_vector_store,
    save_vector_store,
    load_vector_store,
    vector_store_exists,
)
from retrieval.retriever import get_retriever
from llm.llm_client import get_llm
from tools import search_hr_policy_tool
from agent import create_hr_policy_agent


def build_vector_store_for_documents(file_path: str = config.DATA_FILE_PATH):
    """Build the vector store for the documents and save it to the specified path. Load, split, embed, and save the vector store."""
    if vector_store_exists():
        print("Vector store already exists. Loading the existing vector store.")
        return load_vector_store()

    # Load documents
    documents = load_documents(file_path)
    # Split documents into chunks
    chunks = split_into_chunks(documents)
    print(f"Number of chunks created: {len(chunks)} from {file_path}")
    # Build vector store from chunks and save it
    vector_store = build_vector_store(chunks)
    save_vector_store(vector_store)
    print(f"Vector store built and saved to {config.VECTOR_STORE_PATH}")
    return vector_store


def build_hr_policy_assistant(file_path: str = config.DATA_FILE_PATH):
    """Build the HR policy assistant by creating the vector store, retriever, tools, and agent. ready to use."""
    config.check_api_keys()  # Check if API keys are set

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

    return agent


def ask_hr_policy_question(agent, question: str) -> str:
    """Ask a question to the HR policy assistant agent and return the answer."""
    response = agent.invoke({"messages": [{"role": "user", "content": question}]})
    return response["messages"][-1].content

