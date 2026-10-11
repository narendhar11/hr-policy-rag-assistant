"""Step 5: Define a tool for the agent to search HR policies using the retriever."""

from langchain.tools import tool
from utils.logger import get_logger

logger = get_logger(__name__)

def search_hr_policy_tool(retriever):
    """Define a tool for the agent to search HR policies using the retriever."""
    @tool
    def search_hr_policy(question: str) -> str:
        """Search HR policies document for information related to the query and return the relevant chunks."""
        logger.info("Searching HR policies for question: '%s'", question)
        # Use the retriever to get relevant chunks for the query
        relevant_chunks = retriever.invoke(question)
        logger.info("Found %d relevant chunks for question: '%s'", len(relevant_chunks), question)
        return "\n\n".join([chunk.page_content for chunk in relevant_chunks])

    return search_hr_policy
    