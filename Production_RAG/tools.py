"""Step 5: Define a tool for the agent to search HR policies using the retriever."""

from langchain.tools import tool

def search_hr_policy_tool(retriever):
    """Define a tool for the agent to search HR policies using the retriever."""
    @tool
    def search_hr_policy(question: str) -> str:
        """Search HR policies document for information related to the query and return the relevant chunks."""
        # Use the retriever to get relevant chunks for the query
        relevant_chunks = retriever.invoke(question)
        return "\n\n".join([chunk.page_content for chunk in relevant_chunks])

    return search_hr_policy
    