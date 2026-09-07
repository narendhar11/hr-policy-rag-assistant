"""Load HR policy documents from file and return LangChain Document objects."""

from langchain_core.documents import Document
import config


def load_documents(file_path: str = config.DATA_FILE_PATH):
    """Load txt file and return a list of Document objects."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return [Document(page_content=content, metadata={"source": file_path})]
