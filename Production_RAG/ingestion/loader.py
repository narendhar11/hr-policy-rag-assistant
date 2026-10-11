"""Load HR policy documents from file and return LangChain Document objects."""

from langchain_core.documents import Document
import config
from utils.logger import get_logger

logger = get_logger(__name__)


def load_documents(file_path: str = config.DATA_FILE_PATH):
    """Load txt file and return a list of Document objects."""
    logger.info(f"Loading documents from {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    logger.info("Loaded document successfully from %s", file_path)
    return [Document(page_content=content, metadata={"source": file_path})]
