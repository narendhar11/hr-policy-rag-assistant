"""Split documents into smaller chunks for retrieval."""

from langchain_text_splitters import RecursiveCharacterTextSplitter
import config


def split_into_chunks(documents):
    """Split the document into chunks with size and overlapping chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    return text_splitter.split_documents(documents)
