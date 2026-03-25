"""Multi-format document ingestion pipeline."""
from pathlib import Path
from typing import List, Union
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, UnstructuredHTMLLoader,
    TextLoader, CSVLoader, WebBaseLoader,
)
from langchain_core.documents import Document
import logging

logger = logging.getLogger(__name__)

LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".html": UnstructuredHTMLLoader,
    ".htm": UnstructuredHTMLLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
    ".csv": CSVLoader,
}


def load_document(source: Union[str, Path]) -> List[Document]:
    """Load a document from file path or URL."""
    source = str(source)
    if source.startswith("http"):
        loader = WebBaseLoader(source)
    else:
        ext = Path(source).suffix.lower()
        loader_cls = LOADER_MAP.get(ext)
        if not loader_cls:
            raise ValueError(f"Unsupported format: {ext}. Supported: {list(LOADER_MAP.keys())}")
        loader = loader_cls(source)

    docs = loader.load()
    logger.info(f"Loaded {len(docs)} pages from {source}")
    return docs


def load_directory(directory: Union[str, Path], glob: str = "**/*.*") -> List[Document]:
    """Recursively load all supported documents from a directory."""
    from langchain_community.document_loaders import DirectoryLoader
    all_docs = []
    for ext, loader_cls in LOADER_MAP.items():
        loader = DirectoryLoader(str(directory), glob=f"**/*{ext}", loader_cls=loader_cls, silent_errors=True)
        docs = loader.load()
        all_docs.extend(docs)
    logger.info(f"Loaded {len(all_docs)} total documents from {directory}")
    return all_docs
