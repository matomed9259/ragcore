"""Semantic document chunker with sentence-boundary awareness."""
from typing import List
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
import re


class SemanticChunker:
    """Chunks documents using semantic sentence boundaries."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separators: List[str] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators or ["\n\n", "\n", ". ", "! ", "? ", " ", ""],
            length_function=len,
        )

    def chunk(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks with metadata preservation."""
        chunks = self.splitter.split_documents(documents)
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["chunk_size"] = len(chunk.page_content)
        return chunks

    def chunk_text(self, text: str, metadata: dict = None) -> List[Document]:
        """Chunk raw text directly."""
        doc = Document(page_content=text, metadata=metadata or {})
        return self.chunk([doc])
