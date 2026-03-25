"""Cross-encoder reranking for precision retrieval."""
from typing import List, Tuple
from langchain_core.documents import Document


class CrossEncoderReranker:
    """Reranks retrieved documents using a cross-encoder model."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", top_k: int = 5):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)
        self.top_k = top_k

    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """Rerank documents by cross-encoder relevance score."""
        if not documents:
            return []
        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.model.predict(pairs)
        scored = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
        reranked = [doc for _, doc in scored[:self.top_k]]
        for i, (score, doc) in enumerate(sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)[:self.top_k]):
            reranked[i].metadata["rerank_score"] = float(score)
        return reranked
