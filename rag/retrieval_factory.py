from functools import lru_cache

from rag.application.retrieval_service import RetrievalService
from rag.infrastructure.embedding_provider import FastEmbedProvider
from rag.infrastructure.vector_store import QdrantVectorStore


@lru_cache(maxsize=1)
def get_retrieval_service() -> RetrievalService:
    """Singleton — model embedding + kết nối Qdrant chỉ khởi tạo 1 lần."""
    return RetrievalService(embedder=FastEmbedProvider(), vector_store=QdrantVectorStore())
