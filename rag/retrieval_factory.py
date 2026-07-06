"""
rag/retrieval_factory.py — Composition root cho phía TRUY VẤN (khác với
jobs/sync_job.py là composition root cho phía ĐỒNG BỘ).

Đây là "cửa ngõ" duy nhất mà tầng MCP (src/feature_manager.py) cần biết
tới khi muốn dùng RAG để trả lời search_profile / get_profile_detail.
feature_manager.py chỉ cần:

    from rag.retrieval_factory import get_retrieval_service
    service = get_retrieval_service()
    profiles = service.search_profiles(keyword)

mà không cần biết bên trong dùng Qdrant hay fastembed.
"""
from functools import lru_cache

from rag.application.retrieval_service import RetrievalService
from rag.infrastructure.embedding_provider import FastEmbedProvider
from rag.infrastructure.vector_store import QdrantVectorStore


@lru_cache(maxsize=1)
def get_retrieval_service() -> RetrievalService:
    """Singleton — model embedding + kết nối Qdrant chỉ khởi tạo 1 lần."""
    return RetrievalService(embedder=FastEmbedProvider(), vector_store=QdrantVectorStore())
