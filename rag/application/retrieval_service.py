"""
application/retrieval_service.py — Use case "truy vấn dữ liệu RAG".

Đây chính là phần mà MCP tools (search_profile, get_profile_detail)
HOẶC bất kỳ chatbot/service nào khác sẽ gọi vào để lấy dữ liệu đã được
ingest sẵn trong Qdrant. RetrievalService không quan tâm ai gọi nó (MCP,
REST API nội bộ, script...) — nó chỉ nhận câu hỏi, trả về dữ liệu.

Tách riêng khỏi IngestionService vì 2 use case có vòng đời và tần suất
gọi khác nhau: ingestion chạy định kỳ (cron/APScheduler), retrieval
chạy theo mỗi request của người dùng cuối.
"""
from typing import TypeVar

from rag.domain.entities import RetrievedChunk, RetrievedProfile
from rag.ports.interfaces import EmbeddingProviderPort, VectorStorePort

_Scored = TypeVar("_Scored", RetrievedProfile, RetrievedChunk)

# RRF (Reciprocal Rank Fusion) luôn trả đủ top_k kết quả, kể cả khi phần
# lớn không thực sự liên quan — vì vector search không có khái niệm
# "không tìm thấy". Điểm rơi rất nhanh theo dạng 1/rank là dấu hiệu
# điển hình cho thấy phần đuôi chỉ là noise. Lọc bớt tại đây để chatbot
# phía sau nhận được kết quả sạch, không phải tự "đãi cát tìm vàng".
_DEFAULT_SCORE_GAP_RATIO = 0.5


def _filter_by_score_gap(results: list[_Scored], ratio: float) -> list[_Scored]:
    if not results:
        return results
    top_score = results[0].score
    if top_score <= 0:
        return results
    return [r for r in results if r.score >= top_score * ratio]


class RetrievalService:
    def __init__(self, embedder: EmbeddingProviderPort, vector_store: VectorStorePort):
        self._embedder = embedder
        self._vector_store = vector_store

    def search_profiles(
        self, keyword: str, top_k: int = 10, score_gap_ratio: float = _DEFAULT_SCORE_GAP_RATIO
    ) -> list[RetrievedProfile]:
        """Tìm hồ sơ (archive) theo từ khóa tự do — tầng 'định danh hồ sơ'.

        score_gap_ratio: chỉ giữ lại kết quả có điểm >= score_gap_ratio *
        điểm cao nhất. Đặt 0 để tắt lọc (trả nguyên top_k như Qdrant trả về).
        """
        query_embedding = self._embedder.embed_text(keyword)
        results = self._vector_store.search_profiles(query_embedding, top_k)
        return _filter_by_score_gap(results, score_gap_ratio)

    def search_chunks_in_archive(
        self,
        archive_id: str,
        question: str,
        top_k: int = 5,
        score_gap_ratio: float = _DEFAULT_SCORE_GAP_RATIO,
    ) -> list[RetrievedChunk]:
        """Tìm đoạn text liên quan nhất bên TRONG 1 hồ sơ cụ thể — đây là
        phần RAG thật sự (retrieval trên nội dung PDF đã OCR/extract)."""
        query_embedding = self._embedder.embed_text(question)
        results = self._vector_store.search_chunks(query_embedding, archive_id, top_k)
        return _filter_by_score_gap(results, score_gap_ratio)