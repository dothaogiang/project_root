"""
application/retrieval_service.py — Use case "truy vấn dữ liệu RAG".

Đây chính là phần mà MCP tools (search_archives khi fallback semantic,
get_profile_detail, find_profile_and_answer, search_content)
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
from rag.infrastructure.text_normalize import strip_diacritics

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
        query_embedding = self._embedder.embed_text(strip_diacritics(keyword))
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
        phần RAG thật sự (retrieval trên nội dung PDF/MD đã extract)."""
        query_embedding = self._embedder.embed_text(question)
        results = self._vector_store.search_chunks(query_embedding, top_k, archive_id=archive_id)
        return _filter_by_score_gap(results, score_gap_ratio)

    def search_chunks_all(
        self,
        question: str,
        top_k: int = 5,
        score_gap_ratio: float = _DEFAULT_SCORE_GAP_RATIO,
    ) -> list[RetrievedChunk]:
        """Tìm đoạn text liên quan nhất TRÊN TOÀN BỘ hồ sơ đã ingest,
        KHÔNG giới hạn 1 archive_id cụ thể. Dùng cho câu hỏi kiểu liệt
        kê/khám phá khi chưa biết trước hồ sơ nào liên quan, VD "tìm
        những hồ sơ là nông dân". Mỗi RetrievedChunk trả về có kèm
        archive_id để biết đoạn đó thuộc hồ sơ nào."""
        query_embedding = self._embedder.embed_text(question)
        results = self._vector_store.search_chunks(query_embedding, top_k)
        return _filter_by_score_gap(results, score_gap_ratio)

    def find_profile_and_answer(
        self,
        key: str,
        question: str,
        top_k: int = 5,
        score_gap_ratio: float = _DEFAULT_SCORE_GAP_RATIO,
    ) -> tuple[RetrievedProfile | None, list[RetrievedChunk]]:
        """Kết hợp 2 bước cho use case 'hỏi về 1 hồ sơ cụ thể nhưng chưa
        có archive_id': (1) tìm hồ sơ khớp nhất với `key` (VD tên
        người, mã hồ sơ), (2) tìm đoạn text trả lời `question` TRONG
        chính hồ sơ đó. VD: key="Lê Minh Tuấn",
        question="quyết định tăng lương vào ngày nào".

        Trả về (None, []) nếu không tìm thấy hồ sơ nào khớp `key`."""
        profiles = self.search_profiles(key, top_k=1)
        if not profiles:
            return None, []
        best_profile = profiles[0]
        chunks = self.search_chunks_in_archive(
            archive_id=best_profile.archive_id,
            question=question,
            top_k=top_k,
            score_gap_ratio=score_gap_ratio,
        )
        return best_profile, chunks