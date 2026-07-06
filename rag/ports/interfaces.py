"""
ports/interfaces.py — "Cổng" (interface) mà tầng application/ phụ thuộc
vào, KHÔNG phụ thuộc trực tiếp vào implementation cụ thể trong
infrastructure/.

Đây là "Dependency Inversion" của Clean Architecture: application/ chỉ
biết "tôi cần một thứ biết fetch_page(...)", không cần biết đó là
httpx, requests hay mock giả trong unit test.

Muốn đổi hạ tầng (đổi Qdrant -> Milvus, đổi fastembed -> OpenAI
embedding API...) chỉ cần viết class mới implement lại đúng các method
dưới đây trong infrastructure/, KHÔNG cần sửa application/.
"""
from abc import ABC, abstractmethod
from typing import Optional

from rag.domain.entities import (
    ArchiveRecord,
    DocumentChunk,
    Embedding,
    RetrievedChunk,
    RetrievedProfile,
)


class ArchiveApiClientPort(ABC):
    """Nguồn dữ liệu gốc: Public Archive API."""

    @abstractmethod
    async def fetch_page(self, page: int, page_size: int) -> tuple[list[ArchiveRecord], bool]:
        """Trả về (danh_sach_archive_trong_trang, is_last_page)."""

    @abstractmethod
    async def download_file(self, file_url: str) -> bytes:
        """Tải nội dung file PDF (bytes thô)."""


class PdfExtractorPort(ABC):
    """Trích xuất text từ PDF (native hoặc OCR) + chia chunk."""

    @abstractmethod
    def extract_and_chunk(
        self, archive_id: str, file_url: str, project_name: str, pdf_bytes: bytes
    ) -> list[DocumentChunk]:
        """Trả về danh sách DocumentChunk sẵn sàng để embed."""


class EmbeddingProviderPort(ABC):
    """Sinh dense + sparse embedding."""

    @abstractmethod
    def embed_text(self, text: str) -> Embedding:
        """Embed 1 đoạn text đơn lẻ (dùng cho câu query khi search)."""

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[Embedding]:
        """Embed nhiều đoạn text cùng lúc (dùng khi ingest hàng loạt chunk)."""


class VectorStorePort(ABC):
    """Kho vector — nơi lưu + truy vấn embedding (hiện là Qdrant)."""

    @abstractmethod
    def ensure_collections(self) -> None: ...

    @abstractmethod
    def upsert_archive(self, archive: ArchiveRecord, embedding: Embedding) -> None: ...

    @abstractmethod
    def upsert_chunks(self, chunks: list[DocumentChunk], embeddings: list[Embedding]) -> None: ...

    @abstractmethod
    def delete_chunks_by_file(self, archive_id: str, file_url: str) -> None: ...

    @abstractmethod
    def search_profiles(self, query_embedding: Embedding, top_k: int) -> list[RetrievedProfile]: ...

    @abstractmethod
    def search_chunks(
        self, query_embedding: Embedding, archive_id: str, top_k: int
    ) -> list[RetrievedChunk]: ...


class SyncStateRepoPort(ABC):
    """Trạng thái đồng bộ (checkpoint, hash file) để hỗ trợ incremental sync."""

    @abstractmethod
    def get_archive_last_updated(self, archive_id: str) -> Optional[str]: ...

    @abstractmethod
    def set_archive_synced(self, archive_id: str, updated_at: str) -> None: ...

    @abstractmethod
    def get_file_hash(self, archive_id: str, file_url: str) -> Optional[str]: ...

    @abstractmethod
    def set_file_synced(
        self, archive_id: str, file_url: str, content_hash: str, method: str, chunk_count: int
    ) -> None: ...

    @abstractmethod
    def get_checkpoint_page(self) -> int: ...

    @abstractmethod
    def set_checkpoint_page(self, page: int) -> None: ...

    @abstractmethod
    def reset_checkpoint(self) -> None: ...
