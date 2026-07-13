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
        """Tải nội dung file MD (bytes thô)."""


class FileExtractorPort(ABC):
    """Trích xuất text từ file nguồn (hiện là Markdown) + chia chunk."""

    @abstractmethod
    def extract_and_chunk(
        self, archive_id: str, file_url: str, project_name: str, file_bytes: bytes
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