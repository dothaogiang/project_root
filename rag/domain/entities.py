"""
domain/entities.py — Các model dữ liệu THUẦN (plain dataclass), không phụ
thuộc Qdrant, httpx, fastembed hay bất kỳ thư viện hạ tầng nào.

Đây là tầng trong cùng của Clean Architecture: application/ và
infrastructure/ đều được PHÉP import từ đây, nhưng domain/ không được
import ngược lại bất cứ thứ gì từ 2 tầng đó.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ArchiveRecord:
    """1 hồ sơ lấy từ Public Archive API (nguyên bản, trước khi xử lý RAG)."""

    id: str
    title: str
    arc_file_code: Optional[str] = None
    box_code: Optional[str] = None
    warehouse_name: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    status: Optional[str] = None
    updated_at: Optional[str] = None
    staff_metadata: list = field(default_factory=list)
    projects: list = field(default_factory=list)  # mỗi project có "name" + "fileUrls"

    def file_urls(self) -> list[tuple[str, str]]:
        """Trả về list (project_name, file_url) để duyệt qua từng file PDF."""
        pairs = []
        for project in self.projects or []:
            name = project.get("name", "")
            for url in project.get("fileUrls") or []:
                pairs.append((name, url))
        return pairs


@dataclass
class ExtractedPage:
    """Text trích được từ 1 trang PDF, kèm phương pháp đã dùng."""

    page_number: int
    text: str
    extraction_method: str  # "native" | "ocr"


@dataclass
class DocumentChunk:
    """1 đoạn text đã được chia nhỏ, sẵn sàng để embed và lưu vào Qdrant."""

    archive_id: str
    file_url: str
    chunk_index: int
    page_number: int
    text: str
    extraction_method: str
    project_name: Optional[str] = None


@dataclass
class Embedding:
    """Kết quả embed của 1 đoạn text: dense (semantic) + sparse (lexical)."""

    dense: list
    sparse_indices: list
    sparse_values: list


@dataclass
class RetrievedChunk:
    """1 chunk trả về sau khi hybrid search, kèm điểm relevance."""

    text: str
    file_url: str
    page_number: int
    extraction_method: str
    score: float


@dataclass
class RetrievedProfile:
    """1 hồ sơ trả về sau khi hybrid search ở tầng metadata (archive-level)."""

    archive_id: str
    title: Optional[str]
    arc_file_code: Optional[str]
    box_code: Optional[str]
    warehouse_name: Optional[str]
    start_date: Optional[str]
    end_date: Optional[str]
    staff_metadata: list
    score: float
