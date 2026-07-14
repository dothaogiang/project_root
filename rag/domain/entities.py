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
    shelf_code: Optional[str] = None
    shelf_level_code: Optional[str] = None
    warehouse_name: Optional[str] = None
    room_number: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    status: Optional[str] = None
    description: Optional[str] = None
    total_doc: Optional[int] = None
    language: Optional[str] = None
    maintenance: Optional[str] = None
    updated_at: Optional[str] = None
    staff_metadata: list = field(default_factory=list)
    projects: list = field(default_factory=list)  # mỗi project có "name" + "fileUrls"
    borrow_items: list = field(default_factory=list)

    def markdown_documents(self) -> list[tuple[str, str, str]]:
        """
        Trả về list (project_name, file_url, markdown_text) đọc TRỰC
        TIẾP nội dung Markdown đã được API trả kèm sẵn trong
        projects[].documents[].markdown (mảng các đoạn, thường theo
        từng trang tài liệu gốc) — KHÔNG cần gọi download_file() qua
        URL riêng nữa như trước. Rỗng nếu archive không có project/
        document nào hoặc document không có nội dung markdown.
        """
        results = []
        for project in self.projects or []:
            name = project.get("name", "")
            for doc in project.get("documents") or []:
                pages = doc.get("markdown") or []
                text = "\n\n".join(p.strip() for p in pages if p and p.strip())
                if not text:
                    continue
                file_url = doc.get("fileUrl") or doc.get("fileName") or ""
                results.append((name, file_url, text))
        return results


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
    """1 chunk trả về sau khi hybrid search, kèm điểm relevance.

    archive_id/project_name: LUÔN có giá trị (lấy từ payload đã lưu khi
    ingest) — quan trọng khi search KHÔNG giới hạn theo 1 archive_id cụ
    thể (search_chunks_all), vì đó là cách duy nhất biết chunk trả về
    thuộc hồ sơ nào.
    """

    text: str
    file_url: str
    page_number: int
    extraction_method: str
    score: float
    archive_id: Optional[str] = None
    project_name: Optional[str] = None


@dataclass
class RetrievedProfile:
    """1 hồ sơ trả về sau khi hybrid search ở tầng metadata (archive-level)."""

    archive_id: str
    score: float
    title: Optional[str] = None
    arc_file_code: Optional[str] = None
    shelf_code: Optional[str] = None
    shelf_level_code: Optional[str] = None
    warehouse_name: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    staff_metadata: list = field(default_factory=list)