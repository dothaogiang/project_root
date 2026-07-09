"""
infrastructure/archive_api_client.py — Adapter gọi Public Archive API
thật (192.168.1.46:4000 hoặc theo ARCHIVE_API_BASE_URL trong .env).

Implement ArchiveApiClientPort. Đây là nơi DUY NHẤT trong toàn bộ module
rag/ biết địa chỉ, path, format response của Archive API. Nếu API đổi
field name hay endpoint, chỉ cần sửa file này.
"""
import httpx

from rag.config.rag_config import rag_config
from rag.domain.entities import ArchiveRecord
from rag.ports.interfaces import ArchiveApiClientPort
from rag.logger import get_logger

logger = get_logger(__name__)


def _to_archive_record(raw: dict) -> ArchiveRecord:
    return ArchiveRecord(
        id=raw["id"],
        title=raw.get("title", ""),
        arc_file_code=raw.get("arcFileCode"),
        shelf_code=raw.get("shelfCode"),
        shelf_level_code=raw.get("shelfLevelCode"),
        warehouse_name=raw.get("warehouseName"),
        room_number=raw.get("roomNumber"),
        start_date=raw.get("startDate"),
        end_date=raw.get("endDate"),
        status=raw.get("status"),
        description=raw.get("description"),
        total_doc=raw.get("totalDoc"),
        language=raw.get("language"),
        maintenance=raw.get("maintenance"),
        updated_at=raw.get("updatedAt"),
        staff_metadata=raw.get("staffMetadata") or [],
        projects=raw.get("projects") or [],
        borrow_items=raw.get("borrowItems") or [],
    )


class HttpArchiveApiClient(ArchiveApiClientPort):
    def __init__(self):
        self._base_url = rag_config.ARCHIVE_API_BASE_URL
        self._path = rag_config.ARCHIVE_API_PATH
        self._timeout = rag_config.HTTP_TIMEOUT_SECONDS
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient()
        return self

    async def __aexit__(self, *exc):
        if self._client:
            await self._client.aclose()

    def _require_client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError(
                "HttpArchiveApiClient phải được dùng trong `async with` "
                "để quản lý vòng đời kết nối HTTP."
            )
        return self._client

    async def fetch_page(self, page: int, page_size: int) -> tuple[list[ArchiveRecord], bool]:
        client = self._require_client()
        url = f"{self._base_url}{self._path}"
        resp = await client.get(
            url, params={"page": page, "size": page_size}, timeout=self._timeout
        )
        resp.raise_for_status()
        data = resp.json()
        records = [_to_archive_record(r) for r in data.get("content", [])]

        page_info = data.get("page") or {}
        current_number = page_info.get("number", page)
        total_pages = page_info.get("totalPages", 1)
        is_last = (current_number + 1) >= total_pages

        return records, is_last

    async def download_file(self, file_url: str) -> bytes:
        client = self._require_client()
        resp = await client.get(file_url, timeout=self._timeout)
        resp.raise_for_status()
        return resp.content

# """
# infrastructure/archive_api_client.py — Adapter gọi Archive API.
#
# File này hỗ trợ 2 kiểu response:
#
# 1. API chính thức dạng danh sách archive:
#    {
#      "content": [...],
#      "last": true
#    }
#
# 2. API test preview 1 project:
#    {
#      "id": "...",
#      "name": "...",
#      "documents": [
#        {
#          "file_url": "...pdf",
#          "file_name": "..."
#        }
#      ]
#    }
# """
# from urllib.parse import urlparse, urlunparse
#
# import httpx
#
# from rag.config.rag_config import rag_config
# from rag.domain.entities import ArchiveRecord
# from rag.ports.interfaces import ArchiveApiClientPort
# from rag.logger import get_logger
#
# logger = get_logger(__name__)
#
#
# def _normalize_file_url(file_url: str | None) -> str | None:
#     """
#     API test có thể trả file_url dạng http://localhost:4000/...
#     Nếu chạy trong Docker hoặc máy khác, localhost sẽ trỏ sai chỗ.
#     Ta đổi localhost/127.0.0.1 sang host trong ARCHIVE_API_BASE_URL.
#     """
#     if not file_url:
#         return None
#
#     parsed = urlparse(file_url)
#
#     if not parsed.scheme:
#         return f"{rag_config.ARCHIVE_API_BASE_URL.rstrip('/')}/{file_url.lstrip('/')}"
#
#     if parsed.hostname not in {"localhost", "127.0.0.1"}:
#         return file_url
#
#     base = urlparse(rag_config.ARCHIVE_API_BASE_URL)
#     replaced = parsed._replace(scheme=base.scheme, netloc=base.netloc)
#     return urlunparse(replaced)
#
#
# def _normalize_projects(projects: list) -> list:
#     """Chuẩn hóa fileUrls trong projects của API chính thức."""
#     normalized = []
#
#     for project in projects or []:
#         item = dict(project)
#         item["fileUrls"] = [
#             url for url in (_normalize_file_url(u) for u in project.get("fileUrls") or []) if url
#         ]
#         normalized.append(item)
#
#     return normalized
#
#
# def _to_archive_record(raw: dict) -> ArchiveRecord:
#     """Convert response API chính thức thành ArchiveRecord."""
#     return ArchiveRecord(
#         id=raw["id"],
#         title=raw.get("title", ""),
#         arc_file_code=raw.get("arcFileCode"),
#         box_code=raw.get("boxCode"),
#         warehouse_name=raw.get("warehouseName"),
#         start_date=raw.get("startDate"),
#         end_date=raw.get("endDate"),
#         status=raw.get("status"),
#         updated_at=raw.get("updatedAt"),
#         staff_metadata=raw.get("staffMetadata") or [],
#         projects=_normalize_projects(raw.get("projects") or []),
#     )
#
#
# def _preview_to_archive_record(raw: dict) -> ArchiveRecord:
#     """Convert API test /api/v1/projects/preview/{id} thành ArchiveRecord."""
#     file_urls = []
#
#     for doc in raw.get("documents") or []:
#         file_url = _normalize_file_url(doc.get("file_url"))
#         if file_url:
#             file_urls.append(file_url)
#
#     fields_definition = raw.get("config", {}).get("fields_definition") or []
#     staff_metadata = [
#         {
#             "fieldName": field.get("name") or field.get("code"),
#             "value": field.get("example") or field.get("description") or "",
#             "code": field.get("code"),
#         }
#         for field in fields_definition
#     ]
#
#     title = raw.get("name") or raw.get("id") or ""
#
#     return ArchiveRecord(
#         id=raw["id"],
#         title=title,
#         arc_file_code=raw.get("id"),
#         box_code=None,
#         warehouse_name=None,
#         start_date=None,
#         end_date=None,
#         status=raw.get("status"),
#         updated_at=str(raw.get("updated_at") or raw.get("created_at") or ""),
#         staff_metadata=staff_metadata,
#         projects=[
#             {
#                 "name": title,
#                 "fileUrls": file_urls,
#             }
#         ],
#     )
#
#
# class HttpArchiveApiClient(ArchiveApiClientPort):
#     def __init__(self):
#         self._base_url = rag_config.ARCHIVE_API_BASE_URL.rstrip("/")
#         self._path = rag_config.ARCHIVE_API_PATH
#         self._timeout = rag_config.HTTP_TIMEOUT_SECONDS
#         self._client: httpx.AsyncClient | None = None
#
#     async def __aenter__(self):
#         self._client = httpx.AsyncClient()
#         return self
#
#     async def __aexit__(self, *exc):
#         if self._client:
#             await self._client.aclose()
#
#     def _require_client(self) -> httpx.AsyncClient:
#         if self._client is None:
#             raise RuntimeError(
#                 "HttpArchiveApiClient phải được dùng trong `async with` "
#                 "để quản lý vòng đời kết nối HTTP."
#             )
#         return self._client
#
#     async def fetch_page(self, page: int, page_size: int) -> tuple[list[ArchiveRecord], bool]:
#         client = self._require_client()
#         url = f"{self._base_url}{self._path}"
#
#         is_preview_endpoint = "/preview/" in self._path
#
#         if is_preview_endpoint and page > 0:
#             return [], True
#
#         params = None if is_preview_endpoint else {"page": page, "size": page_size}
#
#         resp = await client.get(url, params=params, timeout=self._timeout)
#         resp.raise_for_status()
#
#         data = resp.json()
#
#         # API chính thức: danh sách archive phân trang.
#         if isinstance(data, dict) and "content" in data:
#             records = [_to_archive_record(r) for r in data.get("content", [])]
#             is_last = bool(data.get("last", True))
#             return records, is_last
#
#         # API test: preview 1 project.
#         if isinstance(data, dict) and "documents" in data:
#             return [_preview_to_archive_record(data)], True
#
#         logger.warning(f"Response Archive API không đúng format mong đợi: {data}")
#         return [], True
#
#     async def download_file(self, file_url: str) -> bytes:
#         client = self._require_client()
#         normalized_url = _normalize_file_url(file_url)
#
#         resp = await client.get(normalized_url, timeout=self._timeout)
#         resp.raise_for_status()
#         return resp.content