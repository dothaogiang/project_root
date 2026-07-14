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