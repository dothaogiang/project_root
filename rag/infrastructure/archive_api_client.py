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
        box_code=raw.get("boxCode"),
        warehouse_name=raw.get("warehouseName"),
        start_date=raw.get("startDate"),
        end_date=raw.get("endDate"),
        status=raw.get("status"),
        updated_at=raw.get("updatedAt"),
        staff_metadata=raw.get("staffMetadata") or [],
        projects=raw.get("projects") or [],
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
        is_last = bool(data.get("last", True))
        return records, is_last

    async def download_file(self, file_url: str) -> bytes:
        client = self._require_client()
        resp = await client.get(file_url, timeout=self._timeout)
        resp.raise_for_status()
        return resp.content
