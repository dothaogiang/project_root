"""
archive_api/client.py — Client gọi TRỰC TIẾP Public Archive API, phục vụ
tool `search_archives` (tìm hồ sơ theo keyword/filter chính xác, LIVE).

KHÁC với rag/infrastructure/archive_api_client.py: client đó chỉ phục
vụ ingestion hàng loạt cho pipeline RAG (fetch_page/download_file để
đưa vào Qdrant, không cần auth). Client này phục vụ tool MCP trả lời
LIVE cho người dùng — gọi thẳng theo field lọc chính xác, có auth qua
X-Chatbot-Token, và tự refresh token khi gặp 401.

Response search_archives của API thật hiện đã trả kèm nội dung Markdown
đầy đủ trong từng project/document (documents[].markdown) — nên tool
này không cần gọi thêm endpoint "detail" nào khác để lấy nội dung.

Cố ý đặt trong mcp/ (không phải rag/) vì đây không phải logic
embedding/vector — mcp/ sở hữu tích hợp trực tiếp này để phục vụ đúng
tool của nó, giữ rag/ chỉ tập trung vào ingestion + semantic retrieval
(dùng cho tìm mờ/ngữ nghĩa qua search_profile/get_profile_detail/
find_profile_and_answer/search_content).
"""
import asyncio
from typing import Any, Optional

import httpx

from archive_api.token_manager import get_token_manager
from config.configs import config_object
from logger import get_logger

logger = get_logger(__name__)


class ArchiveApiClient:
    def __init__(self):
        self._base_url = config_object.ARCHIVE_API_BASE_URL
        self._timeout = config_object.HTTP_TIMEOUT_SECONDS
        self._token_manager = get_token_manager()

    async def _request(self, method: str, path: str, params: Optional[dict] = None) -> httpx.Response:
        url = f"{self._base_url}{path}"
        token = await self._token_manager.get_token()
        headers = {"X-Chatbot-Token": token} if token else {}

        async with httpx.AsyncClient() as client:
            resp = await client.request(method, url, params=params, headers=headers, timeout=self._timeout)

            if resp.status_code == 401 and config_object.AUTH_ENABLED:
                logger.warning(f"Token hết hạn/không hợp lệ khi gọi {url}, xin token mới rồi thử lại...")
                token = await self._token_manager.get_token(force_refresh=True)
                headers = {"X-Chatbot-Token": token} if token else {}
                resp = await client.request(method, url, params=params, headers=headers, timeout=self._timeout)

            resp.raise_for_status()
            return resp

    async def search_archives(
            self,
            keywords: Optional[list[str]] = None,
            status: Optional[str] = None,
            warehouse_id: Optional[str] = None,
            language: Optional[str] = None,
            maintenance: Optional[str] = None,
            created_from: Optional[str] = None,
            created_to: Optional[str] = None,
            updated_from: Optional[str] = None,
            updated_to: Optional[str] = None,
            page: int = 0,
            size: int = 20,
    ) -> dict[str, Any]:
        """
        Luôn trả về ĐÚNG 1 TRANG (mặc định page=0, size=20) — KHÔNG tự
        động gộp nhiều trang, để tránh dồn quá nhiều record vào 1 lần
        trả lời LLM (tốn token, dễ vượt context). Response có đủ
        totalPages/totalElements/last để bên gọi (feature_manager) biết
        còn trang tiếp theo hay không và tự quyết định có cần gọi lại
        tool với page+1 hay dừng lại vì đã tìm thấy.

        `keywords`: 1 HOẶC NHIỀU biến thể từ khóa (VD tên có dấu, tên
        không dấu, viết tắt...) trong CÙNG 1 lượt gọi tool — thay vì bên
        gọi (LLM) phải tự lặp lại tool nhiều lần với từng biến thể.
        Public Archive API chỉ nhận 1 `keyword` mỗi request, nên ở đây
        tự fan-out song song (asyncio.gather) — 1 request/biến thể nếu
        có >1 keyword, rồi gộp + khử trùng lặp theo `id` trước khi trả
        về. Với 1 keyword (hoặc không có), chỉ tốn đúng 1 request như
        trước, không phát sinh thêm overhead.
        """
        base_params = {
            "status": status,
            "warehouseId": warehouse_id,
            "language": language,
            "maintenance": maintenance,
            "createdFrom": created_from,
            "createdTo": created_to,
            "updatedFrom": updated_from,
            "updatedTo": updated_to,
            "page": page,
            "size": size,
        }
        base_params = {k: v for k, v in base_params.items() if v is not None}

        # Khử trùng lặp + bỏ chuỗi rỗng, giữ nguyên thứ tự biến thể được truyền vào
        unique_keywords = list(dict.fromkeys(k for k in (keywords or []) if k and k.strip()))

        if len(unique_keywords) <= 1:
            params = dict(base_params)
            if unique_keywords:
                params["keyword"] = unique_keywords[0]
            resp = await self._request("GET", config_object.ARCHIVE_SEARCH_PATH, params=params)
            return resp.json()

        # >1 keyword: gọi song song, mỗi request 1 keyword, rồi gộp kết quả
        async def _search_one(kw: str) -> dict[str, Any]:
            params = {**base_params, "keyword": kw}
            resp = await self._request("GET", config_object.ARCHIVE_SEARCH_PATH, params=params)
            return resp.json()

        pages = await asyncio.gather(*(_search_one(kw) for kw in unique_keywords))

        merged_content: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for page_result in pages:
            for record in page_result.get("content", []):
                rid = record.get("id")
                if rid is not None and rid in seen_ids:
                    continue
                if rid is not None:
                    seen_ids.add(rid)
                merged_content.append(record)

        merged_content = merged_content[:size]
        return {
            "content": merged_content,
            "page": {
                "size": size,
                "number": page,
                "totalElements": len(merged_content),
                "totalPages": 1,
            },
        }


_client: Optional[ArchiveApiClient] = None


def get_archive_api_client() -> ArchiveApiClient:
    global _client
    if _client is None:
        _client = ArchiveApiClient()
    return _client