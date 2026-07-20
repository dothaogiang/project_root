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
"""
import asyncio
import re
from datetime import date
from typing import Any, Optional

import httpx

from archive_api.token_manager import get_token_manager
from config.configs import config_object
from logger import get_logger

logger = get_logger(__name__)

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _validate_iso_date(field_name: str, value: Optional[str]) -> None:
    """
    Chặn sớm các giá trị ngày sai định dạng (VD chỉ có năm "2022",
    hoặc "2022/01/01") TRƯỚC KHI gửi lên Public Archive API.

    LƯU Ý: nếu backend Archive API thật ra cần format DD/MM/YYYY (khớp
    startDate/endDate nó tự trả ra) thay vì ISO YYYY-MM-DD, đổi lại
    _ISO_DATE_RE + cách parse bên dưới cho phù hợp.

    Lý do cần bước này: API thật không validate input tử tế — gửi
    ngày sai định dạng khiến nó crash và trả về lỗi 500 chung chung
    (Internal Server Error) thay vì 400 Bad Request rõ ràng, rất khó
    debug cho bên gọi (LLM/chatbot). Validate ở đây để LLM nhận ngay
    thông báo lỗi dễ hiểu và có thể tự sửa lại tham số ở lượt gọi sau,
    thay vì nhận traceback 500 khó hiểu từ 1 hệ thống khác.
    """
    if not value:
        # Chặn cả None VÀ chuỗi rỗng '' — nhiều MCP client gửi "" cho
        # field optional không dùng tới thay vì omit hẳn, nên phải coi
        # là "không truyền" giống None, không phải giá trị cần validate.
        return
    if not _ISO_DATE_RE.match(value):
        raise ValueError(
            f"Tham số '{field_name}' = {value!r} sai định dạng. "
            "Phải là ngày ĐẦY ĐỦ dạng YYYY-MM-DD (VD 2022-01-01), "
            "không được chỉ gửi năm hoặc dùng dấu '/'."
        )
    try:
        date.fromisoformat(value)
    except ValueError:
        raise ValueError(
            f"Tham số '{field_name}' = {value!r} không phải ngày hợp lệ "
            "(VD tháng/ngày ngoài phạm vi cho phép)."
        )


class ArchiveApiClient:
    def __init__(self):
        self._base_url = config_object.ARCHIVE_API_BASE_URL
        self._timeout = config_object.HTTP_TIMEOUT_SECONDS
        self._token_manager = get_token_manager()
        # QUAN TRỌNG: 1 AsyncClient DUY NHẤT cho suốt vòng đời process,
        # KHÔNG tạo mới mỗi request. httpx.AsyncClient giữ 1 connection
        # pool bên trong (keep-alive) — tái sử dụng được kết nối TCP/TLS
        # đã bắt tay xong cho các request tiếp theo tới CÙNG base_url,
        # thay vì bắt tay lại từ đầu mỗi lần gọi. Quan trọng hơn: khi
        # search_archives fan-out song song nhiều keyword (asyncio.gather
        # ở _search_one), tất cả cùng dùng chung 1 pool này thay vì mỗi
        # nhánh song song tự mở 1 connection riêng.
        # Khởi tạo LAZY (ở _get_client) chứ không tạo ngay ở __init__,
        # vì AsyncClient cần được tạo bên trong 1 event loop đang chạy.
        self._http_client: Optional[httpx.AsyncClient] = None
        self._client_lock = asyncio.Lock()

    async def _get_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            async with self._client_lock:
                # double-checked locking: tránh 2 request đầu tiên chạy
                # song song cùng lúc tạo ra 2 client khác nhau
                if self._http_client is None:
                    self._http_client = httpx.AsyncClient()
        return self._http_client

    async def aclose(self) -> None:
        """Đóng connection pool khi server shutdown. Gọi từ server.py
        (hoặc bỏ qua cũng không sao vì đây là singleton sống hết vòng
        đời process — OS tự dọn socket khi process thoát)."""
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    async def _request(self, method: str, path: str, params: Optional[dict] = None) -> httpx.Response:
        url = f"{self._base_url}{path}"
        token = await self._token_manager.get_token()
        headers = {"X-Chatbot-Token": token} if token else {}
        client = await self._get_client()

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
        _validate_iso_date("created_from", created_from)
        _validate_iso_date("created_to", created_to)
        _validate_iso_date("updated_from", updated_from)
        _validate_iso_date("updated_to", updated_to)

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
        base_params = {k: v for k, v in base_params.items() if v not in (None, "")}

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

        # QUAN TRỌNG: tính totalElements TRƯỚC KHI cắt theo size, không
        # phải sau. feature_manager.py dựa vào so sánh
        # "totalElements > len(content)" để quyết định có cần bổ sung
        # ứng viên qua semantic search hay không (trang hiện tại có bị
        # cắt bớt hồ sơ khớp hay không). Nếu tính totalElements = số
        # lượng SAU KHI đã cắt (bằng đúng len(content) sau cắt), 2 giá
        # trị này luôn bằng nhau -> điều kiện trên không bao giờ đúng
        # khi có >1 keyword, vô tình tắt luôn phần bổ sung semantic
        # fallback ở nhánh phổ biến nhất (nhiều biến thể từ khóa).
        total_elements = len(merged_content)
        merged_content = merged_content[:size]
        return {
            "content": merged_content,
            "page": {
                "size": size,
                "number": page,
                "totalElements": total_elements,
                "totalPages": 1,
            },
        }


_client: Optional[ArchiveApiClient] = None


def get_archive_api_client() -> ArchiveApiClient:
    global _client
    if _client is None:
        _client = ArchiveApiClient()
    return _client