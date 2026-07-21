"""
infrastructure/retry.py — Retry dùng chung cho các lệnh gọi mạng trong
module rag/: Archive API (qua httpx, dùng khi ingest) và Qdrant (qua
qdrant_client, dùng cả khi ingest lẫn khi retrieval).

Tách riêng bản này khỏi mcp/src/archive_api/retry.py (dù logic tương tự)
vì rag/ CỐ Ý không phụ thuộc bất kỳ thứ gì trong mcp/ (xem rag/__init__.py)
— để rag/ tái sử dụng được độc lập ở service khác mà không kéo theo
toàn bộ tầng MCP.

Chỉ retry lỗi TẠM THỜI (mất kết nối, timeout, lỗi 5xx từ server) —
KHÔNG retry lỗi 4xx, vì gọi lại y hệt request vẫn sẽ sai.
"""
import httpx
import tenacity
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from rag.logger import get_logger

logger = get_logger(__name__)


def _is_retryable_http_error(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    return False


def _is_retryable_qdrant_error(exc: BaseException) -> bool:
    if isinstance(exc, ResponseHandlingException):
        # Lỗi tầng mạng khi gọi Qdrant (connection refused, timeout...)
        # — qdrant_client bọc lại thành ResponseHandlingException.
        return True
    if isinstance(exc, UnexpectedResponse):
        # UnexpectedResponse.status_code có thể là None (VD lỗi decode
        # response) — coi là KHÔNG retryable trong trường hợp đó, an
        # toàn hơn là đoán bừa.
        return exc.status_code is not None and exc.status_code >= 500
    return False


def _log_before_sleep(retry_state: tenacity.RetryCallState) -> None:
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    logger.warning(
        f"Gọi mạng lỗi tạm thời ({exc!r}), thử lại lần "
        f"{retry_state.attempt_number}/3..."
    )


_COMMON_RETRY_KWARGS = dict(
    wait=tenacity.wait_exponential(multiplier=0.5, min=0.5, max=4),
    stop=tenacity.stop_after_attempt(3),
    before_sleep=_log_before_sleep,
    reraise=True,
)

# Dùng cho HttpArchiveApiClient.fetch_page (gọi Public Archive API khi
# ingest hàng loạt).
retry_http_transient = tenacity.retry(
    retry=tenacity.retry_if_exception(_is_retryable_http_error), **_COMMON_RETRY_KWARGS
)

# Dùng cho các method của QdrantVectorStore (ensure_collections, upsert_*,
# search_*) — cả lúc ingest lẫn lúc retrieval.
retry_qdrant_transient = tenacity.retry(
    retry=tenacity.retry_if_exception(_is_retryable_qdrant_error), **_COMMON_RETRY_KWARGS
)