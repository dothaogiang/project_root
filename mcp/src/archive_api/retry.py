"""
archive_api/retry.py — Retry dùng chung cho các lệnh gọi Public Archive
API (search_archives, session-token) qua httpx.

Chỉ retry lỗi TẠM THỜI (mất kết nối, timeout, DNS chập chờn, lỗi 5xx từ
server) — KHÔNG retry lỗi 4xx (400 do param sai, 404...), vì gọi lại y
hệt request vẫn sẽ sai, chỉ tổ tăng độ trễ và tốn thêm request vô ích.
401 cũng KHÔNG do tenacity xử lý — client.py đã có cơ chế refresh token
rồi thử lại đúng 1 lần riêng, tránh chồng chéo 2 tầng retry cho cùng 1
loại lỗi.
"""
import httpx
import tenacity

from logger import get_logger

logger = get_logger(__name__)


def _is_retryable_error(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TransportError):
        # ConnectError, ConnectTimeout, ReadTimeout, PoolTimeout,
        # RemoteProtocolError... — đều là lỗi tầng mạng, có khả năng
        # tự khỏi nếu thử lại sau vài trăm ms.
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        # 5xx = lỗi phía server (quá tải, restart...), có thể tạm thời.
        # 4xx KHÔNG retry vì là lỗi request, thử lại vẫn sai y hệt.
        return exc.response.status_code >= 500
    return False


def _log_before_sleep(retry_state: tenacity.RetryCallState) -> None:
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    logger.warning(
        f"Gọi Archive API lỗi tạm thời ({exc!r}), thử lại lần "
        f"{retry_state.attempt_number}/3..."
    )


# Tối đa 3 lần thử (1 gốc + 2 retry), backoff tăng dần 0.5s -> 1s -> 2s
# (chặn ở 4s) — đủ để vượt qua 1 lần mạng chập chờn thoáng qua mà không
# làm caller (tool MCP) phải chờ quá lâu nếu server thật sự đang down.
retry_transient = tenacity.retry(
    retry=tenacity.retry_if_exception(_is_retryable_error),
    wait=tenacity.wait_exponential(multiplier=0.5, min=0.5, max=4),
    stop=tenacity.stop_after_attempt(3),
    before_sleep=_log_before_sleep,
    reraise=True,
)