"""
FeatureManager: chứa các hàm nghiệp vụ thực thi khi tool MCP được gọi.

QUAN TRỌNG: tên hàm PHẢI trùng chính xác với `name_tool` khai báo trong
Resources/tools.yaml. ToolRegistry (tools/registry.py) sẽ tự động match
theo tên -> đây là cơ chế cho phép thêm/sửa/xóa tool chỉ bằng cách sửa
file YAML, không cần đụng vào registry.py hay server.py.

2 nhóm tool:
  - get_profile_detail / find_profile_and_answer / search_content:
    semantic search qua Qdrant (dữ liệu đã ingest sẵn bởi rag/), xem
    rag/README.md. Không còn tool search_profile riêng — search_archives
    đã tự fallback sang semantic search khi không khớp chính xác, nên
    2 tool này chỉ dùng khi đã có archive_id/key cụ thể cần hỏi sâu.
  - search_archives: gọi TRỰC TIẾP (live) vào Public Archive API qua
    archive_api/client.py, tự thử khớp chính xác rồi mới fallback
    semantic. Response API thật đã kèm sẵn Markdown đầy đủ trong
    documents[].markdown nên không cần tool "detail" riêng để lấy
    nội dung khi tìm được qua nhánh khớp chính xác.
"""
import base64
import functools
import traceback

from archive_api.client import get_archive_api_client
from config.configs import config_object
from rag.retrieval_factory import get_retrieval_service
from logger import get_logger
from urllib.parse import urlparse, parse_qs
from typing import Optional

logger = get_logger(__name__)

# Số kết quả tối đa trả về cho MỌI tool search (get_profile_detail,
# find_profile_and_answer, search_content, search_archives) — dù caller truyền top_k/size
# lớn hơn, kết quả cũng bị cắt về tối đa giá trị này để tránh trả
# về quá nhiều hồ sơ/đoạn text không cần thiết cho chatbot.
MAX_TOP_K = 5


def _clamp_top_k(value: int) -> int:
    """Ép value không vượt quá MAX_TOP_K (vẫn cho phép nhỏ hơn nếu caller muốn ít hơn)."""
    if value is None:
        return MAX_TOP_K
    return min(value, MAX_TOP_K)


def _extract_file_key(file_url: str) -> Optional[str]:
    """Tách 'key' từ URL dạng .../files/proxy?key=xxx — trả về đã decode sẵn."""
    parsed = urlparse(file_url)
    qs = parse_qs(parsed.query)
    values = qs.get("key")
    return values[0] if values else None

def catch_tool_errors(func):
    """
    Decorator dùng cho MỌI tool method bên dưới: bắt toàn bộ exception
    (kết nối Qdrant/Archive API bị từ chối, timeout, token sai, dữ
    liệu thiếu field...), log FULL traceback ra log server (để debug),
    đồng thời trả về 1 dict lỗi có cấu trúc rõ ràng thay vì để MCP
    framework tự bọc thành text cụt lủn kiểu
    "Error executing tool X: [WinError 10061] ...".

    Nhờ vậy mỗi lần lỗi, Postman sẽ thấy NGAY nguyên nhân (error_type,
    message) mà không cần lục log server; còn log server thì có đủ
    traceback để debug sâu hơn nếu cần.
    """

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            # logger.exception() tự đính kèm traceback đầy đủ vào log,
            # khác với logger.error(str(e)) chỉ có 1 dòng thông báo
            logger.exception(f"Lỗi khi thực thi tool '{func.__name__}'")
            error_payload = {
                "error": True,
                "tool": func.__name__,
                "error_type": type(e).__name__,
                "message": str(e),
            }
            # Chỉ trả traceback ra response khi DEBUG_TOOL_ERRORS=true —
            # tiện lúc test bằng Postman, nhưng nên tắt ở production để
            # không lộ chi tiết nội bộ (đường dẫn file, stack nội bộ...)
            # cho chatbot/người dùng cuối thấy.
            if config_object.DEBUG_TOOL_ERRORS:
                error_payload["traceback"] = traceback.format_exc()
            return error_payload

    return wrapper


class FeatureManager:

    @staticmethod
    @catch_tool_errors
    async def get_profile_detail(archive_id: str, question: str, top_k: int = MAX_TOP_K) -> dict:
        """
        Trả lời câu hỏi chi tiết TRONG PHẠM VI 1 hồ sơ cụ thể. MCP chỉ làm
        nhiệm vụ RETRIEVAL - trả về các đoạn văn bản liên quan nhất kèm
        nguồn (file_url, page_number). Việc tổng hợp thành câu trả lời tự
        nhiên là do LLM phía chatbot đảm nhận (tầng Generation của RAG).

        top_k luôn bị ép về tối đa MAX_TOP_K (5), kể cả khi caller truyền
        giá trị lớn hơn.
        """
        service = get_retrieval_service()
        chunks = service.search_chunks_in_archive(
            archive_id=archive_id, question=question, top_k=_clamp_top_k(top_k)
        )

        if not chunks:
            return {
                "archive_id": archive_id,
                "question": question,
                "found": False,
                "message": "Không tìm thấy nội dung liên quan trong hồ sơ này.",
                "chunks": [],
            }

        return {
            "archive_id": archive_id,
            "question": question,
            "found": True,
            "chunks": [
                {
                    "text": c.text,
                    "file_url": c.file_url,
                    "page_number": c.page_number,
                    "extraction_method": c.extraction_method,
                    "score": c.score,
                }
                for c in chunks
            ],
        }

    # ────────────────────────────────────────────────────────────────
    # TOOL: Find Profile And Answer — kết hợp tìm hồ sơ theo `key` (VD
    # tên người) + trả lời `question` trong CHÍNH nội dung MD của hồ sơ
    # đó. Dùng khi câu hỏi gắn với 1 hồ sơ cụ thể nhưng chưa biết
    # archive_id, VD: "Lê Minh Tuấn được quyết định tăng lương vào
    # ngày nào?" -> key="Lê Minh Tuấn",
    # question="quyết định tăng lương vào ngày nào".
    # ────────────────────────────────────────────────────────────────
    @staticmethod
    @catch_tool_errors
    async def find_profile_and_answer(key: str, question: str, top_k: int = MAX_TOP_K) -> dict:
        """
        Tìm hồ sơ khớp nhất với `key` (tên người, mã hồ sơ...), sau đó
        trả lời `question` bằng cách tìm đoạn text liên quan nhất TRONG
        CHÍNH hồ sơ đó (nội dung file MD đã ingest sẵn). Dùng khi câu
        hỏi của người dùng gắn với 1 hồ sơ/1 người cụ thể nhưng chưa có
        archive_id — tool này tự tìm hồ sơ bằng semantic search rồi trả
        lời trong chính hồ sơ đó, gộp cả 2 bước trong 1 lần gọi.

        top_k luôn bị ép về tối đa MAX_TOP_K (5).
        """
        service = get_retrieval_service()
        profile, chunks = service.find_profile_and_answer(
            key=key, question=question, top_k=_clamp_top_k(top_k)
        )

        if profile is None:
            return {
                "key": key,
                "question": question,
                "found": False,
                "message": f"Không tìm thấy hồ sơ nào khớp với '{key}'. "
                           f"Hãy báo người dùng là không có kết quả, đừng tự suy đoán hoặc bịa thông tin.",
            }

        return {
            "key": key,
            "question": question,
            "found": bool(chunks),
            "matched_profile": {
                "archive_id": profile.archive_id,
                "title": profile.title,
                "arcFileCode": profile.arc_file_code,
                "score": profile.score,
            },
            "message": None if chunks else (
                "Tìm thấy hồ sơ khớp với key nhưng không có đoạn nội dung nào "
                "liên quan đến câu hỏi. Đừng tự suy đoán hoặc bịa thông tin."
            ),
            "chunks": [
                {
                    "text": c.text,
                    "file_url": c.file_url,
                    "page_number": c.page_number,
                    "extraction_method": c.extraction_method,
                    "score": c.score,
                }
                for c in chunks
            ],
        }

    # ────────────────────────────────────────────────────────────────
    # TOOL: Search Content — tìm nội dung xuyên suốt TẤT CẢ hồ sơ đã
    # ingest (không giới hạn 1 archive_id cụ thể). Dùng cho câu hỏi
    # kiểu liệt kê/khám phá khi chưa biết trước hồ sơ nào liên quan,
    # VD: "tìm những hồ sơ là nông dân", "hồ sơ nào có bố mẹ làm nông
    # nghiệp".
    # ────────────────────────────────────────────────────────────────
    @staticmethod
    @catch_tool_errors
    async def search_content(question: str, top_k: int = MAX_TOP_K) -> dict:
        """
        Tìm đoạn text liên quan nhất đến `question` TRÊN TOÀN BỘ hồ sơ
        đã ingest, KHÔNG giới hạn 1 hồ sơ cụ thể. Dùng khi câu hỏi có
        thể khớp với NHIỀU hồ sơ khác nhau và chưa biết trước hồ sơ
        nào (khác với find_profile_and_answer — tool đó cần biết
        `key` của 1 hồ sơ cụ thể trước). Mỗi kết quả trả về kèm
        archive_id để biết đoạn đó thuộc hồ sơ nào — muốn xem chi tiết
        hồ sơ đó thì gọi tiếp get_profile_detail.

        top_k luôn bị ép về tối đa MAX_TOP_K (5).
        """
        service = get_retrieval_service()
        chunks = service.search_chunks_all(question=question, top_k=_clamp_top_k(top_k))

        return {
            "question": question,
            "found": bool(chunks),
            "note": (
                "Đây CHỈ LÀ vài đoạn text giống câu hỏi nhất (tối đa "
                f"{MAX_TOP_K}), KHÔNG PHẢI danh sách/số lượng đầy đủ. "
                "Nếu người dùng hỏi 'có bao nhiêu'/'tất cả'/'liệt kê "
                "đầy đủ', KHÔNG được suy ra 1 con số hay danh sách "
                "hoàn chỉnh từ đây — phải nói rõ hệ thống chỉ tìm được "
                "vài ví dụ gần đúng nhất, không xác nhận được tổng số. "
                "Điểm cao cũng không đảm bảo đúng LOẠI văn bản người "
                "dùng hỏi — đọc kỹ nội dung 'text' trước khi khẳng định."
            ),
            "message": None if chunks else (
                "Không tìm thấy đoạn nội dung nào liên quan trong bất kỳ hồ sơ nào. "
                "Hãy báo người dùng là không có kết quả, đừng tự suy đoán hoặc bịa thông tin."
            ),
            "chunks": [
                {
                    "archive_id": c.archive_id,
                    "text": c.text,
                    "file_url": c.file_url,
                    "page_number": c.page_number,
                    "project_name": c.project_name,
                    "extraction_method": c.extraction_method,
                    "score": c.score,
                }
                for c in chunks
            ],
        }

    # ────────────────────────────────────────────────────────────────
    # TOOL 3: Search Archives — tìm danh sách hồ sơ theo field lọc
    # CHÍNH XÁC (status, kho, ngôn ngữ, ngày tạo/sửa...), gọi trực tiếp
    # Public Archive API, không qua Qdrant/semantic search.
    # ────────────────────────────────────────────────────────────────
    @staticmethod
    @catch_tool_errors
    async def search_archives(
            keywords: list[str] = None,
            status: str = None,
            warehouse_id: str = None,
            language: str = None,
            maintenance: str = None,
            created_from: str = None,
            created_to: str = None,
            updated_from: str = None,
            updated_to: str = None,
            page: int = 0,
            size: int = MAX_TOP_K,
    ) -> dict:
        """
        Tìm hồ sơ lưu trữ. Luôn thử khớp CHÍNH XÁC theo keywords/filter
        trên hệ thống live trước. CHỈ khi không có kết quả nào (0 hồ sơ)
        VÀ có keywords, tool mới tự động mở rộng sang semantic search
        trên dữ liệu đã index sẵn (bắt các trường hợp từ khóa mơ hồ về
        nghĩa, VD "làm nông" -> "nông dân"). Người gọi không cần tự
        chọn cách tìm, tool tự quyết định.

        `keywords`: truyền 1 HOẶC NHIỀU biến thể từ khóa (VD tên có dấu
        + không dấu + viết tắt) trong CÙNG 1 lần gọi tool này — KHÔNG tự
        gọi lại tool nhiều lần cho từng biến thể, tool đã tự gộp kết quả
        (khử trùng lặp theo id) trong 1 lượt duy nhất, tiết kiệm round-trip.

        size luôn bị ép về tối đa MAX_TOP_K (5), kể cả khi caller truyền
        giá trị lớn hơn, hoặc khi live API trả về nhiều hơn 5 hồ sơ.
        """
        size = _clamp_top_k(size)
        clean_keywords = [k for k in (keywords or []) if k and k.strip()]
        client = get_archive_api_client()
        result = await client.search_archives(
            keywords=clean_keywords, status=status, warehouse_id=warehouse_id,
            language=language, maintenance=maintenance,
            created_from=created_from, created_to=created_to,
            updated_from=updated_from, updated_to=updated_to,
            page=page, size=size,
        )

        # Phòng trường hợp live API không tôn trọng "size" (trả về nhiều
        # hơn yêu cầu) — vẫn cắt về tối đa MAX_TOP_K trước khi trả ra.
        if len(result.get("content", [])) > MAX_TOP_K:
            result["content"] = result["content"][:MAX_TOP_K]
            page_info = result.get("page") or {}
            page_info["size"] = MAX_TOP_K
            result["page"] = page_info

        for record in result.get("content", []):
            record["hasFiles"] = bool(record.get("projects"))
            for project in record.get("projects", []):
                project["fileKeys"] = [
                    _extract_file_key(u) for u in project.get("fileUrls", [])
                ]

        if result.get("content"):
            result["search_mode"] = "keyword"
            result["found"] = True
            return result

        has_other_filters = any([status, warehouse_id, language, maintenance,
                                 created_from, created_to, updated_from, updated_to])
        if not clean_keywords or has_other_filters:
            result["search_mode"] = "keyword"
            result["found"] = False
            result["message"] = (
                "Không tìm thấy hồ sơ nào khớp với điều kiện tìm kiếm. "
                "Hãy báo người dùng là không có kết quả, đừng tự suy đoán hoặc bịa thông tin hồ sơ."
            )
            return result

        # Semantic search chỉ nhận 1 chuỗi keyword -> gộp các biến thể lại
        # thành 1 câu, đủ để embedding bắt được nghĩa chung.
        merged_keyword = " ".join(clean_keywords)
        service = get_retrieval_service()
        profiles = service.search_profiles(keyword=merged_keyword, top_k=_clamp_top_k(size))

        if not profiles:
            return {
                "search_mode": "semantic_fallback",
                "keywords": clean_keywords,
                "found": False,
                "message": (
                    "Đã tìm chính xác lẫn tìm theo nghĩa gần đúng nhưng không thấy hồ sơ nào phù hợp. "
                    "Hãy báo người dùng là không có kết quả, đừng tự suy đoán hoặc bịa thông tin hồ sơ."
                ),
                "content": [],
                "page": {"size": size, "number": 0, "totalElements": 0, "totalPages": 0},
            }

        return {
            "search_mode": "semantic_fallback",
            "keywords": clean_keywords,
            "found": True,
            "content": [
                {
                    "id": p.archive_id,
                    "title": p.title,
                    "arcFileCode": p.arc_file_code,
                    "shelfCode": p.shelf_code,
                    "shelfLevelCode": p.shelf_level_code,
                    "warehouseName": p.warehouse_name,
                    "startDate": p.start_date,
                    "endDate": p.end_date,
                    "staffMetadata": p.staff_metadata,
                    "_score": p.score,
                }
                for p in profiles
            ],
            "page": {"size": size, "number": 0, "totalElements": len(profiles), "totalPages": 1},
        }