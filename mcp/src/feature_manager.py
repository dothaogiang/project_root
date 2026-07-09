"""
FeatureManager: chứa các hàm nghiệp vụ thực thi khi tool MCP được gọi.

QUAN TRỌNG: tên hàm PHẢI trùng chính xác với `name_tool` khai báo trong
Resources/tools.yaml. ToolRegistry (tools/registry.py) sẽ tự động match
theo tên -> đây là cơ chế cho phép thêm/sửa/xóa tool chỉ bằng cách sửa
file YAML, không cần đụng vào registry.py hay server.py.

2 nhóm tool:
  - search_profile / get_profile_detail: semantic search qua Qdrant
    (dữ liệu đã ingest sẵn bởi rag/), xem rag/README.md.
  - search_archives / get_archive_detail / get_staff_archive_metadata /
    get_file_proxy: gọi TRỰC TIẾP (live) vào Public Archive API qua
    archive_api/client.py, không qua Qdrant/embedding.
"""
import base64
import functools
import traceback

from archive_api.client import MAX_INLINE_FILE_BYTES, get_archive_api_client
from config.configs import config_object
from rag.retrieval_factory import get_retrieval_service
from logger import get_logger

logger = get_logger(__name__)


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
    async def search_profile(keyword: str, top_k: int = 10) -> dict:
        """
        Tìm hồ sơ (archive) theo từ khóa tự do, hybrid search (dense+sparse).
        Đây là tầng "định danh hồ sơ" - dùng archive_id trả về để gọi tiếp
        get_profile_detail nếu cần hỏi sâu vào nội dung.
        """
        service = get_retrieval_service()
        profiles = service.search_profiles(keyword=keyword, top_k=top_k)

        return {
            "keyword": keyword,
            "total_found": len(profiles),
            "profiles": [
                {
                    "archive_id": p.archive_id,
                    "title": p.title,
                    "arcFileCode": p.arc_file_code,
                    "boxCode": p.box_code,
                    "warehouseName": p.warehouse_name,
                    "startDate": p.start_date,
                    "endDate": p.end_date,
                    "staffMetadata": p.staff_metadata,
                    "score": p.score,
                }
                for p in profiles
            ],
        }

    @staticmethod
    @catch_tool_errors
    async def get_profile_detail(archive_id: str, question: str, top_k: int = 5) -> dict:
        """
        Trả lời câu hỏi chi tiết TRONG PHẠM VI 1 hồ sơ cụ thể. MCP chỉ làm
        nhiệm vụ RETRIEVAL - trả về các đoạn văn bản liên quan nhất kèm
        nguồn (file_url, page_number). Việc tổng hợp thành câu trả lời tự
        nhiên là do LLM phía chatbot đảm nhận (tầng Generation của RAG).
        """
        service = get_retrieval_service()
        chunks = service.search_chunks_in_archive(archive_id=archive_id, question=question, top_k=top_k)

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
    # TOOL 3: Search Archives — tìm danh sách hồ sơ theo field lọc
    # CHÍNH XÁC (status, kho, ngôn ngữ, ngày tạo/sửa...), gọi trực tiếp
    # Public Archive API, không qua Qdrant/semantic search.
    # ────────────────────────────────────────────────────────────────
    @staticmethod
    @catch_tool_errors
    async def search_archives(
        keyword: str = None,
        status: str = None,
        warehouse_id: str = None,
        language: str = None,
        maintenance: bool = None,
        created_from: str = None,
        created_to: str = None,
        updated_from: str = None,
        updated_to: str = None,
        page: int = 0,
        size: int = 20,
    ) -> dict:
        """
        Tìm kiếm danh sách hồ sơ lưu trữ trực tiếp trên hệ thống, lọc
        theo field CHÍNH XÁC (status, kho, ngôn ngữ, khoảng ngày tạo/
        sửa...). Dùng khi người dùng hỏi "tìm hồ sơ trong kho X", "hồ sơ
        tiếng Việt", "hồ sơ tạo trong khoảng ngày...". Nếu chưa biết
        archive_id, LUÔN dùng tool này trước.

        CHỈ trả về ĐÚNG 1 TRANG mỗi lần gọi (mặc định page=0, size=20) —
        không tự động gộp nhiều trang, để tránh dồn quá nhiều hồ sơ vào
        1 lần trả lời. Nếu KHÔNG thấy hồ sơ phù hợp trong kết quả trang
        này VÀ "last" trong kết quả là false (còn trang tiếp theo), hãy
        GỌI LẠI tool này với page tăng thêm 1 để tìm tiếp — thường trang
        đầu đã đủ vì filter đã thu hẹp kết quả, chỉ cần tìm tiếp khi
        thật sự chưa thấy.
        """
        client = get_archive_api_client()
        result = await client.search_archives(
            keyword=keyword,
            status=status,
            warehouse_id=warehouse_id,
            language=language,
            maintenance=maintenance,
            created_from=created_from,
            created_to=created_to,
            updated_from=updated_from,
            updated_to=updated_to,
            page=page,
            size=size,
        )
        return result

    # ────────────────────────────────────────────────────────────────
    # TOOL 4: Get Archive Detail — lấy toàn bộ chi tiết 1 hồ sơ (metadata,
    # project, lịch sử mượn) theo UUID đã biết trước.
    # ────────────────────────────────────────────────────────────────
    @staticmethod
    @catch_tool_errors
    async def get_archive_detail(archive_id: str) -> dict:
        """
        Lấy TOÀN BỘ chi tiết 1 hồ sơ (metadata, project, borrowItems -
        lịch sử mượn hồ sơ) trực tiếp từ hệ thống lưu trữ theo UUID.
        CHỈ dùng khi ĐÃ BIẾT archive_id (lấy từ search_archives hoặc
        search_profile) — không dùng để tìm kiếm nhiều hồ sơ.
        """
        client = get_archive_api_client()
        return await client.get_archive_detail(archive_id)

    # ────────────────────────────────────────────────────────────────
    # TOOL 5: Staff Archive Metadata — cấu trúc dữ liệu hồ sơ cán bộ
    # (schema field, documentTypes), không phải tìm hồ sơ cụ thể.
    # ────────────────────────────────────────────────────────────────
    @staticmethod
    @catch_tool_errors
    async def get_staff_archive_metadata(only_metadata: bool = True) -> dict:
        """
        Lấy cấu trúc dữ liệu (schema) của hồ sơ cán bộ: danh sách các
        trường metadata, và nếu only_metadata=false thì lấy thêm cả
        documentTypes. Dùng khi người dùng hỏi "hồ sơ cán bộ gồm những
        trường nào", "cấu trúc hồ sơ cán bộ", "document types" — KHÔNG
        dùng để tìm một hồ sơ cụ thể.
        """
        client = get_archive_api_client()
        return await client.get_staff_archive_metadata(only_metadata=only_metadata)

    # ────────────────────────────────────────────────────────────────
    # TOOL 6: File Proxy — lấy nội dung file gốc đính kèm hồ sơ (PDF,
    # ảnh...) để xem/tải/mở.
    # ────────────────────────────────────────────────────────────────
    @staticmethod
    @catch_tool_errors
    async def get_file_proxy(key: str, file_name: str) -> dict:
        """
        Lấy nội dung file gốc đính kèm hồ sơ (PDF, ảnh...), dùng khi
        người dùng muốn xem/tải/mở file đính kèm/tài liệu gốc. `key` và
        `file_name` lấy từ project.fileUrls hoặc metadata trả về bởi
        get_archive_detail. File nhỏ (<8MB) trả về base64 để hiển thị/
        tải trực tiếp; file lớn hơn chỉ trả metadata (kích thước, loại
        file) kèm cảnh báo, tránh làm phình phản hồi tool.
        """
        client = get_archive_api_client()
        content, content_type = await client.get_file_proxy(key=key, file_name=file_name)

        if len(content) > MAX_INLINE_FILE_BYTES:
            return {
                "key": key,
                "file_name": file_name,
                "content_type": content_type,
                "size_bytes": len(content),
                "too_large": True,
                "message": (
                    f"File nặng {len(content) / 1024 / 1024:.1f}MB, vượt ngưỡng "
                    f"{MAX_INLINE_FILE_BYTES / 1024 / 1024:.0f}MB để trả inline qua tool. "
                    f"Hãy dùng đường dẫn gốc (fileUrls) từ get_archive_detail để tải trực tiếp."
                ),
            }

        return {
            "key": key,
            "file_name": file_name,
            "content_type": content_type,
            "size_bytes": len(content),
            "too_large": False,
            "content_base64": base64.b64encode(content).decode("ascii"),
        }
