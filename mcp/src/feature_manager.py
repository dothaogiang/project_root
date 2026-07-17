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

# Số ĐOẠN TEXT (chunk nội dung) tối đa trả về cho các tool RAG:
# get_profile_detail, find_profile_and_answer, search_content. Giữ nhỏ
# vì mỗi chunk tốn nhiều token (kèm nội dung + nguồn).
MAX_TOP_K = 5

# Số HỒ SƠ (metadata) tối đa trả về riêng cho search_archives. Tách
# riêng khỏi MAX_TOP_K vì đây chỉ là liệt kê hồ sơ (rẻ hơn nhiều so với
# 1 chunk nội dung) — trước đây dùng chung MAX_TOP_K=5 khiến size=20
# LLM truyền vào bị ép cứng xuống 5, làm hồ sơ đúng nhưng bị Archive
# API xếp hạng thấp (VD "Phạm Thị Hoa" nằm ngoài top 5 của 14 kết quả
# khớp "Hoa") không bao giờ lọt vào response, dù vẫn tồn tại trong dữ
# liệu. 20 vẫn là mức trần hợp lý để tránh vượt context quá nhiều.
MAX_ARCHIVE_LIST = 20


def _clamp_top_k(value: int, max_value: int = MAX_TOP_K) -> int:
    """Ép value không vượt quá max_value (vẫn cho phép nhỏ hơn nếu caller muốn ít hơn)."""
    if value is None:
        return max_value
    return min(value, max_value)


def _extract_file_key(file_url: Optional[str]) -> Optional[str]:
    """Tách 'key' từ URL dạng .../files/proxy?key=xxx — trả về đã decode sẵn."""
    if not file_url:
        return None
    parsed = urlparse(file_url)
    qs = parse_qs(parsed.query)
    values = qs.get("key")
    return values[0] if values else None


def _normalize_vn(text: Optional[str]) -> str:
    """Chỉ lowercase, CỐ Ý KHÔNG bỏ dấu thanh (sắc/huyền/hỏi/ngã/nặng) —
    tiếng Việt có dấu là 2 từ khác nghĩa (VD "Hoa" tên người khác hẳn
    "Hóa" trong "Hóa đơn"/"Hòa" trong "Hòa Khánh"). Bỏ dấu sẽ khiến các
    từ này bị coi là khớp nhau, đúng vào lỗi đang cần sửa (search "Hoa"
    ra nhầm "Hóa đơn", "Hòa Khánh")."""
    return (text or "").lower()


def _compact_record(record: dict) -> dict:
    """Bỏ các field null/rỗng khỏi record (staffMetadata: [], shelfCode:
    null...) — không mất thông tin (thiếu field = rỗng), chỉ giảm số ký
    tự phải trả về khi có nhiều hồ sơ, phần lớn hồ sơ không dùng hết
    các field optional này."""
    return {k: v for k, v in record.items() if v not in (None, [], "")}


# Field thiết yếu để LIỆT KÊ/xác định hồ sơ (dùng cho search_archives
# khi brief=True — mặc định, tối ưu cho lần gọi đầu tiên). Có "files"
# (fileName + fileKey + fileUrl) để LLM/người dùng lấy link file ngay
# ở lần gọi đầu, không cần gọi thêm brief=False chỉ để lấy link.
BRIEF_FIELDS = {
    "id", "title", "arcFileCode", "status", "warehouseName",
    "roomNumber", "shelfCode", "shelfLevelCode",
    "startDate", "endDate", "hasFiles", "title_match", "files",
}

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


def _files_from_record(record: dict) -> list[dict]:
    """Trích 'files' (fileName + fileKey + fileUrl) từ 1 record thô của
    Public Archive API (projects[].documents[]) — dùng chung cho cả
    nhánh keyword-match và nhánh enrich semantic_fallback bên dưới."""
    files = []
    for project in record.get("projects", []):
        for doc in project.get("documents", []):
            files.append({
                "fileName": doc.get("fileName"),
                "fileKey": _extract_file_key(doc.get("fileUrl")),
                "fileUrl": doc.get("fileUrl"),
            })
    return files


async def _fetch_files_for_profiles(client, profiles: list) -> dict[str, list[dict]]:
    """Qdrant (semantic search) KHÔNG lưu file/fileUrl trong payload —
    RetrievedProfile chỉ có metadata hồ sơ (title, mã, kệ, kho...).
    Để trả link thật cho các hồ sơ tìm được qua semantic_fallback, gọi
    lại LIVE Archive API theo `arcFileCode` (định danh khớp CHÍNH XÁC,
    không mơ hồ như tên người) rồi lấy files từ record trả về. Chỉ 1
    lượt gọi (client tự fan-out song song nếu nhiều mã), không cần
    re-ingest lại toàn bộ Qdrant chỉ để thêm field này.

    Trả về dict archive_id (str) -> list files. Nếu gọi lỗi (API down,
    timeout...), trả về {} và log lỗi — KHÔNG làm hỏng luôn cả response
    semantic_fallback chỉ vì bước enrich phụ này thất bại.
    """
    codes = list({p.arc_file_code for p in profiles if p.arc_file_code})
    if not codes:
        return {}
    try:
        result = await client.search_archives(keywords=codes, size=max(len(codes) * 2, 20))
    except Exception:
        logger.exception("Không lấy được files/link thật cho semantic fallback profiles")
        return {}

    files_by_id: dict[str, list[dict]] = {}
    for record in result.get("content", []):
        rid = record.get("id")
        if rid is None:
            continue
        files = _files_from_record(record)
        if files:
            files_by_id[str(rid)] = files
    return files_by_id


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
            size: int = MAX_ARCHIVE_LIST,
            brief: bool = True,
            include_full_content: bool = False,
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

        size luôn bị ép về tối đa MAX_ARCHIVE_LIST (20), kể cả khi caller
        truyền giá trị lớn hơn, hoặc khi live API trả về nhiều hồ sơ hơn.

        3 MỨC chi tiết trả về, tăng dần payload (chọn mức nhỏ nhất đủ dùng):

        1. `brief=True` (MẶC ĐỊNH — dùng cho LẦN GỌI ĐẦU TIÊN/liệt kê
           nhiều hồ sơ): mỗi hồ sơ gồm id, title, arcFileCode,
           status, warehouseName, roomNumber, shelfCode, shelfLevelCode,
           startDate, endDate, hasFiles, title_match, và `files` (danh
           sách gọn: fileName + fileKey + fileUrl — đủ để trả link file
           ngay lần gọi đầu). Bỏ description, createdAt/updatedAt,
           language, maintenance, staffMetadata, borrowItems. Đủ để
           liệt kê + xác định đúng hồ sơ cần xem tiếp, kèm sẵn link tải.
        2. `brief=False, include_full_content=False`: thêm lại
           description, createdAt/updatedAt, language, maintenance,
           staffMetadata, borrowItems, danh sách `files` (tên file +
           fileKey) — dùng khi đã thu hẹp còn ít hồ sơ và cần các field
           phụ này để trả lời (VD lịch sử mượn trả).
        3. `include_full_content=True`: thêm cả nội dung Markdown OCR
           đầy đủ + fileUrls — CHỈ dùng khi đã xác định chắc chắn đúng
           1 hồ sơ VÀ cần đọc nội dung ngay trong lượt này. Nếu không,
           gọi `get_profile_detail(archive_id=...)` sau khi đã chọn
           đúng hồ sơ (tool đó chỉ cần archive_id, không cần fileKey).

        LƯU Ý: Public Archive API không đảm bảo sắp xếp theo độ liên quan
        khi khớp keyword — 1 hồ sơ khớp đúng vẫn có thể bị xếp ngoài
        trang trả về nếu tổng số khớp (totalElements) lớn hơn `size`.
        Trường hợp đó, tool tự bổ sung thêm ứng viên từ semantic search
        (đánh dấu "_source": "semantic_fallback" trong từng bản ghi bổ
        sung, và "search_mode": "keyword+semantic_fallback") để tăng khả
        năng không bỏ sót hồ sơ đúng, thay vì chỉ dừng lại ở trang đầu.
        """
        size = _clamp_top_k(size, MAX_ARCHIVE_LIST)
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
        # hơn yêu cầu) — vẫn cắt về tối đa `size` (đã clamp theo
        # MAX_ARCHIVE_LIST ở trên) trước khi trả ra.
        if len(result.get("content", [])) > size:
            result["content"] = result["content"][:size]
            page_info = result.get("page") or {}
            page_info["size"] = size
            result["page"] = page_info

        normalized_keywords = [_normalize_vn(k) for k in clean_keywords]

        for record in result.get("content", []):
            record["hasFiles"] = bool(record.get("projects"))

            if include_full_content:
                for project in record.get("projects", []):
                    project["fileKeys"] = [
                        _extract_file_key(u) for u in project.get("fileUrls", [])
                    ]
            else:
                # Cắt gọn: thay "projects" (kèm fileUrls dài + markdown OCR
                # đầy đủ của từng document, rất nặng) bằng 1 danh sách
                # "files" phẳng, chỉ gồm tên file + fileKey — đủ để LLM
                # biết hồ sơ có file gì và gọi get_profile_detail khi cần
                # đọc nội dung, mà không phải tải kèm toàn bộ markdown OCR
                # của MỌI hồ sơ khớp keyword (kể cả hồ sơ không liên quan).
                files = _files_from_record(record)
                record["files"] = files
                record.pop("projects", None)

            # Public Archive API khớp keyword trên NHIỀU field ẩn (không
            # chỉ title) nên trả về nhiều hồ sơ chỉ liên quan mờ nhạt
            # (VD tìm "Hoa" ra cả "KCN Hòa Khánh", hồ sơ TEST không thấy
            # "Hoa" ở đâu). "title_match" đánh dấu record có khớp NGAY
            # TRONG title/arcFileCode hay không (bỏ dấu, không phân biệt
            # hoa/thường) — KHÔNG loại bỏ các record còn lại (tránh mất
            # dữ liệu, vì có thể khách vẫn cần các field khác), chỉ gắn
            # cờ để LLM biết ưu tiên record nào khi trả lời/tóm tắt.
            haystack = _normalize_vn(f"{record.get('title', '')} {record.get('arcFileCode', '')}")
            record["title_match"] = bool(normalized_keywords) and any(
                kw in haystack for kw in normalized_keywords
            )

        # Đưa record khớp title lên đầu danh sách (ổn định thứ tự trong
        # từng nhóm) — giúp LLM đọc/trả lời được ngay mà không cần tự
        # rà qua toàn bộ 14 hồ sơ để tìm ra 1-2 hồ sơ thực sự liên quan.
        result["content"] = sorted(
            result.get("content", []),
            key=lambda r: not r.get("title_match", False),
        )

        if include_full_content:
            pass  # đã giữ nguyên projects/documents/markdown/fileUrls ở nhánh trên
        elif brief:
            # Tier gọn nhất — dùng cho lần liệt kê đầu tiên: chỉ giữ field
            # thiết yếu để nhận diện + hiển thị hồ sơ, bỏ hẳn files/
            # fileKey (không tool nào khác trong hệ thống dùng tới),
            # borrowItems, staffMetadata, description, ngày tạo/sửa...
            result["content"] = [
                _compact_record({k: v for k, v in record.items() if k in BRIEF_FIELDS})
                for record in result["content"]
            ]
        else:
            result["content"] = [_compact_record(r) for r in result["content"]]

        if result.get("content"):
            result["search_mode"] = "keyword"
            result["found"] = True

            page_info = result.get("page") or {}
            total_elements = page_info.get("totalElements", len(result["content"]))

            # Archive API KHÔNG đảm bảo sắp xếp theo độ liên quan — nếu
            # tổng số khớp keyword (totalElements) nhiều hơn số bản ghi
            # đã trả về, hồ sơ đúng vẫn có thể bị rớt ngoài trang này
            # (VD tìm "Hoa" ra 14 hồ sơ khớp nhưng "Phạm Thị Hoa" không
            # nằm trong 5-20 bản ghi đầu). Trường hợp đó, bổ sung thêm
            # ứng viên qua semantic search (Qdrant) để tăng recall,
            # thay vì coi "có kết quả nào đó" là đã tìm đủ.
            if clean_keywords and total_elements > len(result["content"]):
                merged_keyword = " ".join(clean_keywords)
                service = get_retrieval_service()
                extra_profiles = service.search_profiles(keyword=merged_keyword, top_k=size)
                extra_files_by_id = await _fetch_files_for_profiles(client, extra_profiles)

                existing_ids = {
                    str(record.get("id")) for record in result["content"]
                    if record.get("id") is not None
                }
                appended = [
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
                        "hasFiles": bool(extra_files_by_id.get(str(p.archive_id))),
                        "files": extra_files_by_id.get(str(p.archive_id), []),
                        "_score": p.score,
                        "_source": "semantic_fallback",
                    }
                    for p in extra_profiles
                    if str(p.archive_id) not in existing_ids
                ]

                if appended:
                    result["content"].extend(appended)
                    result["search_mode"] = "keyword+semantic_fallback"
                    result["note"] = (
                        f"Live API báo tổng cộng {total_elements} hồ sơ khớp từ khóa nhưng "
                        f"chỉ {total_elements - len(appended)} hồ sơ đầu được trả về ở trang này "
                        "(Archive API không sắp xếp theo độ liên quan). Đã bổ sung thêm "
                        f"{len(appended)} ứng viên qua semantic search (đánh dấu "
                        "\"_source\": \"semantic_fallback\" trong từng bản ghi) để giảm khả năng "
                        "bỏ sót hồ sơ đúng. Hãy xem cả 2 loại kết quả trước khi trả lời."
                    )

            return result

        if not clean_keywords:
            # Không có keywords thì không có gì để semantic search -> chỉ
            # có thể dựa vào filter chính xác, mà filter đã 0 kết quả rồi.
            result["search_mode"] = "keyword"
            result["found"] = False
            result["message"] = (
                "Không tìm thấy hồ sơ nào khớp với điều kiện tìm kiếm. "
                "Hãy báo người dùng là không có kết quả, đừng tự suy đoán hoặc bịa thông tin hồ sơ."
            )
            return result

        # Có keywords nhưng khớp chính xác ra 0 kết quả -> luôn thử semantic
        # fallback, GIỐNG HỆT cách nhánh "có kết quả nhưng thiếu" phía trên
        # đang làm (nhánh đó không chặn theo has_other_filters). Trước đây
        # nhánh này chặn semantic fallback bất cứ khi nào có thêm filter
        # khác (VD created_from) khiến hành vi giữa 2 nhánh không nhất
        # quán: cùng 1 từ khóa gần đúng nhưng có kèm created_from thì bị
        # coi là "không tìm thấy" thay vì được gợi ý ứng viên gần đúng.
        has_other_filters = any([status, warehouse_id, language, maintenance,
                                 created_from, created_to, updated_from, updated_to])

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

        fallback_note = None
        if has_other_filters:
            fallback_note = (
                # "Live API không tìm thấy hồ sơ khớp CHÍNH XÁC theo keywords và các "
                # "filter đã truyền (status/warehouse/ngôn ngữ/ngày...). Danh sách dưới "
                # "đây là ứng viên GẦN ĐÚNG lấy từ semantic search — LƯU Ý: semantic "
                # "search KHÔNG áp dụng lại các filter đó, nên các hồ sơ này có thể nằm "
                # "ngoài điều kiện status/warehouse/ngày đã yêu cầu, cần tự kiểm tra lại "
                # "trước khi trả lời. Đừng khẳng định chắc chắn đây là hồ sơ người dùng "
                # "cần, hãy hỏi lại để xác nhận nếu có nhiều ứng viên."
            )

        files_by_id = await _fetch_files_for_profiles(client, profiles)

        return {
            "search_mode": "semantic_fallback",
            "keywords": clean_keywords,
            "found": True,
            "note": fallback_note,
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
                    "hasFiles": bool(files_by_id.get(str(p.archive_id))),
                    "files": files_by_id.get(str(p.archive_id), []),
                    "_score": p.score,
                }
                for p in profiles
            ],
            "page": {"size": size, "number": 0, "totalElements": len(profiles), "totalPages": 1},
        }