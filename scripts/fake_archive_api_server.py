"""
scripts/fake_archive_api_server.py — Fake/mock Archive API để test pipeline
ingestion phần MD (rag/jobs/sync_job.py) khi chưa có/không tới được Archive
API thật (mặc định http://192.168.1.46:4000).

CHỈ dùng thư viện chuẩn Python (http.server) — KHÔNG thêm dependency mới
vào rag/requirements.txt hay mcp/requirements.txt.

CÁCH DÙNG:
    1. Chạy server này:
           python scripts/fake_archive_api_server.py
       (mặc định lắng nghe 0.0.0.0:8000, đổi bằng --port nếu cần)

    2. Trong file .env ở gốc project, trỏ tạm:
           ARCHIVE_API_BASE_URL=http://localhost:8000
           ARCHIVE_API_PATH=/api/public/archives

    3. Chạy sync job như bình thường:
           python -m rag.jobs.sync_job

    4. Xong việc test thì đổi ARCHIVE_API_BASE_URL lại về API thật.

SERVER GIẢ LẬP 2 ENDPOINT:
    GET /api/public/archive?page=0&size=100
        -> danh sách hồ sơ giả (JSON), đúng field mà
           rag/infrastructure/archive_api_client.py._to_archive_record()
           đọc (title, arcFileCode, shelfCode, projects[].mdFileUrls...),
           có phân trang thật (page/totalPages) để test luôn vòng lặp
           while True trong IngestionService.run().

    GET /files/<key>.md
        -> nội dung 1 file .md giả (bytes UTF-8), để test luôn
           MdExtractor (extract + chunk) và toàn bộ pipeline embed/upsert.

DATASET GIẢ gồm 6 hồ sơ, cố tình có nhiều case khác nhau:
    - archive-001: 1 file MD vừa đủ dài để bị chia thành nhiều chunk
      (CHUNK_SIZE_CHARS mặc định 1200).
    - archive-002: 1 file MD dài hơn nữa (nhiều chunk hơn).
    - archive-003: KHÔNG có project/mdFileUrls nào -> test case
      "archive không có file MD" (ingestion phải bỏ qua nội dung,
      chỉ upsert metadata, xem IngestionService._sync_one_archive).
    - archive-004: 1 file MD ngắn, chỉ ra đúng 1 chunk.
    - archive-005: "Lý lịch quân nhân" đầy đủ (bản thân, gia đình, bảng
      quá trình công tác, nhận xét, chữ ký) — dữ liệu HƯ CẤU, dùng để
      test câu hỏi đóng (VD "sinh ngày nào", "quê quán ở đâu") xem
      thông tin có nằm trọn trong 1 chunk sau khi chia hay bị cắt mất.
    - archive-006: Quyết định khen thưởng dạng thật (số quyết định, số
      sổ vàng) — dữ liệu HƯ CẤU, cùng mục đích test câu hỏi đóng.

Xem thêm scripts/test_closed_questions.py để chạy test câu hỏi đóng
trên archive-005/archive-006.
"""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

# ── Nội dung file MD giả ─────────────────────────────────────────────
FAKE_MD_FILES = {
    "ho-so-001-doc1": (
        "# Hồ sơ 001 - Tài liệu thử nghiệm\n\n"
        + ("Đây là nội dung mẫu dùng để kiểm thử pipeline RAG phần MD. " * 40)
        + "\n\n## Kết luận\n\nHết tài liệu."
    ),
    "ho-so-002-doc1": (
        "# Hồ sơ 002 - Biên bản họp mẫu\n\n"
        + ("Nội dung biên bản họp giả lập dùng để test extract + chunk markdown. " * 60)
    ),
    "ho-so-004-doc1": "# Hồ sơ 004\n\nFile MD ngắn, chỉ nên ra đúng 1 chunk.",
    # ── Hồ sơ dạng "lý lịch quân nhân" thật (bảng, nhiều mục, chữ ký) ──
    # Dữ liệu HOÀN TOÀN HƯ CẤU (tên, ngày tháng, số quyết định...) — chỉ
    # dùng để test format thật của file MD (nhiều heading, bảng markdown,
    # phần chữ ký/công chứng), không phải dữ liệu cá nhân thật.
    "ho-so-005-doc1": (
        "**BẢN TÓM TẮT LÝ LỊCH**\n"
        "Họ tên khai sinh: **NGUYỄN VĂN TEST01** (Nam)\n"
        "Họ tên khác: Không\n"
        "Cấp bậc, hệ số lương, tháng năm: 28, 6/2024\n"
        "SH: 99 001 234\n"
        "**I. BẢN THÂN**\n"
        "Sinh ngày 05 tháng 09 năm 1999     Dân tộc: Kinh\n"
        "Quê quán: xã Bình An, tỉnh Nam Định.\n"
        "Nơi ở hiện nay: xã Bình An, tỉnh Nam Định.\n"
        "Tôn giáo: Không\n"
        "Ngày nhập ngũ: 15/03/2017\n"
        "Ngày chính thức: 20/05/2019\n"
        "Ngày vào Đảng: 20/05/2018\n"
        "Giáo dục phổ thông: 12/12\n"
        "Chức danh khoa học, học vị cao nhất, chuyên ngành, thời gian: Kỹ sư (Cơ khí chế tạo, 6/2024)\n"
        "Chỉ huy quản lý (sơ, trung, cao cấp): sơ cấp\n"
        "Lý luận chính trị (sơ, trung, cao cấp): sơ cấp\n"
        "Chuyên môn, kỹ thuật (sơ, trung, cao cấp): trung cấp\n"
        "Ngoại ngữ, trình độ, tháng năm: Tiếng Anh, B1, 6/2019\n"
        "Tiếng dân tộc, mức độ nghe, nói, viết: không\n"
        "**Qua trường (tên trường, ngành học, chuyên ngành học, bậc học, thời gian, kết quả, loại hình đào tạo):**\n"
        "- Học viện Kỹ thuật Quân sự, Cơ khí chế tạo, Kỹ sư, 2019-2024, khá, chính quy\n"
        "**Đã đi nước ngoài (tên nước, thời gian, lý do):** Không\n"
        "**Sức khỏe loại:** A1   Nhóm máu: O   Bệnh chính: Không\n"
        "**Khen thưởng:** Chiến sĩ tiên tiến 6/2021\n"
        "**QUÁ TRÌNH CÔNG TÁC**\n"
        "| Từ tháng năm | Đến tháng năm | Chức danh, chức vụ, đơn vị công tác | Cấp bậc | Chức vụ Đảng, Đoàn thể |\n"
        "|---|---|---|---|---|\n"
        "| 3/2017 | 8/2019 | Học viên, Học viện Kỹ thuật Quân sự | B2 (3/2017) | Đoàn viên |\n"
        "| 9/2019 | 6/2024 | Sinh viên hệ kỹ sư, Học viện Kỹ thuật Quân sự | H3 (9/2020) | Đảng viên |\n"
        "| 7/2024 | nay | Trợ lý kỹ thuật, Phòng Kỹ thuật, Lữ đoàn Test 01 | 7/2024 | Đảng viên |\n"
        "**II. TÌNH HÌNH KT - CT CỦA GIA ĐÌNH**\n"
        "Họ tên cha: NGUYỄN VĂN CHA01     Sinh 1970     Nghề nghiệp: Công nhân.\n"
        "Họ tên mẹ: PHẠM THỊ MẸ01  Sinh 1972     Nghề nghiệp: Nông dân.\n"
        "Thành phần gia đình: Trung nông\n"
        "Quê quán: xã Bình An, tỉnh Nam Định\n"
        "Nơi ở hiện nay của gia đình: xã Bình An, tỉnh Nam Định\n"
        "**IV. TÓM TẮT NHẬN XÉT**\n"
        "Lý lịch gia đình hai bên nội ngoại cơ bản rõ ràng, chấp hành tốt chủ trương, "
        "đường lối của Đảng và pháp luật Nhà nước. Bản thân đồng chí Nguyễn Văn Test01 "
        "nhập ngũ tháng 3/2017, được cử đi học hệ kỹ sư tại Học viện Kỹ thuật Quân sự "
        "từ 9/2019 đến 6/2024, chuyên ngành Cơ khí chế tạo, tốt nghiệp loại khá. "
        "Hướng sử dụng làm cán bộ kỹ thuật.\n"
        "Ngày trích: 01/07/2024\n"
        "(Đây là dữ liệu giả lập dùng để test, không phải văn bản thật)\n"
        "TRƯỞNG PHÒNG (giả lập)\n"
        "Trung tá Test Ký Tên 01\n"
        "(Đã ký - dữ liệu test)"
    ),
    "ho-so-006-doc1": (
        "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\n"
        "Độc lập - Tự do - Hạnh phúc\n"
        "***\n"
        "**(DỮ LIỆU GIẢ LẬP - DÙNG ĐỂ TEST, KHÔNG PHẢI VĂN BẢN THẬT)**\n"
        "**TẶNG DANH HIỆU**\n"
        "**CHIẾN SĨ THI ĐUA CƠ SỞ**\n"
        "Thượng úy **LÊ THỊ TEST02**\n"
        "Trợ lý, Phòng Chính trị, Lữ đoàn Test 02\n"
        "Đã có thành tích tiêu biểu xuất sắc trong phong trào thi đua Quyết thắng năm 2023,\n"
        "góp phần vào sự nghiệp xây dựng đơn vị.\n"
        "Số Quyết định: 0000/QĐ-TEST ngày 01 tháng 01 năm 2024\n"
        "Số sổ vàng: 000 (giả lập)\n"
        "Hà Nội, ngày 01 tháng 01 năm 2024\n"
        "**CHỈ HUY ĐƠN VỊ (giả lập)**\n"
        "Đại tá Test Ký Tên 02\n"
        "(Đã ký và đóng dấu - dữ liệu test)"
    ),
}


def _build_archives(base_url: str) -> list[dict]:
    """base_url dạng http://host:port, dùng để build mdFileUrls tuyệt đối
    (download_file() gọi thẳng URL này, không ghép với ARCHIVE_API_BASE_URL)."""
    return [
        {
            "id": "archive-001",
            "title": "Hồ sơ thử nghiệm 001",
            "arcFileCode": "AF-001",
            "shelfCode": "K1",
            "shelfLevelCode": "T1",
            "warehouseName": "Kho A",
            "roomNumber": "101",
            "startDate": "2020-01-01",
            "endDate": "2020-12-31",
            "status": "ACTIVE",
            "description": "Hồ sơ giả — file MD nhiều chunk",
            "totalDoc": 1,
            "language": "vi",
            "maintenance": "NONE",
            "updatedAt": "2026-07-01T00:00:00Z",
            "staffMetadata": [{"fieldName": "Người lập", "value": "Nguyễn Văn A"}],
            "projects": [
                {"name": "Dự án 001", "mdFileUrls": [f"{base_url}/files/ho-so-001-doc1.md"]}
            ],
            "borrowItems": [],
        },
        {
            "id": "archive-002",
            "title": "Hồ sơ thử nghiệm 002",
            "arcFileCode": "AF-002",
            "shelfCode": "K1",
            "shelfLevelCode": "T2",
            "warehouseName": "Kho A",
            "roomNumber": "101",
            "startDate": "2021-01-01",
            "endDate": "2021-12-31",
            "status": "ACTIVE",
            "description": "Hồ sơ giả — file MD dài hơn, nhiều chunk hơn",
            "totalDoc": 1,
            "language": "vi",
            "maintenance": "NONE",
            "updatedAt": "2026-07-02T00:00:00Z",
            "staffMetadata": [{"fieldName": "Người lập", "value": "Trần Thị B"}],
            "projects": [
                {"name": "Dự án 002", "mdFileUrls": [f"{base_url}/files/ho-so-002-doc1.md"]}
            ],
            "borrowItems": [],
        },
        {
            "id": "archive-003",
            "title": "Hồ sơ thử nghiệm 003 - không có file MD",
            "arcFileCode": "AF-003",
            "shelfCode": "K2",
            "shelfLevelCode": "T1",
            "warehouseName": "Kho B",
            "roomNumber": "202",
            "startDate": "2022-01-01",
            "endDate": "2022-12-31",
            "status": "ACTIVE",
            "description": "Test case: archive KHÔNG có mdFileUrls -> ingestion phải bỏ qua nội dung, chỉ upsert metadata",
            "totalDoc": 0,
            "language": "vi",
            "maintenance": "NONE",
            "updatedAt": "2026-07-03T00:00:00Z",
            "staffMetadata": [],
            "projects": [],
            "borrowItems": [],
        },
        {
            "id": "archive-004",
            "title": "Hồ sơ thử nghiệm 004",
            "arcFileCode": "AF-004",
            "shelfCode": "K2",
            "shelfLevelCode": "T2",
            "warehouseName": "Kho B",
            "roomNumber": "202",
            "startDate": "2023-01-01",
            "endDate": "2023-12-31",
            "status": "INACTIVE",
            "description": "Hồ sơ giả — file MD ngắn, đúng 1 chunk",
            "totalDoc": 1,
            "language": "vi",
            "maintenance": "NONE",
            "updatedAt": "2026-07-04T00:00:00Z",
            "staffMetadata": [{"fieldName": "Người lập", "value": "Lê Văn C"}],
            "projects": [
                {"name": "Dự án 004", "mdFileUrls": [f"{base_url}/files/ho-so-004-doc1.md"]}
            ],
            "borrowItems": [],
        },
        {
            "id": "archive-005",
            "title": "Lý lịch quân nhân — Nguyễn Văn Test01 (dữ liệu giả lập)",
            "arcFileCode": "AF-005",
            "shelfCode": "K3",
            "shelfLevelCode": "T1",
            "warehouseName": "Kho C",
            "roomNumber": "301",
            "startDate": "2024-07-01",
            "endDate": None,
            "status": "ACTIVE",
            "description": "Hồ sơ giả dạng 'lý lịch quân nhân' đầy đủ (bản thân, gia đình, quá trình công tác, nhận xét) — dùng để test câu hỏi đóng",
            "totalDoc": 1,
            "language": "vi",
            "maintenance": "NONE",
            "updatedAt": "2026-07-05T00:00:00Z",
            "staffMetadata": [{"fieldName": "Người lập", "value": "Test Data"}],
            "projects": [
                {"name": "Lý lịch Test01", "mdFileUrls": [f"{base_url}/files/ho-so-005-doc1.md"]}
            ],
            "borrowItems": [],
        },
        {
            "id": "archive-006",
            "title": "Quyết định khen thưởng — Lê Thị Test02 (dữ liệu giả lập)",
            "arcFileCode": "AF-006",
            "shelfCode": "K3",
            "shelfLevelCode": "T2",
            "warehouseName": "Kho C",
            "roomNumber": "301",
            "startDate": "2024-01-01",
            "endDate": None,
            "status": "ACTIVE",
            "description": "Hồ sơ giả dạng quyết định khen thưởng, có số quyết định/số sổ vàng — dùng để test câu hỏi đóng",
            "totalDoc": 1,
            "language": "vi",
            "maintenance": "NONE",
            "updatedAt": "2026-07-06T00:00:00Z",
            "staffMetadata": [{"fieldName": "Người lập", "value": "Test Data"}],
            "projects": [
                {"name": "Khen thưởng Test02", "mdFileUrls": [f"{base_url}/files/ho-so-006-doc1.md"]}
            ],
            "borrowItems": [],
        },
    ]


class FakeArchiveApiHandler(BaseHTTPRequestHandler):
    archive_path = "/api/public/archives"  # khớp mặc định ARCHIVE_API_PATH trong .env.example; đổi bằng --archive-path nếu .env bạn dùng giá trị khác

    def log_message(self, fmt, *args):  # noqa: D401 - im lặng log mặc định, tự log gọn hơn
        print(f"[fake-archive-api] {self.address_string()} - {fmt % args}")

    def _send_json(self, payload: dict, status: int = 200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, text: str, content_type: str, status: int = 200):
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 - tên method do BaseHTTPRequestHandler quy định
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        base_url = f"http://{self.headers.get('Host', 'localhost')}"

        if parsed.path == self.archive_path:
            self._handle_archive_list(qs, base_url)
        elif parsed.path.startswith("/files/") and parsed.path.endswith(".md"):
            self._handle_file(parsed.path)
        else:
            self._send_json({"error": "not_found", "path": parsed.path}, status=404)

    def _handle_archive_list(self, qs: dict, base_url: str):
        page = int(qs.get("page", ["0"])[0])
        size = int(qs.get("size", ["100"])[0])

        archives = _build_archives(base_url)
        total = len(archives)
        total_pages = max(1, (total + size - 1) // size)

        start = page * size
        end = start + size
        content = archives[start:end]

        self._send_json(
            {
                "content": content,
                "page": {
                    "number": page,
                    "size": size,
                    "totalElements": total,
                    "totalPages": total_pages,
                },
            }
        )

    def _handle_file(self, path: str):
        key = path[len("/files/"):-len(".md")]
        text = FAKE_MD_FILES.get(key)
        if text is None:
            self._send_json({"error": "file_not_found", "key": key}, status=404)
            return
        self._send_text(text, content_type="text/markdown")


def main():
    parser = argparse.ArgumentParser(description="Fake Archive API server (test phần MD)")
    parser.add_argument("--host", default="0.0.0.0", help="Host lắng nghe (mặc định 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port lắng nghe (mặc định 8000)")
    parser.add_argument(
        "--archive-path",
        default="/api/public/archives",
        help="Path của endpoint danh sách hồ sơ, phải khớp ARCHIVE_API_PATH trong .env (mặc định /api/public/archives)",
    )
    args = parser.parse_args()

    FakeArchiveApiHandler.archive_path = args.archive_path

    server = ThreadingHTTPServer((args.host, args.port), FakeArchiveApiHandler)
    print(f"[fake-archive-api] Đang chạy tại http://{args.host}:{args.port}")
    print(f"[fake-archive-api] Endpoint danh sách hồ sơ: {args.archive_path}")
    print(f"[fake-archive-api] Endpoint file MD: /files/<key>.md")
    print("[fake-archive-api] Nhớ set trong .env:")
    print(f"    ARCHIVE_API_BASE_URL=http://localhost:{args.port}")
    print(f"    ARCHIVE_API_PATH={args.archive_path}")
    print("[fake-archive-api] Ctrl+C để dừng.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[fake-archive-api] Đang dừng server...")
        server.shutdown()


if __name__ == "__main__":
    main()