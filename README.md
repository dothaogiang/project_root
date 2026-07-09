# Profile Lookup — MCP Server + RAG Pipeline

Hệ thống tra cứu hồ sơ lưu trữ (archive) qua chatbot, dùng chuẩn **MCP
(Model Context Protocol)** để expose 2 tool cho LLM/chatbot gọi:
`search_profile` (tìm hồ sơ theo từ khóa) và `get_profile_detail` (hỏi
sâu nội dung PDF bên trong 1 hồ sơ cụ thể).

Dữ liệu hồ sơ (metadata + file PDF) được lấy về từ Public Archive API
nội bộ, OCR/trích xuất text, embed rồi lưu vào Qdrant để phục vụ tìm
kiếm ngữ nghĩa (semantic) kết hợp từ khóa (hybrid search).

## Dự án làm gì

Hệ thống expose **6 tool MCP**, chia làm 2 nhóm khác bản chất:

**Nhóm A — Semantic search qua Qdrant** (dữ liệu đã ingest sẵn, xem
`rag/`):

- `search_profile` — tìm hồ sơ bằng hybrid search (semantic + từ khóa)
- `get_profile_detail` — hỏi sâu nội dung PDF (đã OCR) bên trong 1 hồ sơ

**Nhóm B — Live query trực tiếp vào Public Archive API** (không qua
Qdrant, có xác thực `X-Chatbot-Token`, xem `mcp/src/archive_api/`):

- `search_archives` — tìm hồ sơ theo field lọc chính xác (status, kho, ngôn ngữ, khoảng ngày...)
- `get_archive_detail` — lấy toàn bộ chi tiết 1 hồ sơ theo UUID (metadata, project, lịch sử mượn)
- `get_staff_archive_metadata` — lấy cấu trúc/schema của hồ sơ cán bộ
- `get_file_proxy` — lấy nội dung file gốc (PDF, ảnh...) đính kèm hồ sơ

Quy tắc chọn tool cho chatbot: chưa biết `archive_id` → luôn
`search_archives` (hoặc `search_profile` nếu muốn semantic) trước; đã
biết `archive_id` mà cần chi tiết đầy đủ (kể cả lịch sử mượn) →
`get_archive_detail`; cần hỏi sâu nội dung PDF → `get_profile_detail`;
cần schema hồ sơ cán bộ → `get_staff_archive_metadata`; cần xem/tải
file → `get_file_proxy`.

## Cấu trúc dự án

Dự án chia làm **2 folder độc lập, ngang hàng nhau**:

```
project_root/
├── mcp/                          # Chỉ lo giao thức MCP — nhận request, trả response
│   ├── src/
│   │   ├── server.py             # Entry point, chạy MCP server (streamable-http)
│   │   ├── feature_manager.py    # Nơi định nghĩa method cho từng tool (khớp tên với tools.yaml)
│   │   ├── archive_api/          # Client LIVE query trực tiếp Public Archive API (nhóm B)
│   │   │   ├── client.py         #   search_archives, get_archive_detail, get_staff_archive_metadata, get_file_proxy
│   │   │   └── token_manager.py  #   Quản lý X-Chatbot-Token (cache + auto refresh khi 401)
│   │   ├── logger.py
│   │   ├── config/configs.py     # Cấu hình riêng của MCP (port, resources dir, archive API paths, token...)
│   │   └── tools/
│   │       ├── manager.py
│   │       └── registry.py       # Đọc tools.yaml, tự đăng ký tool theo tên hàm khớp
│   ├── Resources/tools.yaml      # Khai báo 6 tool + input schema
│   └── requirements.txt          # Dependency riêng của mcp/
│
├── rag/                          # Chỉ lo dữ liệu — ingest vào Qdrant & truy vấn ra
│   ├── domain/entities.py        # Model dữ liệu thuần (ArchiveRecord, DocumentChunk...)
│   ├── ports/interfaces.py       # Interface — application phụ thuộc vào đây, không phụ thuộc infra cụ thể
│   ├── application/
│   │   ├── ingestion_service.py  # Use case: API -> extract/OCR -> chunk -> embed -> Qdrant
│   │   └── retrieval_service.py  # Use case: search_profiles / search_chunks_in_archive (+ lọc score-gap)
│   ├── infrastructure/
│   │   ├── archive_api_client.py # Gọi Public Archive API thật (httpx)
│   │   ├── pdf_extractor.py      # PyMuPDF + pytesseract (native/OCR + chunk)
│   │   ├── embedding_provider.py # fastembed (dense multilingual-e5-large + sparse bm25)
│   │   ├── vector_store.py       # Qdrant (2 collection: archives, document_chunks)
│   │   └── sync_state_repo.py    # SQLite checkpoint (incremental sync)
│   ├── config/rag_config.py      # Cấu hình riêng của rag/ (đọc từ .env)
│   ├── jobs/sync_job.py          # Entry point chạy đồng bộ: python -m rag.jobs.sync_job
│   ├── logger.py                 # Logger riêng, rag/ không phụ thuộc mcp/
│   ├── retrieval_factory.py      # Nơi mcp/ (hoặc chatbot khác) gọi vào để lấy RetrievalService
│   └── requirements.txt          # Dependency riêng của rag/
│
├── Dockerfile                    # Build 1 image chứa cả 2 folder (mcp_server chạy chung tiến trình, import rag trực tiếp)
├── docker-compose.yaml           # 3 service: qdrant, mcp_server, sync_cron
├── .env.example                  # Copy thành .env rồi điền giá trị thật
└── .gitignore
```

**Vì sao tách 2 folder độc lập:** `mcp/` không biết gì về Qdrant/OCR/
embedding — nó chỉ biết gọi vào `rag.retrieval_factory.get_retrieval_service()`.
`rag/` không biết gì về MCP/tools.yaml — nó chỉ lo ingest và trả dữ
liệu. Nhờ vậy có thể đổi hạ tầng (Qdrant → DB khác, model embedding
khác...) chỉ bằng cách sửa file trong `rag/infrastructure/`, không đụng
gì tới `mcp/`; và `rag/` có thể tái sử dụng cho 1 chatbot/service khác
sau này mà không cần kéo theo code MCP.

## Cách 1 — Chạy bằng Docker Compose (khuyến nghị)

```bash
cd project_root
cp .env.example .env
# mở .env, sửa ARCHIVE_API_BASE_URL cho đúng địa chỉ Public Archive API thật

docker-compose up --build
```

Lệnh này khởi động 3 service:

| Service      | Vai trò                                                              |
| ------------ | -------------------------------------------------------------------- |
| `qdrant`     | Vector DB, port `6333`                                               |
| `mcp_server` | Server MCP, port `8090`, chạy `python mcp/src/server.py`             |
| `sync_cron`  | Vòng lặp đồng bộ dữ liệu mỗi giờ, chạy `python -m rag.jobs.sync_job` |

Ingest dữ liệu ngay lần đầu (không cần đợi cron chạy theo giờ):

```bash
docker-compose run --rm sync_cron python -m rag.jobs.sync_job
```

Theo dõi log:

```bash
docker-compose logs -f sync_cron
docker-compose logs -f mcp_server
```

Dừng toàn bộ:

```bash
docker-compose down
```

## Cách 2 — Chạy local bằng venv (tiện debug từng bước)

```bash
cd project_root
python -m venv venv
source venv/bin/activate          # Windows PowerShell: venv\Scripts\activate

pip install --upgrade pip
pip install -r mcp/requirements.txt -r rag/requirements.txt
# venv chạy CHUNG 1 tiến trình mcp_server nên bắt buộc cài đủ cả 2 file

cp .env.example .env
# sửa QDRANT_URL=http://localhost:6333
# sửa ARCHIVE_API_BASE_URL cho đúng địa chỉ thật
```

**1. Chạy Qdrant** (container riêng, hoặc dùng service `qdrant` trong `docker-compose.yaml`):

```bash
docker run -p 6333:6333 qdrant/qdrant:latest
```

**2. Ingest dữ liệu** (chạy 1 lần để có dữ liệu test, log sẽ hiện tiến độ từng archive/file):

```bash
python -m rag.jobs.sync_job
```

**3. Chạy MCP server:**

```bash
python mcp/src/server.py
```

Server log dòng `Server sẵn sàng, bắt đầu lắng nghe...` và lắng nghe
tại `http://localhost:8090/mcp` (đường dẫn `/mcp` mặc định của FastMCP
với `transport="streamable-http"`).

> Mỗi lần sửa code trong `mcp/` hoặc `rag/`, phải `Ctrl+C` dừng
> `server.py`/`sync_job` đang chạy rồi chạy lại lệnh — Python không tự
> hot-reload.

## Test bằng Postman

**Cách nhanh nhất (Postman ≥ v11 có sẵn MCP request):**

1. New → **MCP Request** → URL `http://localhost:8090/mcp`, transport **Streamable HTTP** → Connect.
2. Postman tự liệt kê 2 tool `search_profile`, `get_profile_detail` kèm form nhập tham số.
3. Chọn tool, điền tham số, Run, xem kết quả.

**Cách thủ công (JSON-RPC qua POST, header `Content-Type: application/json` + `Accept: application/json, text/event-stream`):**

1. `initialize` → lấy `Mcp-Session-Id` ở response header, dùng cho mọi request sau.
2. `notifications/initialized`.
3. `tools/list` → xác nhận có đúng 2 tool.
4. `tools/call` với `name: "search_profile"`, `arguments: {"keyword": "..."}`.
5. `tools/call` với `name: "get_profile_detail"`, `arguments: {"archive_id": "...", "question": "..."}`.

## Xử lý sự cố thường gặp

| Hiện tượng                                                               | Nguyên nhân                                              | Cách kiểm tra                                                                                           |
| ------------------------------------------------------------------------ | -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `pip` báo dependency conflict (obspy, torchvision...)                    | Đang cài vào env chung với project khác                  | Tạo venv mới sạch, không dùng chung env                                                                 |
| `httpx.ConnectError: All connection attempts failed` khi chạy `sync_job` | Không kết nối được `ARCHIVE_API_BASE_URL` (sai mạng/VPN) | `Invoke-WebRequest` thử gọi thẳng URL đó                                                                |
| `Model ... is not supported in TextEmbedding`                            | Bản `fastembed` đang cài không có model đó               | `TextEmbedding.list_supported_models()` để xem model nào khả dụng, đổi `DENSE_MODEL_NAME` trong `.env`  |
| `tesseract is not installed or it's not in your PATH`                    | Máy chưa cài Tesseract OCR (chỉ cần khi PDF là bản scan) | Cài Tesseract (+ gói ngôn ngữ `vie`) và thêm vào PATH, hoặc dùng Docker (đã cài sẵn trong `Dockerfile`) |
| `search_profile` trả về quá nhiều hồ sơ không liên quan                  | Bản chất vector search luôn trả top-K dù không liên quan | Đã có lọc score-gap trong `retrieval_service.py`; điều chỉnh `score_gap_ratio` nếu cần lọc gắt/lỏng hơn |
| `ModuleNotFoundError: rag` khi chạy `mcp/src/server.py`                  | `sys.path` chưa trỏ đúng tới `project_root`              | Kiểm tra `PROJECT_ROOT = Path(__file__).resolve().parents[2]` trong `server.py`                         |

## Biến môi trường (`.env`)

Xem đầy đủ trong `.env.example` — dùng chung 1 file `.env` ở project
root cho cả `mcp/` (`SERVER_NAME`, `PORT_SERVER`...) và `rag/`
(`ARCHIVE_API_BASE_URL`, `QDRANT_URL`, `DENSE_MODEL_NAME`, `OCR_*`,
`CHUNK_*`...).
