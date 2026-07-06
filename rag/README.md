# `rag/` — Module RAG độc lập

Module này chịu trách nhiệm **toàn bộ vòng đời dữ liệu RAG**: lấy hồ sơ
từ Public Archive API → extract/OCR PDF → chunk → embed → lưu vào
Qdrant, và cung cấp API truy vấn (retrieval) cho tầng khác dùng (MCP
tools hiện tại, hoặc sau này 1 chatbot/service khác gọi thẳng).

`rag/` **không biết gì về MCP** (không import `mcp`, không đụng
`tools.yaml`). Nó chỉ quan tâm: dữ liệu vào (ingestion) và dữ liệu ra
(retrieval). MCP server (`src/`) là MỘT trong nhiều "khách hàng" có thể
dùng module này.

## Cấu trúc thư mục (Clean Architecture)

```
rag/
├── domain/                  # Tầng lõi — model dữ liệu thuần, không phụ thuộc gì
│   └── entities.py          #   ArchiveRecord, DocumentChunk, RetrievedChunk...
│
├── ports/                   # "Hợp đồng" (interface) — application phụ thuộc vào đây,
│   └── interfaces.py        # KHÔNG phụ thuộc trực tiếp vào Qdrant/httpx/fastembed
│
├── application/             # Use case — logic nghiệp vụ, orchestration
│   ├── ingestion_service.py #   Use case "đồng bộ": API -> extract -> embed -> Qdrant
│   └── retrieval_service.py #   Use case "truy vấn": search_profiles / search_chunks
│
├── infrastructure/          # Implementation cụ thể của từng port
│   ├── archive_api_client.py#   Gọi Public Archive API thật (httpx)
│   ├── pdf_extractor.py     #   PyMuPDF + pytesseract (native/OCR + chunk)
│   ├── embedding_provider.py#   fastembed (dense bge-m3 + sparse bm25)
│   ├── vector_store.py      #   Qdrant (2 collection: archives, document_chunks)
│   └── sync_state_repo.py   #   SQLite checkpoint (incremental sync)
│
├── config/
│   └── rag_config.py        # Đọc .env riêng cho module này
│
├── jobs/
│   └── sync_job.py          # Entry point CHẠY đồng bộ (composition root)
│
└── retrieval_factory.py     # Composition root cho phía TRUY VẤN (MCP dùng cái này)
```

### Vì sao chia như vậy?

| Tầng | Phụ thuộc vào | Lý do |
|---|---|---|
| `domain` | Không gì cả | Model dữ liệu sống lâu nhất, không nên bị kéo theo khi đổi thư viện |
| `ports` | `domain` | Định nghĩa "cái application cần", để tách khỏi "cái infrastructure có" |
| `application` | `ports`, `domain` | Chứa logic nghiệp vụ thật, test được mà không cần Qdrant/mạng thật |
| `infrastructure` | `ports`, thư viện ngoài | Nơi DUY NHẤT biết Qdrant/httpx/fastembed cụ thể ra sao |
| `jobs` / `retrieval_factory` | tất cả | "Lắp ráp" (dependency injection) — chỗ duy nhất new() các class cụ thể |

**Lợi ích thực tế:**
- Đổi Qdrant → Milvus/pgvector: chỉ viết `infrastructure/vector_store.py`
  mới, `application/` không đổi 1 dòng.
- Đổi fastembed → gọi API embedding ngoài (OpenAI, Cohere...): chỉ viết
  `infrastructure/embedding_provider.py` mới.
- Test `IngestionService`/`RetrievalService` bằng cách truyền vào các
  class giả (fake) implement đúng `ports/interfaces.py`, không cần
  Qdrant hay Archive API thật chạy nền.
- Chatbot/service khác (không qua MCP) vẫn dùng lại được toàn bộ
  `rag/` bằng cách import `rag.retrieval_factory.get_retrieval_service()`.

## 2 use case chính

### 1. Ingestion (đồng bộ dữ liệu) — chạy định kỳ

```bash
# Từ project root:
python -m rag.jobs.sync_job
```

Luồng: `HttpArchiveApiClient.fetch_page()` → với mỗi archive:
embed metadata → `QdrantVectorStore.upsert_archive()` (collection
`archives`); với mỗi file PDF: `download_file()` → so hash MD5 với
`SqliteSyncStateRepo` (bỏ qua nếu không đổi) → `PyMuPdfExtractor` tách
text/OCR + chunk → `FastEmbedProvider.embed_batch()` →
`QdrantVectorStore.upsert_chunks()` (collection `document_chunks`).

Tự resume theo checkpoint page nếu bị crash giữa chừng (SQLite
`sync_state.db`). Đặt trong crontab/APScheduler/Docker Compose
`sync_cron` để chạy mỗi giờ (hoặc tần suất phù hợp thực tế dữ liệu).

### 2. Retrieval (truy vấn) — chatbot/MCP gọi theo mỗi request

```python
from rag.retrieval_factory import get_retrieval_service

service = get_retrieval_service()

# Tầng 1: tìm ĐÚNG hồ sơ nào (metadata-level)
profiles = service.search_profiles(keyword="Trần Xuân Sang")

# Tầng 2: hỏi sâu nội dung PDF bên trong 1 hồ sơ (chunk-level — RAG thật sự)
chunks = service.search_chunks_in_archive(
    archive_id=profiles[0].archive_id,
    question="tốt nghiệp năm nào",
)
```

Đây chính là nơi MCP `feature_manager.py` (`search_profile`,
`get_profile_detail`) gọi vào — xem `src/feature_manager.py` đã được
cập nhật để chỉ còn "dịch" kết quả `RetrievalService` sang format tool.

Sau này nếu có một chatbot/service KHÁC (không qua MCP) cần dữ liệu
này, chỉ cần import đúng 2 dòng trên — không cần biết gì về Qdrant hay
embedding model bên dưới.

## Biến môi trường liên quan (`.env`)

Xem `rag/config/rag_config.py` — dùng chung file `.env` ở project root
với các biến: `ARCHIVE_API_BASE_URL`, `ARCHIVE_API_PATH`,
`ARCHIVE_API_PAGE_SIZE`, `QDRANT_URL`, `QDRANT_API_KEY`,
`DENSE_MODEL_NAME`, `SPARSE_MODEL_NAME`, `DENSE_VECTOR_SIZE`,
`OCR_LANG`, `OCR_MIN_CHARS_PER_PAGE`, `OCR_DPI`, `OCR_CONCURRENCY`,
`CHUNK_SIZE_CHARS`, `CHUNK_OVERLAP_CHARS`, `SYNC_DB_PATH`.

## Dọn dẹp sau khi merge vào project

Các file cũ dưới đây đã được thay thế hoàn toàn bởi `rag/` và có thể
**xóa** để tránh trùng logic (an toàn xóa vì không còn ai import tới):

- `src/common_utils/qdrant_utils.py` → `rag/infrastructure/vector_store.py`
- `src/common_utils/embedding_utils.py` → `rag/infrastructure/embedding_provider.py`
- `src/common_utils/pdf_utils.py` → `rag/infrastructure/pdf_extractor.py`
- `src/common_utils/sync_state.py` → `rag/infrastructure/sync_state_repo.py`
- `src/common_utils/constants.py` → gộp vào `pdf_extractor.py`/`vector_store.py`
- `src/ingestion/sync_job.py` → `rag/jobs/sync_job.py`

`src/config/configs.py` vẫn giữ lại (dùng cho cấu hình riêng của MCP
server như `PORT_SERVER`, `RESOURCES_DIR`) — chỉ phần cấu hình liên
quan RAG đã trùng với `rag/config/rag_config.py`, có thể rút gọn dần.

## Cần cập nhật thêm ở `docker-compose.yaml`

Service `sync_cron` đang chạy `python src/ingestion/sync_job.py` — đổi
thành:

```yaml
entrypoint: >
  sh -c "while true; do python -m rag.jobs.sync_job; echo 'Chờ 3600s...'; sleep 3600; done"
```
