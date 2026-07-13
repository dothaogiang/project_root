# `rag/` — Module RAG độc lập

Module này chịu trách nhiệm **toàn bộ vòng đời dữ liệu RAG**: lấy hồ sơ
từ Public Archive API → extract file Markdown → chunk → embed → lưu
vào Qdrant, và cung cấp API truy vấn (retrieval) cho tầng khác dùng
(MCP tools hiện tại, hoặc sau này 1 chatbot/service khác gọi thẳng).

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
│   ├── ingestion_service.py #   Use case "đồng bộ" (chạy 1 lần): API -> extract -> embed -> Qdrant
│   └── retrieval_service.py #   Use case "truy vấn": search_profiles / search_chunks
│
├── infrastructure/          # Implementation cụ thể của từng port
│   ├── archive_api_client.py#   Gọi Public Archive API thật (httpx)
│   ├── md_extractor.py      #   Trích text từ file Markdown + chunk
│   ├── embedding_provider.py#   fastembed (dense multilingual-e5-large + sparse bm25)
│   └── vector_store.py      #   Qdrant (2 collection: archives, document_chunks)
│
├── config/
│   └── rag_config.py        # Đọc .env riêng cho module này
│
├── jobs/
│   └── sync_job.py          # Entry point CHẠY đồng bộ 1 lần (composition root)
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

### 1. Ingestion (nhúng dữ liệu vào Qdrant) — chạy 1 lần, thủ công

```bash
# Từ project root:
python -m rag.jobs.sync_job
```

Luồng: `HttpArchiveApiClient.fetch_page()` → với mỗi archive: embed
metadata → `QdrantVectorStore.upsert_archive()` (collection
`archives`); với mỗi file MD: `download_file()` → `MdExtractor` trích
text + chunk → `FastEmbedProvider.embed_batch()` →
`QdrantVectorStore.upsert_chunks()` (collection `document_chunks`).

Đây là job **one-off**: không có cron/APScheduler/checkpoint, mỗi lần
chạy sẽ duyệt và nhúng lại toàn bộ dữ liệu lấy được từ Archive API tại
thời điểm chạy. Chạy lại nhiều lần vẫn an toàn — point ID trong Qdrant
được tính deterministic từ `(archive_id, file_url, chunk_index)`, nên
upsert chỉ ghi đè dữ liệu cũ, không tạo bản trùng.

Muốn cập nhật dữ liệu về sau (có archive mới, nội dung đổi...), chạy
lại đúng lệnh trên theo tay — không có cơ chế tự động theo lịch.

### 2. Retrieval (truy vấn) — chatbot/MCP gọi theo mỗi request

```python
from rag.retrieval_factory import get_retrieval_service

service = get_retrieval_service()

# Tầng 1: tìm ĐÚNG hồ sơ nào (metadata-level)
profiles = service.search_profiles(keyword="Trần Xuân Sang")

# Tầng 2: hỏi sâu nội dung file MD bên trong 1 hồ sơ (chunk-level — RAG thật sự)
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
`CHUNK_SIZE_CHARS`, `CHUNK_OVERLAP_CHARS`, `MD_FILE_URL_FIELD`.