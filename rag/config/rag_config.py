"""
config/rag_config.py — Cấu hình RIÊNG cho module rag/, đọc từ cùng file
.env ở gốc project.

Module rag/ đọc cấu hình ĐỘC LẬP với src/config/configs.py (tầng MCP),
để rag/ có thể được copy/tái sử dụng ở project khác mà không cần kéo
theo toàn bộ code MCP. Nếu muốn dùng chung 1 file .env (khuyến nghị),
chỉ cần đảm bảo các biến bên dưới có mặt trong .env gốc.
"""
import os

from dotenv import load_dotenv

load_dotenv()


class RagConfig:
    # --- Archive API (nguồn dữ liệu gốc) ---
    # QUAN TRỌNG: rag/ và mcp/ (mcp/src/config/configs.py) đều đọc biến
    # ARCHIVE_API_BASE_URL — nếu chỉ set 1 biến này, MỌI thứ (cả tool
    # search_archives live lẫn MD ingestion) sẽ trỏ chung 1 nơi.
    #
    # Để test riêng phần MD ingestion bằng fake API mà KHÔNG ảnh hưởng
    # tới các tool live (search_archives, get_archive_detail...) vẫn
    # đang cần gọi backend thật, ưu tiên đọc RAG_ARCHIVE_API_BASE_URL
    # (chỉ dành riêng cho rag/) nếu có; nếu không set thì fallback về
    # ARCHIVE_API_BASE_URL như cũ (giữ nguyên hành vi production khi
    # dùng chung 1 file .env, không phải sửa gì nếu không cần tách).
    #
    # VD trong .env khi test MD:
    #   ARCHIVE_API_BASE_URL=http://192.168.1.32:4010      # mcp/ dùng - backend thật
    #   RAG_ARCHIVE_API_BASE_URL=http://localhost:8000     # rag/ dùng - fake API test MD
    ARCHIVE_API_BASE_URL = (
        os.getenv("RAG_ARCHIVE_API_BASE_URL")
        or os.getenv("ARCHIVE_API_BASE_URL", "http://192.168.1.32:4010")
    )
    ARCHIVE_API_PATH = os.getenv("ARCHIVE_API_PATH", "/api/public/archive")
    ARCHIVE_API_PAGE_SIZE = int(os.getenv("ARCHIVE_API_PAGE_SIZE", "100"))
    HTTP_TIMEOUT_SECONDS = int(os.getenv("HTTP_TIMEOUT_SECONDS", "60"))

    # --- Chatbot session token (X-Chatbot-Token) ---
    CHATBOT_TOKEN_PATH = os.getenv("CHATBOT_TOKEN_PATH", "/api/v1/chatbot/session-token")
    CHATBOT_CLIENT_ID = os.getenv("CHATBOT_CLIENT_ID") or None
    CHATBOT_CLIENT_SECRET = os.getenv("CHATBOT_CLIENT_SECRET") or None
    # Refresh token sớm hơn hạn thật bao nhiêu giây, tránh race condition
    # ngay sát lúc hết hạn (request đang bay thì token hết hạn giữa chừng)
    TOKEN_REFRESH_BUFFER_SECONDS = int(os.getenv("TOKEN_REFRESH_BUFFER_SECONDS", "60"))

    # --- Qdrant ---
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
    COLLECTION_ARCHIVES = os.getenv("RAG_COLLECTION_ARCHIVES", "archives")
    COLLECTION_CHUNKS = os.getenv("RAG_COLLECTION_CHUNKS", "document_chunks")

    # --- Embedding models (fastembed, chạy local) ---
    DENSE_MODEL_NAME = os.getenv("DENSE_MODEL_NAME", "multilingual-e5-large")
    SPARSE_MODEL_NAME = os.getenv("SPARSE_MODEL_NAME", "Qdrant/bm25")
    DENSE_VECTOR_SIZE = int(os.getenv("DENSE_VECTOR_SIZE", "1024"))  # bge-m3 output dim


    # --- Chunking ---
    CHUNK_SIZE_CHARS = int(os.getenv("CHUNK_SIZE_CHARS", "1200"))
    CHUNK_OVERLAP_CHARS = int(os.getenv("CHUNK_OVERLAP_CHARS", "200"))


rag_config = RagConfig()