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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class RagConfig:
    # --- Archive API (nguồn dữ liệu gốc) ---
    ARCHIVE_API_BASE_URL = os.getenv("ARCHIVE_API_BASE_URL", "http://192.168.1.46:4000")
    ARCHIVE_API_PATH = os.getenv("ARCHIVE_API_PATH", "/api/public/archive")
    ARCHIVE_API_PAGE_SIZE = int(os.getenv("ARCHIVE_API_PAGE_SIZE", "100"))
    HTTP_TIMEOUT_SECONDS = int(os.getenv("HTTP_TIMEOUT_SECONDS", "60"))

    # --- Staff Archive Metadata API ---
    STAFF_ARCHIVE_PATH = os.getenv("STAFF_ARCHIVE_PATH", "/api/public/staff-archive")

    # --- File Proxy API ---
    FILE_PROXY_PATH = os.getenv("FILE_PROXY_PATH", "/api/public/files/proxy")

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

    # --- OCR ---
    OCR_LANG = os.getenv("OCR_LANG", "vie")
    OCR_MIN_CHARS_PER_PAGE = int(os.getenv("OCR_MIN_CHARS_PER_PAGE", "50"))
    OCR_DPI = int(os.getenv("OCR_DPI", "200"))
    OCR_CONCURRENCY = int(os.getenv("OCR_CONCURRENCY", "4"))
    # Đường dẫn tới binary tesseract.exe — chỉ cần set nếu KHÔNG có tesseract
    # trong PATH hệ thống (thường gặp trên Windows nếu bỏ qua bước add PATH
    # lúc cài đặt). Để trống nếu tesseract đã chạy được từ terminal.
    TESSERACT_CMD = os.getenv("TESSERACT_CMD", "")

    # --- Chunking ---
    CHUNK_SIZE_CHARS = int(os.getenv("CHUNK_SIZE_CHARS", "1200"))
    CHUNK_OVERLAP_CHARS = int(os.getenv("CHUNK_OVERLAP_CHARS", "200"))

    # --- Sync state (SQLite, để cron chạy incremental) ---
    SYNC_DB_PATH = os.getenv("SYNC_DB_PATH", os.path.join(BASE_DIR, "sync_state.db"))


rag_config = RagConfig()