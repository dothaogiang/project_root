"""
config/rag_config.py — Cấu hình RIÊNG cho module rag/, đọc từ cùng file
.env ở gốc project.
"""
import os

from dotenv import load_dotenv

load_dotenv()


class RagConfig:
    ARCHIVE_API_BASE_URL = (
        os.getenv("RAG_ARCHIVE_API_BASE_URL")
        or os.getenv("ARCHIVE_API_BASE_URL", "http://192.168.1.100:4010")
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