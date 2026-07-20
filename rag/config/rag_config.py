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

    # LƯU Ý: rag/ (pipeline ingest one-off) KHÔNG dùng X-Chatbot-Token —
    # HttpArchiveApiClient gọi thẳng không auth. Cấu hình token
    # (CHATBOT_TOKEN_PATH/CLIENT_ID/CLIENT_SECRET) chỉ dùng cho tool
    # search_archives (live query), xem mcp/src/config/configs.py và
    # mcp/src/archive_api/token_manager.py.

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

    # --- Retrieval ---
    # Ngưỡng lọc kết quả theo tỉ lệ so với điểm cao nhất (0 < ratio <= 1):
    # 1 kết quả bị loại nếu score < top_score * SCORE_GAP_RATIO. 
    SCORE_GAP_RATIO = float(os.getenv("SCORE_GAP_RATIO", "0.5"))


rag_config = RagConfig()