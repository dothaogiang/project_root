"""
Cấu hình của riêng MCP server (folder mcp/).

Gồm 2 nhóm:
  1. Cấu hình khởi động server (SERVER_NAME, PORT_SERVER, RESOURCES_DIR)
  2. Cấu hình cho archive_api/ — client gọi TRỰC TIẾP (live) Public
     Archive API để phục vụ 4 tool: search_archives, get_archive_detail,
     get_staff_archive_metadata, get_file_proxy.

KHÔNG chứa cấu hình pipeline RAG (Qdrant, embedding, OCR, chunking,
sync state) — phần đó nằm trong rag/config/rag_config.py, vì đó là
concern riêng của ingestion/semantic-retrieval, không phải của mcp/.
"""
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Config:
    # --- MCP server ---
    SERVER_NAME = os.getenv("SERVER_NAME", "profile_lookup")
    URL_HOST_SERVER = os.getenv("URL_HOST_SERVER", "0.0.0.0")
    PORT_SERVER = int(os.getenv("PORT_SERVER", "8090"))

    # --- Resources (nơi chứa tools.yaml) ---
    RESOURCES_DIR = os.path.join(BASE_DIR, "..", "Resources")

    # --- Debug: có trả traceback đầy đủ trong response tool lỗi không.
    # Bật khi test (Postman thấy ngay nguyên nhân lỗi), TẮT khi lên
    # production (tránh lộ đường dẫn/nội bộ hệ thống cho chatbot/người
    # dùng cuối). Set DEBUG_TOOL_ERRORS=false trong .env production. ---
    DEBUG_TOOL_ERRORS = os.getenv("DEBUG_TOOL_ERRORS", "true").lower() == "true"

    # --- Public Archive API (live query, dùng bởi archive_api/client.py) ---
    ARCHIVE_API_BASE_URL = os.getenv("ARCHIVE_API_BASE_URL", "http://192.168.1.46:4000")
    ARCHIVE_SEARCH_PATH = os.getenv("ARCHIVE_SEARCH_PATH", "/api/public/archives")
    ARCHIVE_DETAIL_PATH = os.getenv("ARCHIVE_DETAIL_PATH", "/api/public/archives/{id}")
    STAFF_ARCHIVE_PATH = os.getenv("STAFF_ARCHIVE_PATH", "/api/public/staff-archive")
    FILE_PROXY_PATH = os.getenv("FILE_PROXY_PATH", "/api/public/files/proxy")
    HTTP_TIMEOUT_SECONDS = int(os.getenv("HTTP_TIMEOUT_SECONDS", "60"))

    AUTH_ENABLED = os.getenv("AUTH_ENABLED", "false").lower() == "true"
    CHATBOT_TOKEN_PATH = os.getenv("CHATBOT_TOKEN_PATH", "/api/v1/chatbot/session-token")
    CHATBOT_CLIENT_ID = os.getenv("CHATBOT_CLIENT_ID", "")
    CHATBOT_CLIENT_SECRET = os.getenv("CHATBOT_CLIENT_SECRET", "")


config_object = Config()
