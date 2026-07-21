"""
Cấu hình của riêng MCP server (folder mcp/).

Gồm 2 nhóm:
  1. Cấu hình khởi động server (SERVER_NAME, PORT_SERVER, RESOURCES_DIR)
  2. Cấu hình cho archive_api/ — client gọi TRỰC TIẾP (live) Public
     Archive API để phục vụ tool search_archives.
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
    ARCHIVE_API_BASE_URL = os.getenv("ARCHIVE_API_BASE_URL", "http://192.168.1.100:4010")
    ARCHIVE_SEARCH_PATH = os.getenv("ARCHIVE_SEARCH_PATH", "/api/public/archives")
    HTTP_TIMEOUT_SECONDS = int(os.getenv("HTTP_TIMEOUT_SECONDS", "60"))

    AUTH_ENABLED = os.getenv("AUTH_ENABLED", "false").lower() == "true"
    CHATBOT_TOKEN_PATH = os.getenv("CHATBOT_TOKEN_PATH", "/api/v1/chatbot/session-token")
    CHATBOT_CLIENT_ID = os.getenv("CHATBOT_CLIENT_ID", "")
    CHATBOT_CLIENT_SECRET = os.getenv("CHATBOT_CLIENT_SECRET", "")

    # Trần số lượng keyword được fan-out song song trong 1 lần gọi
    # search_archives (mỗi keyword = 1 request riêng qua asyncio.gather).
    # Không giới hạn sẽ để LLM phía chatbot vô tình truyền quá nhiều biến
    # thể từ khóa trong 1 lượt, bắn hàng chục request đồng thời vào chính
    # Archive API nội bộ. Các keyword vượt trần bị cắt bớt (giữ đúng thứ
    # tự được truyền vào), không báo lỗi — đủ dùng cho use case thực tế
    # (VD tên có dấu + không dấu + viết tắt thường chỉ 2-4 biến thể).
    MAX_KEYWORDS_FANOUT = int(os.getenv("MAX_KEYWORDS_FANOUT", "8"))


config_object = Config()