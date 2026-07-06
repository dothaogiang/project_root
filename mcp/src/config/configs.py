"""
Cấu hình của riêng MCP server (folder mcp/). KHÔNG chứa cấu hình RAG
nữa (Archive API, Qdrant, embedding, OCR, chunking, sync state) — toàn
bộ phần đó đã chuyển sang rag/config/rag_config.py, vì mcp/ chỉ cần
biết cách khởi động server và tìm tools.yaml, không cần biết chi tiết
hạ tầng RAG bên dưới.
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


config_object = Config()
