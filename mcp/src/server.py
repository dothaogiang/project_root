"""
Entry point MCP server. Chạy: python src/server.py
"""
import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.registry import ToolRegistry
from logger import get_logger
from archive_api.client import get_archive_api_client

logger = get_logger(__name__)

mcp = ToolRegistry(name="profile_lookup")


async def main():
    server_mcp = await mcp.register_tools(category="mcp")
    return server_mcp


if __name__ == "__main__":
    server_mcp = asyncio.run(main())
    logger.info("Server sẵn sàng, bắt đầu lắng nghe...")
    try:
        server_mcp.run(transport="streamable-http")
    finally:
        # Đóng connection pool tới Archive API khi server tắt (Ctrl+C /
        # SIGTERM), tránh giữ socket mở lại vô ích sau khi process dừng
        # nhận request mới.
        asyncio.run(get_archive_api_client().aclose())