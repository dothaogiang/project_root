"""
Entry point MCP server. Chạy: python src/server.py
"""
import asyncio
import sys
from pathlib import Path

# Cấu trúc: project_root/mcp/src/server.py  và  project_root/rag/
# server.py nằm ở độ sâu mcp/src -> cần đi lên 2 cấp (bỏ "src", bỏ "mcp")
# mới tới project_root, nơi chứa folder rag/. Thêm project_root vào
# sys.path để feature_manager.py import được `rag.retrieval_factory`.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.registry import ToolRegistry
from logger import get_logger

logger = get_logger(__name__)

mcp = ToolRegistry(name="profile_lookup")


async def main():
    server_mcp = await mcp.register_tools(category="mcp")
    return server_mcp


if __name__ == "__main__":
    server_mcp = asyncio.run(main())
    logger.info("Server sẵn sàng, bắt đầu lắng nghe...")
    server_mcp.run(transport="streamable-http")
