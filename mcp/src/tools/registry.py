"""
ToolRegistry: match tool_name (định nghĩa trong tools.yaml) với hàm thực
thi tương ứng trong FeatureManager (match theo TÊN HÀM), rồi build Tool
object cho FastMCP.

Đây là cơ chế "dynamic registration" -> thêm tool mới chỉ cần:
  1. Thêm entry trong Resources/tools.yaml
  2. Viết 1 method cùng tên trong FeatureManager
Không cần sửa registry.py hay server.py.
"""
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.tools.base import Tool
from mcp.server.fastmcp.utilities.func_metadata import func_metadata

from tools.manager import CustomToolManager
from feature_manager import FeatureManager
from config.configs import config_object
from logger import get_logger

logger = get_logger(__name__)


class ToolRegistry:
    def __init__(self, name: str):
        self.name = name
        self.tool_manager = CustomToolManager()
        self.list_tool = []

    async def register_tools(self, category: str = "mcp") -> FastMCP:
        tools_to_register = self.tool_manager.get_tools_by_category(category)

        if not tools_to_register:
            logger.warning(f"Không có tool nào trong category '{category}'")

        for tool_name, tool_definition in tools_to_register.items():
            fn = getattr(FeatureManager, tool_name, None)
            if fn is None:
                logger.error(
                    f"Tool '{tool_name}' khai báo trong tools.yaml nhưng KHÔNG "
                    f"tìm thấy hàm cùng tên trong FeatureManager -> bỏ qua. "
                    f"Kiểm tra lại chính tả name_tool hay method tương ứng."
                )
                continue

            fn_metadata = func_metadata(fn, skip_names=[])

            tool = Tool(
                fn=fn,
                title=tool_name,
                name=tool_name,
                description=tool_definition["description"],
                parameters=tool_definition["inputSchema"],
                is_async=True,
                fn_metadata=fn_metadata,
                context_kwarg=None,
                annotations=None,
            )
            self.list_tool.append(tool)
            logger.info(f"Đã đăng ký tool: {tool_name}")

        server_mcp = FastMCP(
            name=self.name,
            tools=self.list_tool,
            host=config_object.URL_HOST_SERVER,
            port=config_object.PORT_SERVER,
        )
        return server_mcp
