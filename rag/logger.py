"""
rag/logger.py — Logger RIÊNG của module rag/.

Trước đây rag/ mượn tạm `logger.py` của mcp/src, khiến rag/ bị phụ
thuộc ngược vào mcp/ (vi phạm nguyên tắc 2 folder độc lập). File này
thay thế hoàn toàn cho việc import `from logger import get_logger`
(logger của mcp) bằng `from rag.logger import get_logger` (logger của
chính rag/).
"""
import logging
import os

_LOG_LEVEL = os.getenv("RAG_LOG_LEVEL", "INFO").upper()

_FORMATTER = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:  # tránh add handler trùng khi get_logger gọi nhiều lần
        handler = logging.StreamHandler()
        handler.setFormatter(_FORMATTER)
        logger.addHandler(handler)
        logger.setLevel(_LOG_LEVEL)
        logger.propagate = False
    return logger
