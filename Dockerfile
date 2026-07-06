# Dockerfile đặt ở project root — build 1 image dùng chung cho cả 2
# service (mcp_server và sync_cron), vì mcp/src/server.py import trực
# tiếp module rag/ (feature_manager.py -> rag.retrieval_factory) trong
# CÙNG một tiến trình Python, nên venv chạy server bắt buộc phải có đủ
# dependency của cả 2 folder.
FROM python:3.11-slim

# pytesseract (OCR) cần binary tesseract-ocr + gói ngôn ngữ tiếng Việt
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-vie \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Cài dependency của cả 2 folder vào chung 1 venv hệ thống
COPY mcp/requirements.txt ./mcp-requirements.txt
COPY rag/requirements.txt ./rag-requirements.txt
RUN pip install --no-cache-dir -r mcp-requirements.txt -r rag-requirements.txt

# Copy source: giữ đúng cấu trúc 2 folder ngang hàng như trên máy dev
COPY mcp/ ./mcp/
COPY rag/ ./rag/

# PYTHONPATH gồm project root (để `import rag...` hoạt động) và mcp/src
# (để `import feature_manager`, `import logger`, `import tools...` bên
# trong mcp hoạt động như cấu trúc gốc, không cần sửa import trong tools/registry.py)
ENV PYTHONPATH=/app:/app/mcp/src

# Lệnh chạy cụ thể (server hay sync job) được override ở docker-compose.yaml
CMD ["python", "mcp/src/server.py"]
