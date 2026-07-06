"""
rag/ — Module RAG (Retrieval-Augmented Generation) độc lập.

Trách nhiệm DUY NHẤT của module này: lấy hồ sơ từ Public Archive API,
xử lý (extract/OCR PDF, chunk, embed) và lưu vào Qdrant, đồng thời cung
cấp cổng truy vấn (retrieval) cho tầng khác dùng (MCP tools, chatbot...).

Module này KHÔNG biết gì về MCP, FastMCP, hay tools.yaml — nó chỉ quan
tâm tới dữ liệu: nạp vào (ingestion) và lấy ra (retrieval). Điều này cho
phép:
  - Tái sử dụng ở nơi khác (một script batch, một service khác) mà
    không cần kéo theo toàn bộ tầng MCP.
  - Test độc lập, mock từng thành phần qua các port (interface) trong
    rag/ports/interfaces.py.
  - Đổi hạ tầng (VD: đổi Qdrant sang Milvus, đổi fastembed sang OpenAI
    embedding) chỉ cần viết 1 adapter mới trong rag/infrastructure/,
    không đụng vào application/ hay domain/.
"""
