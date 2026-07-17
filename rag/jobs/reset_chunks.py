"""
rag/jobs/reset_chunks.py — Xóa sạch collection 'document_chunks' (giữ
nguyên 'archives') rồi tạo lại rỗng, dùng 1 LẦN sau khi fix bug trùng
lặp chunk do đổi host API (xem _stable_file_key trong vector_store.py).

Vì "archives" không bị ảnh hưởng bởi bug này (point ID không phụ thuộc
file_url), KHÔNG cần xóa "archives" — chỉ cần xóa "document_chunks" rồi
chạy lại `python -m rag.jobs.sync_job` để ingest lại sạch với code mới
(dùng file_key ổn định, không phụ thuộc host).

Chạy:
    python rag/jobs/reset_chunks.py
    (rồi ngay sau đó: python -m rag.jobs.sync_job)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qdrant_client import QdrantClient

from rag.config.rag_config import rag_config

client = QdrantClient(url=rag_config.QDRANT_URL, api_key=rag_config.QDRANT_API_KEY)

collection = rag_config.COLLECTION_CHUNKS

if client.collection_exists(collection):
    before = client.count(collection_name=collection).count
    client.delete_collection(collection_name=collection)
    print(f"Đã xóa collection '{collection}' (trước đó có {before} point).")
else:
    print(f"Collection '{collection}' chưa tồn tại, không cần xóa.")

print("Giờ chạy: python -m rag.jobs.sync_job  (sẽ tự tạo lại collection rỗng và ingest lại sạch)")