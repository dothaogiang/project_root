"""
rag/jobs/check_chunks.py — Script debug nhanh: kiểm tra collection
'document_chunks' trong Qdrant có point nào ứng với 1 archive_id cụ
thể hay không (dùng khi search_archives/search_profiles thấy hồ sơ
nhưng get_profile_detail/find_profile_and_answer lại found=False).

Chạy:
    python rag/jobs/check_chunks.py <archive_id>
    hoặc
    python -m rag.jobs.check_chunks <archive_id>
"""
import sys
from pathlib import Path

# Giống sync_job.py: cho phép chạy trực tiếp file này lẫn qua -m,
# thêm project root (cha của rag/) vào sys.path để `import rag...` hoạt động.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qdrant_client import QdrantClient, models

from rag.config.rag_config import rag_config

archive_id = sys.argv[1] if len(sys.argv) > 1 else "50000000-0000-0000-0000-000000000003"

client = QdrantClient(url=rag_config.QDRANT_URL, api_key=rag_config.QDRANT_API_KEY)

count = client.count(
    collection_name=rag_config.COLLECTION_CHUNKS,
    count_filter=models.Filter(
        must=[models.FieldCondition(key="archive_id", match=models.MatchValue(value=archive_id))]
    ),
).count

print(f"Collection: {rag_config.COLLECTION_CHUNKS}")
print(f"archive_id: {archive_id}")
print(f"Số chunk tìm thấy: {count}")

if count == 0:
    total = client.count(collection_name=rag_config.COLLECTION_CHUNKS).count
    print(f"\n-> KHÔNG có chunk nào cho archive này.")
    print(f"Tổng số chunk toàn bộ collection: {total}")
    print("Kiểm tra log lúc chạy `python -m rag.jobs.sync_job` xem có dòng nào là:")
    print(f'  "Lỗi khi sync file ... của archive {archive_id}: ..."')
    print(f'  "Archive {archive_id}: không có nội dung Markdown, bỏ qua nội dung chi tiết"')
    print(f'  "Không trích được text từ: ..."')
else:
    sample = client.scroll(
        collection_name=rag_config.COLLECTION_CHUNKS,
        scroll_filter=models.Filter(
            must=[models.FieldCondition(key="archive_id", match=models.MatchValue(value=archive_id))]
        ),
        limit=3,
        with_payload=True,
        with_vectors=False,
    )[0]
    print("\nVí dụ payload:")
    for p in sample:
        print(f"  - file_url={p.payload.get('file_url')!r} text[:80]={p.payload.get('text', '')[:80]!r}")