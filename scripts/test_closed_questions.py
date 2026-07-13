"""
scripts/test_closed_questions.py — Test "câu hỏi đóng" (câu hỏi có 1 đáp án
cụ thể, ví dụ "sinh ngày nào", "số quyết định là gì") trên dữ liệu MD giả
lập (archive-005, archive-006 trong scripts/fake_archive_api_server.py).

MỤC ĐÍCH: kiểm tra xem sau khi MdExtractor chia file MD thành các chunk
(theo đúng CHUNK_SIZE_CHARS/CHUNK_OVERLAP_CHARS thật trong .env), thông
tin cần trả lời có còn nằm TRỌN VẸN trong 1 chunk hay bị cắt đứt giữa 2
chunk — đây là điều kiện CẦN để tầng retrieval (RetrievalService) sau
này có cơ hội trả lời đúng câu hỏi đóng.

LƯU Ý: script này KHÔNG gọi embedding/Qdrant thật (fastembed + Qdrant
cần model/network riêng) — nó chỉ dùng đúng code fetch (
HttpArchiveApiClient) + chunk (MdExtractor) thật trong rag/, rồi so khớp
chuỗi (substring match) để xác nhận dữ liệu có "retrievable" hay không.
Muốn test luôn semantic search thật, dùng RetrievalService.search_chunks_in_archive
sau khi đã chạy `python -m rag.jobs.sync_job` để ingest data giả vào Qdrant.

CÁCH DÙNG:
    1. Chạy fake API:  python scripts/fake_archive_api_server.py
    2. Set .env:        ARCHIVE_API_BASE_URL=http://localhost:8000
                         ARCHIVE_API_PATH=/api/public/archive
    3. Chạy:            python scripts/test_closed_questions.py
"""
import asyncio

from rag.infrastructure.archive_api_client import HttpArchiveApiClient
from rag.infrastructure.md_extractor import MdExtractor

# (archive_id, câu hỏi đóng, chuỗi con PHẢI xuất hiện trong đáp án đúng)
CLOSED_QUESTIONS = [
    ("archive-005", "Nguyễn Văn Test01 sinh ngày nào?", "Sinh ngày 05 tháng 09 năm 1999"),
    ("archive-005", "Quê quán ở đâu?", "xã Bình An, tỉnh Nam Định"),
    ("archive-005", "Ngày nhập ngũ là khi nào?", "Ngày nhập ngũ: 15/03/2017"),
    ("archive-005", "Học vị cao nhất, chuyên ngành là gì?", "Kỹ sư (Cơ khí chế tạo, 6/2024)"),
    ("archive-005", "Họ tên cha là gì?", "NGUYỄN VĂN CHA01"),
    ("archive-005", "Đơn vị công tác hiện nay là gì?", "Lữ đoàn Test 01"),
    ("archive-006", "Số quyết định khen thưởng là gì?", "0000/QĐ-TEST"),
    ("archive-006", "Danh hiệu được tặng là gì?", "CHIẾN SĨ THI ĐUA CƠ SỞ"),
    ("archive-006", "Ngày ký quyết định là khi nào?", "01 tháng 01 năm 2024"),
    ("archive-006", "Số sổ vàng là bao nhiêu?", "Số sổ vàng: 000"),
]


async def main():
    extractor = MdExtractor()

    async with HttpArchiveApiClient() as client:
        # Gom toàn bộ archive (chỉ có 6 hồ sơ giả nên 1 page là đủ).
        archives, _ = await client.fetch_page(0, 100)
        by_id = {a.id: a for a in archives}

        # Chunk sẵn từng archive liên quan, tránh tải/chunk lại nhiều lần.
        chunks_by_archive: dict[str, list[str]] = {}
        for archive_id in {q[0] for q in CLOSED_QUESTIONS}:
            archive = by_id.get(archive_id)
            if archive is None:
                print(f"[BỎ QUA] Không tìm thấy archive_id={archive_id} trong fake API")
                continue
            md_urls = archive.md_file_urls()
            if not md_urls:
                print(f"[BỎ QUA] {archive_id} không có file MD")
                continue
            project_name, file_url = md_urls[0]
            file_bytes = await client.download_file(file_url)
            chunks = extractor.extract_and_chunk(archive_id, file_url, project_name, file_bytes)
            chunks_by_archive[archive_id] = [c.text for c in chunks]
            print(f"[INFO] {archive_id}: chia thành {len(chunks)} chunk "
                  f"(độ dài: {[len(t) for t in chunks_by_archive[archive_id]]})")

    print("\n" + "=" * 70)
    print("KẾT QUẢ TEST CÂU HỎI ĐÓNG")
    print("=" * 70)

    passed, failed = 0, 0
    for archive_id, question, expected_substring in CLOSED_QUESTIONS:
        chunks = chunks_by_archive.get(archive_id, [])
        found_in = [i for i, text in enumerate(chunks) if expected_substring in text]

        if found_in:
            passed += 1
            status = f"✅ PASS (nằm trong chunk #{found_in})"
        else:
            failed += 1
            status = "❌ FAIL (đáp án bị cắt rời hoặc không tồn tại trong bất kỳ chunk nào)"

        print(f"\n[{archive_id}] Q: {question}")
        print(f"   Kỳ vọng chứa: \"{expected_substring}\"")
        print(f"   {status}")

    print("\n" + "-" * 70)
    print(f"Tổng: {passed} PASS / {failed} FAIL / {len(CLOSED_QUESTIONS)} câu hỏi")


if __name__ == "__main__":
    asyncio.run(main())