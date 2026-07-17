"""
application/ingestion_service.py — Use case "đồng bộ dữ liệu": lấy hồ
sơ từ Archive API -> extract MD -> chunk -> embed -> upsert Qdrant.

Đây là bản chuyển thể của sync_job.py cũ, nhưng KHÔNG còn import trực
tiếp httpx/fastembed/qdrant_client — toàn bộ phụ thuộc hạ tầng được
TRUYỀN VÀO qua constructor (dependency injection), theo đúng port khai
báo ở rag/ports/interfaces.py. Nhờ vậy có thể unit test IngestionService
bằng cách truyền vào các fake/mock implement port, không cần Qdrant hay
mạng thật.

LƯU Ý: job này chạy MỘT LẦN DUY NHẤT (one-off), không có cơ chế
checkpoint/resume hay incremental sync — mỗi lần chạy sẽ duyệt và
nhúng lại toàn bộ dữ liệu từ đầu. Việc upsert vào Qdrant vẫn an toàn
khi chạy lại nhiều lần vì point ID được tính deterministic từ
(archive_id, file_url, chunk_index) — chạy lại chỉ ghi đè, không tạo
bản trùng.
"""
import asyncio

from rag.domain.entities import ArchiveRecord
from rag.ports.interfaces import (
    ArchiveApiClientPort,
    EmbeddingProviderPort,
    FileExtractorPort,
    VectorStorePort,
)
from rag.logger import get_logger
from rag.infrastructure.text_normalize import strip_diacritics

logger = get_logger(__name__)


def _build_archive_search_text(archive: ArchiveRecord) -> str:
    staff_lines = "; ".join(
        f"{m.get('fieldName', '')}: {m.get('value', '')}" for m in archive.staff_metadata
    )
    project_names = "; ".join(p.get("name", "") for p in archive.projects)
    parts = [
        f"Tiêu đề: {archive.title}",
        f"Mã hồ sơ: {archive.arc_file_code}",
        f"Thời gian: {archive.start_date} - {archive.end_date}",
        f"Thông tin cán bộ: {staff_lines}" if staff_lines else "",
        f"Tài liệu: {project_names}" if project_names else "",
    ]
    text = ". ".join(p for p in parts if p)
    return strip_diacritics(text)


class IngestionService:
    def __init__(
        self,
        archive_api: ArchiveApiClientPort,
        md_extractor: FileExtractorPort,
        embedder: EmbeddingProviderPort,
        vector_store: VectorStorePort,
        page_size: int = 100,
        sync_concurrency: int = 4,
    ):
        self._archive_api = archive_api
        self._md_extractor = md_extractor
        self._embedder = embedder
        self._vector_store = vector_store
        self._page_size = page_size
        self._sync_semaphore = asyncio.Semaphore(sync_concurrency)

    async def run(self) -> None:
        self._vector_store.ensure_collections()
        logger.info("Bắt đầu đồng bộ (chạy 1 lần, toàn bộ dữ liệu)")

        page = 0
        while True:
            archives, is_last = await self._archive_api.fetch_page(page, self._page_size)
            if not archives:
                break

            for archive in archives:
                try:
                    await self._sync_one_archive(archive)
                except Exception as e:
                    logger.error(f"Lỗi khi sync archive {archive.id}: {e}")

            if is_last:
                break
            page += 1

        logger.info("Đồng bộ hoàn tất.")

    async def _sync_one_archive(self, archive: ArchiveRecord) -> None:
        await self._sync_archive_metadata(archive)

        markdown_docs = archive.markdown_documents()
        if not markdown_docs:
            logger.info(f"Archive {archive.id}: không có nội dung Markdown, bỏ qua nội dung chi tiết")
            return

        # QUAN TRỌNG: try/except TỪNG file riêng biệt. Trước đây nếu 1
        # file trong archive lỗi lúc chunk/embed/upsert (network timeout,
        # embedding lỗi...), exception bay thẳng lên _sync_one_archive
        # -> bị try/except ở run() nuốt -> CÁC FILE CÒN LẠI của CÙNG
        # archive đó cũng bị bỏ qua theo, dù bản thân chúng hoàn toàn ổn.
        # Hậu quả: archive vẫn có metadata trong collection "archives"
        # (search_profiles/search_archives thấy bình thường) nhưng
        # "document_chunks" thiếu 1 phần hoặc toàn bộ nội dung, khiến
        # get_profile_detail/find_profile_and_answer luôn found=False dù
        # hồ sơ rõ ràng tồn tại. Cô lập lỗi theo từng file để 1 file hỏng
        # không kéo sập các file lành trong cùng archive.
        for project_name, file_url, text in markdown_docs:
            try:
                await self._sync_file(archive, project_name, file_url, text)
            except Exception as e:
                logger.error(
                    f"Lỗi khi sync file '{file_url or project_name}' của archive {archive.id}: {e}"
                )

    async def _sync_archive_metadata(self, archive: ArchiveRecord) -> None:
        search_text = _build_archive_search_text(archive)
        embedding = self._embedder.embed_text(search_text)
        self._vector_store.upsert_archive(archive, embedding)

    async def _sync_file(self, archive: ArchiveRecord, project_name: str, file_url: str, text: str) -> None:
        async with self._sync_semaphore:
            chunks = self._md_extractor.extract_and_chunk(archive.id, file_url, project_name, text)

            if not chunks:
                logger.warning(f"Không trích được text từ: {file_url or project_name}")
                return

            embeddings = self._embedder.embed_batch([c.text for c in chunks])
            self._vector_store.delete_chunks_by_file(archive.id, file_url)
            self._vector_store.upsert_chunks(chunks, embeddings)

            logger.info(f"Đã đồng bộ {len(chunks)} chunk từ: {file_url or project_name}")