"""
application/ingestion_service.py — Use case "đồng bộ dữ liệu": lấy hồ
sơ từ Archive API -> extract/OCR PDF -> chunk -> embed -> upsert Qdrant.

Đây là bản chuyển thể của sync_job.py cũ, nhưng KHÔNG còn import trực
tiếp httpx/fastembed/qdrant_client — toàn bộ phụ thuộc hạ tầng được
TRUYỀN VÀO qua constructor (dependency injection), theo đúng port khai
báo ở rag/ports/interfaces.py. Nhờ vậy có thể unit test IngestionService
bằng cách truyền vào các fake/mock implement port, không cần Qdrant hay
mạng thật.
"""
import asyncio
import hashlib

from rag.domain.entities import ArchiveRecord
from rag.ports.interfaces import (
    ArchiveApiClientPort,
    EmbeddingProviderPort,
    PdfExtractorPort,
    SyncStateRepoPort,
    VectorStorePort,
)
from rag.logger import get_logger

logger = get_logger(__name__)


def _build_archive_search_text(archive: ArchiveRecord) -> str:
    """Gộp field metadata thành 1 đoạn text tự nhiên để embed cho search_profile."""
    staff_lines = "; ".join(
        f"{m.get('fieldName', '')}: {m.get('value', '')}" for m in archive.staff_metadata
    )
    project_names = "; ".join(p.get("name", "") for p in archive.projects)
    parts = [
        f"Tiêu đề: {archive.title}",
        f"Mã hồ sơ: {archive.arc_file_code}",
        f"Kho: {archive.warehouse_name}",
        f"Thời gian: {archive.start_date} - {archive.end_date}",
        f"Thông tin cán bộ: {staff_lines}" if staff_lines else "",
        f"Tài liệu: {project_names}" if project_names else "",
    ]
    return ". ".join(p for p in parts if p)


class IngestionService:
    def __init__(
        self,
        archive_api: ArchiveApiClientPort,
        pdf_extractor: PdfExtractorPort,
        embedder: EmbeddingProviderPort,
        vector_store: VectorStorePort,
        sync_state: SyncStateRepoPort,
        page_size: int = 100,
        ocr_concurrency: int = 4,
    ):
        self._archive_api = archive_api
        self._pdf_extractor = pdf_extractor
        self._embedder = embedder
        self._vector_store = vector_store
        self._sync_state = sync_state
        self._page_size = page_size
        self._ocr_semaphore = asyncio.Semaphore(ocr_concurrency)

    async def run(self, resume: bool = True) -> None:
        self._vector_store.ensure_collections()

        start_page = self._sync_state.get_checkpoint_page() if resume else 0
        logger.info(f"Bắt đầu đồng bộ từ page {start_page}")

        page = start_page
        while True:
            archives, is_last = await self._archive_api.fetch_page(page, self._page_size)
            if not archives:
                break

            for archive in archives:
                try:
                    await self._sync_one_archive(archive)
                except Exception as e:
                    logger.error(f"Lỗi khi sync archive {archive.id}: {e}")

            self._sync_state.set_checkpoint_page(page)
            if is_last:
                break
            page += 1

        self._sync_state.reset_checkpoint()
        logger.info("Đồng bộ hoàn tất.")

    async def _sync_one_archive(self, archive: ArchiveRecord) -> None:
        last_synced = self._sync_state.get_archive_last_updated(archive.id)
        if last_synced == archive.updated_at:
            logger.info(f"Archive không đổi, bỏ qua: {archive.id}")
            return

        await self._sync_archive_metadata(archive)

        for project_name, file_url in archive.file_urls():
            await self._sync_file(archive, project_name, file_url)

        self._sync_state.set_archive_synced(archive.id, archive.updated_at)

    async def _sync_archive_metadata(self, archive: ArchiveRecord) -> None:
        search_text = _build_archive_search_text(archive)
        embedding = self._embedder.embed_text(search_text)
        self._vector_store.upsert_archive(archive, embedding)

    async def _sync_file(self, archive: ArchiveRecord, project_name: str, file_url: str) -> None:
        async with self._ocr_semaphore:
            try:
                pdf_bytes = await self._archive_api.download_file(file_url)
            except Exception as e:
                logger.error(f"Không tải được file {file_url}: {e}")
                return

            content_hash = hashlib.md5(pdf_bytes).hexdigest()
            old_hash = self._sync_state.get_file_hash(archive.id, file_url)
            if old_hash == content_hash:
                logger.info(f"File không đổi, bỏ qua: {file_url}")
                return

            logger.info(f"Đang extract: {file_url}")
            chunks = self._pdf_extractor.extract_and_chunk(archive.id, file_url, project_name, pdf_bytes)

            if not chunks:
                logger.warning(f"Không trích được text từ: {file_url}")
                self._sync_state.set_file_synced(archive.id, file_url, content_hash, "none", 0)
                return

            embeddings = self._embedder.embed_batch([c.text for c in chunks])

            # Xóa chunk cũ trước khi upsert mới, tránh chunk rác nếu file rút gọn nội dung
            self._vector_store.delete_chunks_by_file(archive.id, file_url)
            self._vector_store.upsert_chunks(chunks, embeddings)

            method = chunks[0].extraction_method
            self._sync_state.set_file_synced(archive.id, file_url, content_hash, method, len(chunks))
            logger.info(f"Đã đồng bộ {len(chunks)} chunk ({method}) từ: {file_url}")
