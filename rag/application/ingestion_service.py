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

from rag.domain.entities import ArchiveRecord, DocumentChunk
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
        self._max_concurrency = sync_concurrency
        self._sync_semaphore = asyncio.Semaphore(sync_concurrency)

    async def run(self) -> None:
        self._vector_store.ensure_collections()
        logger.info("Bắt đầu đồng bộ (chạy 1 lần, toàn bộ dữ liệu, song song tối đa %d)", self._max_concurrency)

        page = 0
        while True:
            archives, is_last = await self._archive_api.fetch_page(page, self._page_size)
            if not archives:
                break

            # QUAN TRỌNG: xử lý các archive trong CÙNG 1 trang SONG SONG
            # (asyncio.gather) thay vì tuần tự từng archive như trước.
            # Trước đây có semaphore(sync_concurrency) khai báo sẵn ở
            # __init__ nhưng KHÔNG có chỗ nào gọi nhiều archive/file cùng
            # lúc để nó thực sự giới hạn — nên dù cấu hình 4, job vẫn
            # chỉ chạy như 1 luồng. Giờ gather() thực sự chạy song song,
            # semaphore(_max_concurrency) trong _sync_one_archive mới có
            # tác dụng giới hạn số archive đang embed/upsert cùng lúc
            # (tránh quá tải CPU/RAM khi trang có hàng trăm archive).
            await asyncio.gather(*(self._sync_archive_safe(archive) for archive in archives))

            if is_last:
                break
            page += 1

        logger.info("Đồng bộ hoàn tất.")

    async def _sync_archive_safe(self, archive: ArchiveRecord) -> None:
        """Bọc lỗi RIÊNG cho từng archive khi chạy trong gather() — 1
        archive lỗi không được để làm gather() hủy các archive khác
        đang chạy song song cùng lúc."""
        try:
            await self._sync_one_archive(archive)
        except Exception as e:
            logger.error(f"Lỗi khi sync archive {archive.id}: {e}")

    async def _sync_one_archive(self, archive: ArchiveRecord) -> None:
        await self._sync_archive_metadata(archive)

        markdown_docs = archive.markdown_documents()
        if not markdown_docs:
            logger.info(f"Archive {archive.id}: không có nội dung Markdown, bỏ qua nội dung chi tiết")
            return

        # BƯỚC 1 — extract + chunk TỪNG FILE riêng biệt, vẫn cô lập lỗi
        # theo file ở bước này: đây là bước CPU thuần (parse Markdown,
        # cắt chunk), rẻ và nhanh, rủi ro lỗi thấp — 1 file OCR lỗi định
        # dạng không nên chặn các file lành trong cùng archive.
        all_chunks: list[DocumentChunk] = []
        touched_file_urls: set[str] = set()
        for project_name, file_url, text in markdown_docs:
            try:
                chunks = await asyncio.to_thread(
                    self._md_extractor.extract_and_chunk, archive.id, file_url, project_name, text
                )
                if not chunks:
                    logger.warning(f"Không trích được text từ: {file_url or project_name}")
                    continue
                all_chunks.extend(chunks)
                touched_file_urls.add(file_url)
            except Exception as e:
                logger.error(
                    f"Lỗi khi extract file '{file_url or project_name}' của archive {archive.id}: {e}"
                )

        if not all_chunks:
            return

        # BƯỚC 2 — embed TOÀN BỘ chunk của archive này trong 1 lần gọi
        # embed_batch() DUY NHẤT, thay vì gọi riêng cho từng file như
        # trước. fastembed/ONNX có chi phí cố định (tokenize, khởi tạo
        # phiên...) cho MỖI lần gọi ngoài phần tính toán thật -> archive
        # có nhiều file nhỏ sẽ nhanh hơn rõ rệt khi gộp thành 1 batch
        # lớn thay vì N lần gọi nhỏ.
        # ĐÁNH ĐỔI: mất cô lập lỗi theo file ở bước embed/upsert — nếu
        # embed_batch() lỗi (hết bộ nhớ, model lỗi...), CẢ archive mất
        # chunk cho lần chạy này, không chỉ 1 file. Chấp nhận được vì
        # upsert là idempotent (point ID tính từ archive_id/file_url/
        # chunk_index) — chạy lại sync_job sẽ tự bù lại, không tạo bản
        # trùng. Nếu sau này thấy archive rất lớn (hàng trăm file) hay
        # bị mất trắng vì 1 lỗi nhỏ, có thể tách nhỏ theo lô (batch theo
        # từng 20-30 file) thay vì gộp toàn bộ archive.
        #
        # asyncio.to_thread(): embed_batch/upsert_chunks là hàm ĐỒNG BỘ
        # (port không async) — gọi trực tiếp trong coroutine sẽ CHẶN
        # event loop, khiến các archive khác đang "chạy song song" qua
        # gather() thực ra vẫn phải xếp hàng chờ. Đẩy sang thread riêng
        # để nhường event loop cho các archive khác trong lúc archive
        # này đang embed/upsert.
        async with self._sync_semaphore:
            embeddings = await asyncio.to_thread(self._embedder.embed_batch, [c.text for c in all_chunks])
            for file_url in touched_file_urls:
                await asyncio.to_thread(self._vector_store.delete_chunks_by_file, archive.id, file_url)
            await asyncio.to_thread(self._vector_store.upsert_chunks, all_chunks, embeddings)

        logger.info(
            f"Archive {archive.id}: đã đồng bộ {len(all_chunks)} chunk từ {len(touched_file_urls)} file"
        )

    async def _sync_archive_metadata(self, archive: ArchiveRecord) -> None:
        search_text = _build_archive_search_text(archive)
        async with self._sync_semaphore:
            embedding = await asyncio.to_thread(self._embedder.embed_text, search_text)
            await asyncio.to_thread(self._vector_store.upsert_archive, archive, embedding)