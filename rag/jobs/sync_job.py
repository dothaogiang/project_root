"""
jobs/sync_job.py — Entry point chạy đồng bộ dữ liệu (composition root
của module rag/).

Đây là nơi DUY NHẤT "lắp ráp" (wire) các adapter cụ thể
(HttpArchiveApiClient, PyMuPdfExtractor, FastEmbedProvider,
QdrantVectorStore, SqliteSyncStateRepo) vào IngestionService thông qua
constructor injection. application/ingestion_service.py không hề biết
đến các class cụ thể này.

Chạy thủ công:
    python -m rag.jobs.sync_job
Chạy định kỳ: đặt trong crontab / APScheduler / k8s CronJob / Docker
Compose service `sync_cron` (mỗi giờ là hợp lý cho archive/document ít
thay đổi trong ngày).
"""
import asyncio
import sys
from pathlib import Path

# Cho phép chạy trực tiếp file này (`python rag/jobs/sync_job.py`) lẫn
# qua `python -m rag.jobs.sync_job`. rag/ giờ không còn phụ thuộc mcp/src
# (đã có logger.py riêng), nên chỉ cần thêm project root (cha của rag/)
# vào sys.path để `import rag...` hoạt động.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rag.application.ingestion_service import IngestionService
from rag.config.rag_config import rag_config
from rag.infrastructure.archive_api_client import HttpArchiveApiClient
from rag.infrastructure.embedding_provider import FastEmbedProvider
from rag.infrastructure.pdf_extractor import PyMuPdfExtractor
from rag.infrastructure.sync_state_repo import SqliteSyncStateRepo
from rag.infrastructure.vector_store import QdrantVectorStore
from rag.logger import get_logger

logger = get_logger(__name__)


async def main():
    async with HttpArchiveApiClient() as archive_api:
        service = IngestionService(
            archive_api=archive_api,
            pdf_extractor=PyMuPdfExtractor(),
            embedder=FastEmbedProvider(),
            vector_store=QdrantVectorStore(),
            sync_state=SqliteSyncStateRepo(),
            page_size=rag_config.ARCHIVE_API_PAGE_SIZE,
            ocr_concurrency=rag_config.OCR_CONCURRENCY,
        )
        await service.run(resume=True)


if __name__ == "__main__":
    asyncio.run(main())
