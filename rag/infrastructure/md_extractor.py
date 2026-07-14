from rag.config.rag_config import rag_config
from rag.domain.entities import DocumentChunk
from rag.ports.interfaces import FileExtractorPort
from rag.logger import get_logger

logger = get_logger(__name__)

EXTRACTION_MD = "md"


class MdExtractor(FileExtractorPort):
    def extract_and_chunk(
        self, archive_id: str, file_url: str, project_name: str, text: str
    ) -> list[DocumentChunk]:
        text = (text or "").strip()
        if not text:
            return []

        raw_chunks = self._chunk_text(text)
        return [
            DocumentChunk(
                archive_id=archive_id,
                file_url=file_url,
                chunk_index=i,
                page_number=1,  # MD không có khái niệm "trang" như PDF
                text=chunk,
                extraction_method=EXTRACTION_MD,
                project_name=project_name,
            )
            for i, chunk in enumerate(raw_chunks)
        ]

    def _chunk_text(self, text: str) -> list[str]:
        chunk_size = rag_config.CHUNK_SIZE_CHARS
        overlap = rag_config.CHUNK_OVERLAP_CHARS

        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            piece = text[start:end].strip()
            if piece:
                chunks.append(piece)
            if end == len(text):
                break
            start = end - overlap
        return chunks