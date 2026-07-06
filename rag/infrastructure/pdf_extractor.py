"""
infrastructure/pdf_extractor.py — Trích text từ PDF (native hoặc OCR
tiếng Việt), rồi chia chunk. Implement PdfExtractorPort.

Logic auto-detect: thử extract text trực tiếp trước (rẻ). Nếu mật độ
ký tự/trang quá thấp (dấu hiệu PDF chỉ chứa ảnh scan) -> fallback OCR
bằng pytesseract.
"""
import io

import fitz  # PyMuPDF
import pytesseract
from PIL import Image

from rag.config.rag_config import rag_config
from rag.domain.entities import DocumentChunk
from rag.ports.interfaces import PdfExtractorPort
from rag.logger import get_logger

logger = get_logger(__name__)

EXTRACTION_NATIVE = "native"
EXTRACTION_OCR = "ocr"


class PyMuPdfExtractor(PdfExtractorPort):
    def extract_and_chunk(
        self, archive_id: str, file_url: str, project_name: str, pdf_bytes: bytes
    ) -> list[DocumentChunk]:
        pages, method = self._extract_pages(pdf_bytes)
        raw_chunks = self._chunk_text(pages)

        return [
            DocumentChunk(
                archive_id=archive_id,
                file_url=file_url,
                chunk_index=i,
                page_number=c["page_number"],
                text=c["text"],
                extraction_method=method,
                project_name=project_name,
            )
            for i, c in enumerate(raw_chunks)
        ]

    # -- internal helpers -------------------------------------------------

    def _ocr_page(self, page: "fitz.Page") -> str:
        pix = page.get_pixmap(dpi=rag_config.OCR_DPI)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        return pytesseract.image_to_string(img, lang=rag_config.OCR_LANG)

    def _extract_pages(self, pdf_bytes: bytes) -> tuple[list[str], str]:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        native_pages = [page.get_text() for page in doc]
        total_chars = sum(len(p.strip()) for p in native_pages)
        avg_chars_per_page = total_chars / max(len(doc), 1)

        if avg_chars_per_page >= rag_config.OCR_MIN_CHARS_PER_PAGE:
            logger.info(f"PDF text thật (avg {avg_chars_per_page:.0f} ký tự/trang) -> native extract")
            return native_pages, EXTRACTION_NATIVE

        logger.info(f"PDF nghi là scan (avg {avg_chars_per_page:.0f} ký tự/trang) -> chạy OCR")
        ocr_pages = []
        for i, page in enumerate(doc):
            try:
                ocr_pages.append(self._ocr_page(page))
            except Exception as e:
                logger.error(f"OCR lỗi tại trang {i}: {e}")
                ocr_pages.append("")
        return ocr_pages, EXTRACTION_OCR

    def _chunk_text(self, pages: list[str]) -> list[dict]:
        chunk_size = rag_config.CHUNK_SIZE_CHARS
        overlap = rag_config.CHUNK_OVERLAP_CHARS

        full_text = ""
        char_to_page = []
        for page_no, page_text in enumerate(pages, start=1):
            full_text += page_text
            char_to_page.extend([page_no] * len(page_text))

        full_text = full_text.strip()
        if not full_text:
            return []

        chunks = []
        start = 0
        while start < len(full_text):
            end = min(start + chunk_size, len(full_text))
            chunk_str = full_text[start:end].strip()
            if chunk_str:
                page_idx = min(start, len(char_to_page) - 1) if char_to_page else 0
                page_number = char_to_page[page_idx] if char_to_page else 1
                chunks.append({"text": chunk_str, "page_number": page_number})
            if end == len(full_text):
                break
            start = end - overlap
        return chunks
