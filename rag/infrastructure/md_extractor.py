from rag.config.rag_config import rag_config
from rag.domain.entities import DocumentChunk
from rag.ports.interfaces import FileExtractorPort
from rag.logger import get_logger

logger = get_logger(__name__)

EXTRACTION_MD = "md"

# Thứ tự ưu tiên tách — từ "thô" (giữ nhiều ngữ cảnh nhất) tới "mịn"
# (chỉ dùng khi bắt buộc): tách theo heading/đoạn Markdown trước, rồi
# xuống dòng, rồi câu, rồi từ, cuối cùng mới tới ký tự. Chỉ tách sâu
# hơn 1 mức khi phần hiện tại VẪN dài hơn chunk_size — nhờ vậy 1 câu
# ngắn không bao giờ bị cắt ngang giữa chừng chỉ vì đoạn chứa nó dài,
# và ranh giới chunk luôn rơi vào chỗ "trọn ý" nhất có thể.
_DEFAULT_SEPARATORS = [
    "\n## ",   # heading cấp 2 Markdown — ranh giới mục lớn
    "\n### ",  # heading cấp 3
    "\n\n",    # đoạn văn
    "\n",      # xuống dòng đơn
    ". ",
    "! ",
    "? ",
    "; ",
    ", ",
    " ",       # từ
    "",        # ký tự — chốt chặn cuối, luôn cắt được
]


def _split_keep_separator(text: str, separator: str) -> list[str]:
    """Tách text theo separator nhưng GIỮ separator ở cuối mỗi phần
    (trừ phần cuối cùng), để không mất dấu câu/ngắt dòng khi gộp lại —
    khác với str.split() thông thường sẽ bỏ mất separator."""
    if separator == "":
        return list(text)
    parts = text.split(separator)
    return [p + separator if i < len(parts) - 1 else p for i, p in enumerate(parts)]


def _recursive_split(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    """Tách `text` thành các đoạn nhỏ (chưa gộp theo chunk_size) bằng
    cách thử lần lượt từng separator theo thứ tự ưu tiên trong danh
    sách. Nếu 1 đoạn con vẫn dài hơn chunk_size sau khi tách, đệ quy
    tách tiếp bằng separator MỊN HƠN (phần tử kế tiếp của danh sách)
    — đây chính là phần "hierarchical/recursive" của thuật toán,
    tương tự RecursiveCharacterTextSplitter."""
    if not text:
        return []
    if len(text) <= chunk_size or not separators:
        return [text]

    sep, *rest = separators
    parts = _split_keep_separator(text, sep)
    if len(parts) == 1:
        # separator hiện tại không xuất hiện trong text -> thử separator
        # mịn hơn kế tiếp thay vì trả nguyên khối
        return _recursive_split(text, chunk_size, rest)

    pieces: list[str] = []
    for part in parts:
        if not part:
            continue
        if len(part) > chunk_size:
            pieces.extend(_recursive_split(part, chunk_size, rest))
        else:
            pieces.append(part)
    return pieces


def _merge_pieces(pieces: list[str], chunk_size: int, overlap: int) -> list[str]:
    """Gộp các đoạn nhỏ (câu/đoạn) do _recursive_split trả về thành
    từng chunk có độ dài gần chunk_size nhất có thể, KHÔNG BAO GIỜ cắt
    ngang giữa 1 đoạn nhỏ đã tách. Overlap được tạo bằng cách giữ lại
    vài đoạn nhỏ CUỐI của chunk vừa đóng làm điểm bắt đầu cho chunk kế
    tiếp, thay vì lùi lại theo số ký tự cố định như sliding-window
    thô — nhờ vậy phần overlap cũng luôn là câu/đoạn trọn vẹn."""
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for piece in pieces:
        piece_len = len(piece)
        if current and current_len + piece_len > chunk_size:
            joined = "".join(current).strip()
            if joined:
                chunks.append(joined)

            overlap_pieces: list[str] = []
            overlap_len = 0
            for p in reversed(current):
                if overlap_len >= overlap:
                    break
                overlap_pieces.insert(0, p)
                overlap_len += len(p)
            current = overlap_pieces
            current_len = overlap_len

        current.append(piece)
        current_len += piece_len

    tail = "".join(current).strip()
    if tail:
        chunks.append(tail)
    return chunks


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
                page_number=1,
                text=chunk,
                extraction_method=EXTRACTION_MD,
                project_name=project_name,
            )
            for i, chunk in enumerate(raw_chunks)
        ]

    def _chunk_text(self, text: str) -> list[str]:
        """Chunking đệ quy/phân cấp (recursive/hierarchical): thử tách
        theo heading -> đoạn -> dòng -> câu -> từ -> ký tự, chỉ tách
        mịn hơn khi phần hiện tại còn dài hơn CHUNK_SIZE_CHARS, sau đó
        gộp các đoạn nhỏ lại thành chunk ~CHUNK_SIZE_CHARS có overlap
        ~CHUNK_OVERLAP_CHARS. Khác biệt so với sliding-window ký tự
        thô trước đây: ranh giới chunk luôn rơi vào chỗ ngắt câu/đoạn
        tự nhiên, không cắt ngang giữa câu -> chunk giữ trọn ngữ cảnh,
        có lợi cho chất lượng embedding/retrieval."""
        chunk_size = rag_config.CHUNK_SIZE_CHARS
        overlap = rag_config.CHUNK_OVERLAP_CHARS

        pieces = _recursive_split(text, chunk_size, _DEFAULT_SEPARATORS)
        return _merge_pieces(pieces, chunk_size, overlap)