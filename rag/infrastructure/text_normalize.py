import unicodedata


def strip_diacritics(text: str) -> str:
    """Bỏ dấu tiếng Việt, giữ nguyên chữ cái gốc (đ vẫn cần map tay vì
    unicodedata không tự tách đ -> d)."""
    if not text:
        return text
    text = text.replace("đ", "d").replace("Đ", "D")
    normalized = unicodedata.normalize("NFD", text)
    return "".join(c for c in normalized if unicodedata.category(c) != "Mn")