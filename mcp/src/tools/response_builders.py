"""
Helpers for shaping MCP tool responses.

FeatureManager should orchestrate tool calls; this module keeps repeated
payload formatting and light Archive API record normalization out of it.
"""
from typing import Optional
from urllib.parse import parse_qs, urlparse

from logger import get_logger

logger = get_logger(__name__)

BRIEF_FIELDS = {
    "id", "title", "arcFileCode", "status", "warehouseName",
    "roomNumber", "shelfCode", "shelfLevelCode",
    "startDate", "endDate", "hasFiles", "title_match", "files",
}

SEMANTIC_FALLBACK_FILTER_NOTE = (
    "Kết quả semantic fallback chỉ là ứng viên gần đúng và chưa áp dụng lại "
    "các filter đã truyền; cần kiểm tra field/filter trước khi khẳng định."
)


def extract_file_key(file_url: Optional[str]) -> Optional[str]:
    """Extract the stable file key from proxy URLs like .../files/proxy?key=xxx."""
    if not file_url:
        return None
    parsed = urlparse(file_url)
    qs = parse_qs(parsed.query)
    values = qs.get("key")
    return values[0] if values else None


def normalize_vn(text: Optional[str]) -> str:
    """Lowercase only; keep Vietnamese diacritics because they change meaning."""
    return (text or "").lower()


def compact_record(record: dict) -> dict:
    """Drop empty optional fields from a response record."""
    return {k: v for k, v in record.items() if v not in (None, [], "")}


def files_from_record(record: dict) -> list[dict]:
    """Flatten projects[].documents[] into compact file metadata."""
    files = []
    for project in record.get("projects", []):
        for doc in project.get("documents", []):
            files.append({
                "fileName": doc.get("fileName"),
                "fileKey": extract_file_key(doc.get("fileUrl")),
                "fileUrl": doc.get("fileUrl"),
            })
    return files


async def fetch_files_for_profiles(client, profiles: list) -> dict[str, list[dict]]:
    """
    Enrich semantic profile hits with real file links from the live Archive API.

    This is best-effort. If the live API is unavailable, semantic fallback
    results are still useful without file links.
    """
    codes = list({p.arc_file_code for p in profiles if p.arc_file_code})
    if not codes:
        return {}
    try:
        result = await client.search_archives(keywords=codes, size=max(len(codes) * 2, 20))
    except Exception:
        logger.exception("Không lấy được files/link thật cho semantic fallback profiles")
        return {}

    files_by_id: dict[str, list[dict]] = {}
    for record in result.get("content", []):
        rid = record.get("id")
        if rid is None:
            continue
        files = files_from_record(record)
        if files:
            files_by_id[str(rid)] = files
    return files_by_id


def chunk_to_dict(chunk, with_archive_id: bool = False) -> dict:
    """Convert a RetrievedChunk into the public tool response shape."""
    payload = {
        "text": chunk.text,
        "file_url": chunk.file_url,
        "page_number": chunk.page_number,
        "extraction_method": chunk.extraction_method,
        "score": chunk.score,
    }
    if with_archive_id:
        payload = {
            "archive_id": chunk.archive_id,
            **payload,
            "project_name": chunk.project_name,
        }
    return payload


def profile_to_record(profile, files_by_id: dict[str, list[dict]], source: Optional[str] = None) -> dict:
    """Convert a RetrievedProfile into the Archive API-like record shape."""
    record = {
        "id": profile.archive_id,
        "title": profile.title,
        "arcFileCode": profile.arc_file_code,
        "shelfCode": profile.shelf_code,
        "shelfLevelCode": profile.shelf_level_code,
        "warehouseName": profile.warehouse_name,
        "startDate": profile.start_date,
        "endDate": profile.end_date,
        "staffMetadata": profile.staff_metadata,
        "hasFiles": bool(files_by_id.get(str(profile.archive_id))),
        "files": files_by_id.get(str(profile.archive_id), []),
        "_score": profile.score,
    }
    if source:
        record["_source"] = source
    return record
