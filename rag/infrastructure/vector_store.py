"""
infrastructure/vector_store.py — Adapter thao tác Qdrant. Implement
VectorStorePort.

2 collection:
  - "archives"        : 1 point = 1 hồ sơ (metadata) -> phục vụ search_profile
  - "document_chunks" : 1 point = 1 đoạn text trích từ PDF -> phục vụ
                         get_profile_detail (filter theo archive_id)

Đây là nơi DUY NHẤT trong rag/ import qdrant_client -> muốn đổi sang
vector DB khác (Milvus, pgvector...) chỉ cần viết 1 class mới implement
lại VectorStorePort, không đụng application/.
"""
import hashlib

from qdrant_client import QdrantClient, models

from rag.config.rag_config import rag_config
from rag.domain.entities import ArchiveRecord, DocumentChunk, Embedding, RetrievedChunk, RetrievedProfile
from rag.ports.interfaces import VectorStorePort
from rag.logger import get_logger

logger = get_logger(__name__)

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


def _point_id(*parts: str) -> str:
    """Point ID xác định (deterministic) -> upsert lại không tạo bản trùng."""
    return hashlib.md5(":".join(parts).encode("utf-8")).hexdigest()


def _to_qdrant_vector(embedding: Embedding) -> dict:
    return {
        DENSE_VECTOR_NAME: embedding.dense,
        SPARSE_VECTOR_NAME: models.SparseVector(
            indices=embedding.sparse_indices, values=embedding.sparse_values
        ),
    }


class QdrantVectorStore(VectorStorePort):
    def __init__(self):
        self._client: QdrantClient | None = None

    def _client_ready(self) -> QdrantClient:
        if self._client is None:
            self._client = QdrantClient(url=rag_config.QDRANT_URL, api_key=rag_config.QDRANT_API_KEY)
        return self._client

    def ensure_collections(self) -> None:
        client = self._client_ready()
        vectors_config = {
            DENSE_VECTOR_NAME: models.VectorParams(
                size=rag_config.DENSE_VECTOR_SIZE, distance=models.Distance.COSINE
            )
        }
        sparse_vectors_config = {SPARSE_VECTOR_NAME: models.SparseVectorParams()}

        for collection in (rag_config.COLLECTION_ARCHIVES, rag_config.COLLECTION_CHUNKS):
            if not client.collection_exists(collection):
                logger.info(f"Tạo collection: {collection}")
                client.create_collection(
                    collection_name=collection,
                    vectors_config=vectors_config,
                    sparse_vectors_config=sparse_vectors_config,
                )
            else:
                logger.info(f"Collection đã tồn tại: {collection}")

    def upsert_archive(self, archive: ArchiveRecord, embedding: Embedding) -> None:
        client = self._client_ready()
        payload = {
            "archive_id": archive.id,
            "title": archive.title,
            "arcFileCode": archive.arc_file_code,
            "boxCode": archive.box_code,
            "warehouseName": archive.warehouse_name,
            "startDate": archive.start_date,
            "endDate": archive.end_date,
            "status": archive.status,
            "staffMetadata": archive.staff_metadata,
        }
        client.upsert(
            collection_name=rag_config.COLLECTION_ARCHIVES,
            points=[
                models.PointStruct(
                    id=_point_id("archive", archive.id),
                    vector=_to_qdrant_vector(embedding),
                    payload=payload,
                )
            ],
        )

    def upsert_chunks(self, chunks: list[DocumentChunk], embeddings: list[Embedding]) -> None:
        if not chunks:
            return
        client = self._client_ready()
        points = []
        for chunk, emb in zip(chunks, embeddings):
            payload = {
                "archive_id": chunk.archive_id,
                "file_url": chunk.file_url,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
                "project_name": chunk.project_name,
                "extraction_method": chunk.extraction_method,
            }
            points.append(
                models.PointStruct(
                    id=_point_id("chunk", chunk.archive_id, chunk.file_url, str(chunk.chunk_index)),
                    vector=_to_qdrant_vector(emb),
                    payload=payload,
                )
            )
        client.upsert(collection_name=rag_config.COLLECTION_CHUNKS, points=points)

    def delete_chunks_by_file(self, archive_id: str, file_url: str) -> None:
        client = self._client_ready()
        client.delete(
            collection_name=rag_config.COLLECTION_CHUNKS,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(key="archive_id", match=models.MatchValue(value=archive_id)),
                        models.FieldCondition(key="file_url", match=models.MatchValue(value=file_url)),
                    ]
                )
            ),
        )

    def _hybrid_search(self, collection: str, embedding: Embedding, limit: int, query_filter=None):
        client = self._client_ready()
        result = client.query_points(
            collection_name=collection,
            prefetch=[
                models.Prefetch(
                    query=embedding.dense, using=DENSE_VECTOR_NAME, limit=limit * 3, filter=query_filter
                ),
                models.Prefetch(
                    query=models.SparseVector(indices=embedding.sparse_indices, values=embedding.sparse_values),
                    using=SPARSE_VECTOR_NAME,
                    limit=limit * 3,
                    filter=query_filter,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        return result.points

    def search_profiles(self, query_embedding: Embedding, top_k: int) -> list[RetrievedProfile]:
        points = self._hybrid_search(rag_config.COLLECTION_ARCHIVES, query_embedding, top_k)
        return [
            RetrievedProfile(
                archive_id=p.payload.get("archive_id"),
                title=p.payload.get("title"),
                arc_file_code=p.payload.get("arcFileCode"),
                box_code=p.payload.get("boxCode"),
                warehouse_name=p.payload.get("warehouseName"),
                start_date=p.payload.get("startDate"),
                end_date=p.payload.get("endDate"),
                staff_metadata=p.payload.get("staffMetadata") or [],
                score=round(p.score, 4),
            )
            for p in points
        ]

    def search_chunks(self, query_embedding: Embedding, archive_id: str, top_k: int) -> list[RetrievedChunk]:
        archive_filter = models.Filter(
            must=[models.FieldCondition(key="archive_id", match=models.MatchValue(value=archive_id))]
        )
        points = self._hybrid_search(rag_config.COLLECTION_CHUNKS, query_embedding, top_k, archive_filter)
        return [
            RetrievedChunk(
                text=p.payload.get("text"),
                file_url=p.payload.get("file_url"),
                page_number=p.payload.get("page_number"),
                extraction_method=p.payload.get("extraction_method"),
                score=round(p.score, 4),
            )
            for p in points
        ]
