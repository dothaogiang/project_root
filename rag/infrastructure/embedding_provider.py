"""
infrastructure/embedding_provider.py — Sinh dense + sparse embedding
bằng fastembed (chạy local, không tốn phí theo request). Implement
EmbeddingProviderPort.

- Dense (BAAI/bge-m3): bắt nghĩa (semantic).
- Sparse (Qdrant/bm25): bắt từ khóa chính xác (lexical).
Model được load 1 lần (singleton) vì load model tốn vài giây.
"""
from functools import lru_cache

from fastembed import SparseTextEmbedding, TextEmbedding

from rag.config.rag_config import rag_config
from rag.domain.entities import Embedding
from rag.ports.interfaces import EmbeddingProviderPort
from rag.logger import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def _dense_model() -> TextEmbedding:
    logger.info(f"Loading dense embedding model: {rag_config.DENSE_MODEL_NAME}")
    return TextEmbedding(model_name=rag_config.DENSE_MODEL_NAME)


@lru_cache(maxsize=1)
def _sparse_model() -> SparseTextEmbedding:
    logger.info(f"Loading sparse embedding model: {rag_config.SPARSE_MODEL_NAME}")
    return SparseTextEmbedding(model_name=rag_config.SPARSE_MODEL_NAME)


def _safe(text: str) -> str:
    return text if text and text.strip() else " "


class FastEmbedProvider(EmbeddingProviderPort):
    def embed_text(self, text: str) -> Embedding:
        text = _safe(text)
        dense = list(_dense_model().embed([text]))[0].tolist()
        sparse = list(_sparse_model().embed([text]))[0]
        return Embedding(dense=dense, sparse_indices=sparse.indices.tolist(), sparse_values=sparse.values.tolist())

    def embed_batch(self, texts: list[str]) -> list[Embedding]:
        if not texts:
            return []
        safe_texts = [_safe(t) for t in texts]
        dense_vecs = [e.tolist() for e in _dense_model().embed(safe_texts)]
        sparse_vecs = list(_sparse_model().embed(safe_texts))
        return [
            Embedding(dense=d, sparse_indices=s.indices.tolist(), sparse_values=s.values.tolist())
            for d, s in zip(dense_vecs, sparse_vecs)
        ]
