"""Kết nối vector store Day 6 (PostgreSQL + pgvector) và embedding model Day 6.

- Embedding: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (384 chiều,
  chuẩn hóa L2, chạy CPU), đúng model đã dùng để tạo index Day 6.
- Tìm kiếm: cosine chính xác (`<=>`), không ANN, giống Day 6.
- Kết quả trả về theo định dạng chunk Day 7: id, text, source_id, source, page,
  metadata, score, rank.

Biến môi trường: DATABASE_URL, EMBEDDING_MODEL, DAY6_STRATEGY (mặc định B).
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

load_dotenv(override=False)

TABLE = "chunks"
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# Cột được phép filter (whitelist, chống SQL injection qua tên cột)
FILTER_COLUMNS = {"district", "section", "document_id", "page"}


class Day6Embedder:
    """Nhúng query giống hệt cách Day 6 nhúng tài liệu."""

    def __init__(self, model_name: str | None = None, device: str = "cpu") -> None:
        from sentence_transformers import SentenceTransformer  # import nặng

        self.model_name = model_name or os.getenv("EMBEDDING_MODEL") or DEFAULT_MODEL
        self.model = SentenceTransformer(self.model_name, device=device)
        get_dim = getattr(self.model, "get_embedding_dimension", None) or \
            self.model.get_sentence_embedding_dimension
        self.dimensions = int(get_dim())

    def embed_query(self, text: str) -> list[float]:
        vec = self.model.encode([text], normalize_embeddings=True, convert_to_numpy=True,
                                show_progress_bar=False)[0]
        return vec.tolist()


class Day6Store:
    def __init__(self, database_url: str | None = None, strategy: str | None = None) -> None:
        self.database_url = database_url or os.getenv("DATABASE_URL", "")
        if not self.database_url:
            raise ValueError("Chưa có DATABASE_URL trong .env")
        self.strategy = strategy or os.getenv("DAY6_STRATEGY", "B")

    def connect(self) -> psycopg.Connection:
        conn = psycopg.connect(self.database_url, connect_timeout=5)
        register_vector(conn)
        return conn

    def dimensions(self) -> int:
        with self.connect() as conn:
            return conn.execute(
                "SELECT atttypmod FROM pg_attribute WHERE attrelid = %s::regclass "
                "AND attname = 'embedding'", (TABLE,)).fetchone()[0]

    def count(self) -> int:
        with self.connect() as conn:
            return conn.execute(f"SELECT count(*) FROM {TABLE} WHERE strategy = %s",
                                (self.strategy,)).fetchone()[0]

    def all_chunks(self) -> list[dict[str, Any]]:
        """Toàn bộ chunk của strategy (dùng cho BM25/TF-IDF trên cùng corpus)."""
        with self.connect() as conn:
            rows = conn.execute(
                f"SELECT chunk_id, document_id, source, page, district, section, text, metadata "
                f"FROM {TABLE} WHERE strategy = %s ORDER BY page, chunk_id",
                (self.strategy,)).fetchall()
        return [_to_chunk(r) for r in rows]

    def search(self, query_vector: list[float], top_k: int = 5,
               filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        where, params = ["strategy = %s"], [self.strategy]
        for key, value in (filters or {}).items():
            if key not in FILTER_COLUMNS:
                raise ValueError(f"Không cho phép filter '{key}'; chỉ dùng {sorted(FILTER_COLUMNS)}")
            where.append(f"{key} = %s")
            params.append(value)
        qv = np.asarray(query_vector, dtype=np.float32)
        sql = (f"SELECT chunk_id, document_id, source, page, district, section, text, metadata, "
               f"1 - (embedding <=> %s) AS score FROM {TABLE} WHERE {' AND '.join(where)} "
               f"ORDER BY embedding <=> %s LIMIT %s")
        with self.connect() as conn:
            rows = conn.execute(sql, [qv, *params, qv, top_k]).fetchall()
        results = []
        for rank, r in enumerate(rows, 1):
            chunk = _to_chunk(r[:-1])
            chunk["score"] = round(float(r[-1]), 4)
            chunk["rank"] = rank
            results.append(chunk)
        return results


def _to_chunk(r) -> dict[str, Any]:
    chunk_id, document_id, source, page, district, section, text, metadata = r
    meta = dict(metadata or {})
    meta.setdefault("district", district)
    meta.setdefault("section", section)
    return {"id": chunk_id, "text": text, "source_id": document_id, "source": source,
            "page": page, "metadata": meta}


class DenseRetriever:
    """Dense retrieval trên vector store Day 6 — khớp Protocol Retriever của starter."""

    name = "dense"

    def __init__(self, store: Day6Store | None = None, embedder: Day6Embedder | None = None) -> None:
        self.store = store or Day6Store()
        self.embedder = embedder or Day6Embedder()
        db_dim = self.store.dimensions()
        if db_dim != self.embedder.dimensions:
            raise RuntimeError(f"Index có vector({db_dim}) nhưng model cho {self.embedder.dimensions} chiều")

    def search(self, query: str, *, filters: dict[str, Any] | None = None,
               top_k: int = 5) -> list[dict[str, Any]]:
        return self.store.search(self.embedder.embed_query(query), top_k=top_k, filters=filters)
