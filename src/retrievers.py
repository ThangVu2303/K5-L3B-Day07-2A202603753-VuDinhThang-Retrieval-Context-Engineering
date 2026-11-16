"""Các phương pháp retrieval Day 7 — cùng corpus (329 chunk strategy B của Day 6).

Mọi retriever có cùng giao diện (khớp Protocol `Retriever` của starter):
    search(query, *, filters=None, top_k=5) -> list[chunk + score + rank]

Retriever CHỈ nhận câu hỏi và filter do người hỏi cung cấp — không bao giờ nhận gold.

| Tên            | Ý tưởng                                                                |
|----------------|------------------------------------------------------------------------|
| dense          | Embedding MiniLM-L12 (Day 6) + cosine trên pgvector                    |
| bm25           | BM25Okapi trên token tiếng Anh (lowercase, bỏ stopword)                |
| tfidf          | TF-IDF của starter (tham chiếu, cùng họ lexical với BM25)              |
| hybrid         | Reciprocal Rank Fusion của dense + bm25                                |
| *_rerank       | Lấy N candidate rồi xếp lại bằng cross-encoder ms-marco-MiniLM-L-6-v2  |
"""
from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "starter"))
from baseline import TfidfIndex  # noqa: E402

# Stopword tiếng Anh tối thiểu (corpus Beige Book là tiếng Anh)
STOPWORDS = set("""a an and are as at be been but by did do does for from had has have how in into is it
its of on or over per such than that the their there these they this to was were what when where which
while who why will with within would""".split())


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold())
    return [t for t in tokens if t not in STOPWORDS]


def match_filters(chunk: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    """Filter metadata bằng so khớp chính xác (giống WHERE của pgvector)."""
    for key, value in (filters or {}).items():
        actual = chunk.get(key) if key in ("source_id", "page") else chunk["metadata"].get(key)
        if actual != value:
            return False
    return True


def _ranked(chunks_scores: list[tuple[dict, float]], top_k: int) -> list[dict[str, Any]]:
    ordered = sorted(chunks_scores, key=lambda cs: (-cs[1], cs[0]["id"]))[:top_k]
    return [{**c, "score": round(float(s), 4), "rank": i} for i, (c, s) in enumerate(ordered, 1)]


class BM25Retriever:
    name = "bm25"

    def __init__(self, chunks: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.tokens = [tokenize(c["text"]) for c in chunks]
        self.token_sets = [set(t) for t in self.tokens]
        self.index = BM25Okapi(self.tokens, k1=k1, b=b)

    def search(self, query: str, *, filters: dict[str, Any] | None = None,
               top_k: int = 5) -> list[dict[str, Any]]:
        q = tokenize(query)
        scores = self.index.get_scores(q)
        # Giữ chunk có ít nhất một từ của query (không dùng score > 0: BM25Okapi cho IDF = 0
        # với từ xuất hiện ở >= 50% chunk, sẽ loại nhầm chunk có từ khóa).
        qset = set(q)
        pairs = [(c, s) for c, s, ts in zip(self.chunks, scores, self.token_sets)
                 if qset & ts and match_filters(c, filters)]
        return _ranked(pairs, top_k)


class TfidfRetriever:
    """Bọc TF-IDF của starter, thêm filter metadata để so sánh công bằng."""

    name = "tfidf"

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.index = TfidfIndex(chunks)

    def search(self, query: str, *, filters: dict[str, Any] | None = None,
               top_k: int = 5) -> list[dict[str, Any]]:
        results = self.index.search(query, top_k=len(self.index.chunks))
        pairs = [(c, c["score"]) for c in results if match_filters(c, filters)]
        return [{k: v for k, v in r.items() if k != "strategy"} for r in _ranked(pairs, top_k)]


class HybridRRF:
    """Reciprocal Rank Fusion: score(d) = Σ 1 / (k + rank_i(d)).

    Dùng rank thay cho score nên không cần chuẩn hóa thang điểm cosine và BM25.
    """

    name = "hybrid"

    def __init__(self, retrievers: list, candidates: int = 20, k: int = 60) -> None:
        self.retrievers, self.candidates, self.k = retrievers, candidates, k

    def search(self, query: str, *, filters: dict[str, Any] | None = None,
               top_k: int = 5) -> list[dict[str, Any]]:
        fused: dict[str, float] = {}
        by_id: dict[str, dict] = {}
        for r in self.retrievers:
            for hit in r.search(query, filters=filters, top_k=self.candidates):
                fused[hit["id"]] = fused.get(hit["id"], 0.0) + 1.0 / (self.k + hit["rank"])
                by_id.setdefault(hit["id"], {k: v for k, v in hit.items() if k not in ("score", "rank")})
        return _ranked([(by_id[i], s) for i, s in fused.items()], top_k)


class CrossEncoderReranker:
    """Lấy `candidates` kết quả từ retriever gốc rồi xếp lại bằng cross-encoder."""

    MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, base, candidates: int = 20, model_name: str | None = None, model=None) -> None:
        self.base, self.candidates = base, candidates
        self.model_name = model_name or self.MODEL
        if model is None:
            from sentence_transformers import CrossEncoder  # import nặng
            model = CrossEncoder(self.model_name, device="cpu")
        self.model = model
        self.name = f"{base.name}_rerank"

    def search(self, query: str, *, filters: dict[str, Any] | None = None,
               top_k: int = 5) -> list[dict[str, Any]]:
        cands = self.base.search(query, filters=filters, top_k=self.candidates)
        if not cands:
            return []
        scores = self.model.predict([(query, c["text"]) for c in cands], show_progress_bar=False)
        pairs = [({k: v for k, v in c.items() if k not in ("score", "rank")}, s)
                 for c, s in zip(cands, scores)]
        return _ranked(pairs, top_k)
