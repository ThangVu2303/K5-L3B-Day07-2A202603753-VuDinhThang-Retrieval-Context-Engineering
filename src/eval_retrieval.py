"""Benchmark retrieval trên cùng corpus, cùng bộ câu hỏi và cùng định nghĩa hit.

Metrics (mẫu số = số câu CÓ đáp án của split; câu thiếu bằng chứng không tính):
  Hit@1, Evidence Hit@3, Hit@10, MRR@10
  Context Evidence Rate: context = top-CONTEXT_K kết quả, loại trùng, xếp theo rank,
      cắt theo budget ký tự (gồm cả header nguồn); đạt khi context chứa chunk có evidence gold.
  Hit: chunk đúng source_id + page và chứa nguyên văn evidence (một trong các gold thay thế).

Gold CHỈ dùng sau khi retriever trả kết quả, để chấm. Retriever nhận (question, filter).
Filter = `filter` do người hỏi nêu trong câu hỏi (có sẵn trước khi truy xuất), không lấy từ gold.

Chạy:  .venv\\Scripts\\python src/eval_retrieval.py --split dev
Test bị khóa cho tới khi có results/selection.json (chốt cấu hình trên dev).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from day6_store import DenseRetriever  # noqa: E402
from retrievers import BM25Retriever, CrossEncoderReranker, HybridRRF, TfidfRetriever  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "day6"
CONTEXT_K, BUDGET_CHARS, EVAL_K = 5, 2000, 10


def is_hit(chunk: dict, gold: list[dict]) -> bool:
    return any(chunk["source_id"] == g["source_id"] and chunk["page"] == g["page"]
               and g["evidence"] in chunk["text"] for g in gold)


def pack_context(results: list[dict], k: int = CONTEXT_K, budget: int = BUDGET_CHARS):
    blocks, ids, seen = [], [], set()
    for c in results[:k]:
        if c["id"] in seen:
            continue
        seen.add(c["id"])
        block = f"[{c['id']} | {c['source']} | page {c['page']}]\n{c['text']}"
        if len("\n\n".join(blocks + [block])) <= budget:
            blocks.append(block)
            ids.append(c["id"])
    return "\n\n".join(blocks), ids


def build_retrievers(chunks: list[dict]) -> dict:
    t = time.perf_counter()
    dense = DenseRetriever()
    bm25 = BM25Retriever(chunks)
    tfidf = TfidfRetriever(chunks)
    hybrid = HybridRRF([dense, bm25], candidates=20, k=60)
    dense_rr = CrossEncoderReranker(dense, candidates=20)
    hybrid_rr = CrossEncoderReranker(hybrid, candidates=20, model=dense_rr.model)
    print(f"Nạp model/index: {time.perf_counter() - t:.1f}s")
    return {"tfidf": tfidf, "bm25": bm25, "dense": dense, "hybrid": hybrid,
            "dense_rerank": dense_rr, "hybrid_rerank": hybrid_rr}


def run_config(name: str, retriever, use_filter: bool, questions: list[dict]) -> tuple[dict, list]:
    traces, times = [], []
    for q in questions:
        filters = q.get("filter") if use_filter else None
        t = time.perf_counter()
        try:
            results, error = retriever.search(q["question"], filters=filters, top_k=EVAL_K), None
        except Exception as exc:  # ghi lỗi thật, không thay bằng kết quả giả
            results, error = [], f"{type(exc).__name__}: {exc}"
        ms = (time.perf_counter() - t) * 1000
        times.append(ms)
        context, ctx_ids = pack_context(results)
        # --- chấm điểm (sau khi đã truy xuất) ---
        hits = [is_hit(c, q["gold"]) for c in results]
        first = next((i for i, h in enumerate(hits, 1) if h), None)
        traces.append({
            "config": name, "query_id": q["id"], "split": q["split"], "question": q["question"],
            "filter": filters, "elapsed_ms": round(ms, 1), "error": error,
            "results": [{"rank": c["rank"], "id": c["id"], "page": c["page"],
                         "district": c["metadata"].get("district"), "score": c["score"], "hit": h}
                        for c, h in zip(results, hits)],
            "context_ids": ctx_ids, "context_chars": len(context),
            "hit1": bool(hits[:1] and hits[0]), "hit3": any(hits[:3]), "hit10": any(hits[:10]),
            "rr": round(1 / first, 4) if first else 0.0,
            "context_hit": any(h and c["id"] in ctx_ids for c, h in zip(results, hits)),
            "gold_pages": sorted({g["page"] for g in q["gold"]}),
        })
    n = len(questions)
    summary = {
        "config": name, "filter": use_filter, "n": n,
        "hit1": sum(t["hit1"] for t in traces), "hit3": sum(t["hit3"] for t in traces),
        "hit10": sum(t["hit10"] for t in traces),
        "mrr10": round(sum(t["rr"] for t in traces) / n, 3),
        "context_hit": sum(t["context_hit"] for t in traces),
        "avg_context_chars": round(statistics.mean(t["context_chars"] for t in traces)),
        "median_ms": round(statistics.median(times), 1), "errors": sum(bool(t["error"]) for t in traces),
        "cost_usd": 0.0,
    }
    return summary, traces


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--configs", nargs="*", help="giới hạn cấu hình, ví dụ dense+filter bm25+filter")
    ap.add_argument("--output", type=Path, default=ROOT / "results" / "retrieval")
    args = ap.parse_args()
    if args.split == "test" and not (ROOT / "results" / "selection.json").exists():
        sys.exit("Test bị khóa: chốt cấu hình trên dev và ghi results/selection.json trước.")

    chunks = json.loads((DATA / "chunks.json").read_text(encoding="utf-8"))
    questions = [q for q in json.loads((DATA / "questions.json").read_text(encoding="utf-8"))
                 if q["split"] == args.split]
    retrievers = build_retrievers(chunks)

    configs = [(f"{m}{'+filter' if f else ''}", m, f) for m in retrievers for f in (False, True)]
    if args.configs:
        configs = [c for c in configs if c[0] in args.configs]

    summaries, all_traces = [], []
    for label, method, use_filter in configs:
        s, tr = run_config(label, retrievers[method], use_filter, questions)
        summaries.append(s)
        all_traces += tr
        print(f"{label:<22} Hit@1 {s['hit1']:>2}/{s['n']}  Hit@3 {s['hit3']:>2}/{s['n']}  "
              f"Hit@10 {s['hit10']:>2}/{s['n']}  MRR {s['mrr10']:.3f}  "
              f"Ctx {s['context_hit']:>2}/{s['n']}  {s['median_ms']:>6.1f} ms  err {s['errors']}")

    out = args.output / args.split
    out.mkdir(parents=True, exist_ok=True)
    meta = {"run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "split": args.split,
            "corpus": "data/day6/chunks.json (329 chunks, Day 6 strategy B)",
            "embedding": retrievers["dense"].embedder.model_name,
            "reranker": retrievers["dense_rerank"].model_name,
            "context": {"top_k": CONTEXT_K, "budget_chars": BUDGET_CHARS},
            "eval_k": EVAL_K, "hybrid": {"rrf_k": 60, "candidates": 20},
            "rerank_candidates": 20, "configs": summaries}
    (out / "summary.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "traces.jsonl").write_text("\n".join(json.dumps(t, ensure_ascii=False) for t in all_traces),
                                      encoding="utf-8")
    print(f"\n→ {out / 'summary.json'}\n→ {out / 'traces.jsonl'}")


if __name__ == "__main__":
    main()
