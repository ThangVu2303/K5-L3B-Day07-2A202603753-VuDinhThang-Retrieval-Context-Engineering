"""Chạy agent trên một split và lưu trace, tổng hợp tự động và file để chấm tay.

Chạy:
  .venv\\Scripts\\python src/run_agent.py --split dev
  .venv\\Scripts\\python src/run_agent.py --split dev --retriever dense --name dense-agent
  .venv\\Scripts\\python src/run_agent.py --split dev --ids q03 u01      # chạy vài câu

Đầu ra: results/agent/<split>/<name>/
  traces.json         trace từng câu (định dạng tương thích starter/agent_metrics.py)
  summary.json        số liệu tự động: status, tool use, số lượt search, Context Evidence Rate,
                      token, chi phí, thời gian (KHÔNG phải điểm grounded — phần đó chấm tay)
  review.md           bảng đọc nhanh để chấm tay
  reviews_template.json  khung nhãn (answer_grade / citation_support / abstention_correct = null)

Gold chỉ được đọc ở bước chấm sau khi agent chạy xong (hàm `score_context`).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from agent import Agent, AgentConfig  # noqa: E402
from llm import get_client  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "day6"


def build_retriever(name: str, chunks: list[dict]):
    from day6_store import DenseRetriever
    from retrievers import BM25Retriever, CrossEncoderReranker, HybridRRF

    if name == "bm25":
        return BM25Retriever(chunks)
    dense = DenseRetriever()
    if name == "dense":
        return dense
    if name == "dense_rerank":
        return CrossEncoderReranker(dense, candidates=20)
    hybrid = HybridRRF([dense, BM25Retriever(chunks)], candidates=20, k=60)
    if name == "hybrid":
        return hybrid
    if name == "hybrid_rerank":
        return CrossEncoderReranker(hybrid, candidates=20)
    raise ValueError(f"retriever không hỗ trợ: {name}")


def score_context(trace: dict, case: dict, chunks_by_id: dict) -> None:
    """Chấm SAU khi chạy: context cuối có chứa evidence gold không (chỉ câu có đáp án)."""
    if not case.get("answerable", True) or trace["status"] == "error":
        trace["context_hit"] = None
        return
    def hit(cid):
        c = chunks_by_id[cid]
        return any(c["source_id"] == g["source_id"] and c["page"] == g["page"] and g["evidence"] in c["text"]
                   for g in case["gold"])
    trace["context_hit"] = any(hit(cid) for cid in trace["context_ids"])
    trace["cited_gold"] = any(hit(cid) for cid in trace["citations"])


def summarize(traces: list[dict], cases: dict) -> dict:
    ok = [t for t in traces if t["status"] != "error"]
    ans = [t for t in ok if cases[t["query_id"]].get("answerable", True)]
    una = [t for t in ok if not cases[t["query_id"]].get("answerable", True)]
    frac = lambda n, N: {"n": n, "N": N, "rate": round(n / N, 3) if N else None}  # noqa: E731
    return {
        "cases": len(traces), "errors": len(traces) - len(ok),
        "status": dict(Counter(t["status"] for t in traces)),
        "tool_use": frac(sum(t["num_searches"] > 0 for t in ok), len(ok)),
        "searches_distribution": dict(sorted(Counter(t["num_searches"] for t in ok).items())),
        "context_evidence_rate": frac(sum(bool(t["context_hit"]) for t in ans), len(ans)),
        "cited_gold_chunk (tự động, chưa phải citation hợp lệ)": frac(sum(bool(t.get("cited_gold")) for t in ans), len(ans)),
        "answered_with_citation": frac(sum(bool(t["citations"]) for t in ok if t["status"] == "answered"),
                                       sum(t["status"] == "answered" for t in ok)),
        "abstain_on_unanswerable (chờ xác nhận tay)": frac(sum(t["status"] == "abstain" for t in una), len(una)),
        "false_abstain_on_answerable": frac(sum(t["status"] == "abstain" for t in ans), len(ans)),
        "tokens": {"input": sum(t["tokens"]["input"] for t in traces),
                   "output": sum(t["tokens"]["output"] for t in traces)},
        "cost_usd_total": round(sum(t["cost_usd"] for t in traces), 5),
        "median_seconds_per_question": round(statistics.median(t["elapsed_ms"] for t in traces) / 1000, 2),
        "avg_context_chars": round(statistics.mean(t["context_chars"] for t in ok)) if ok else None,
    }


def write_review(path: Path, traces: list[dict], cases: dict) -> None:
    lines = ["# Bảng chấm tay\n",
             "Đọc câu trả lời, mở chunk được trích trong `traces.json` (`context`) và đối chiếu bản gốc.\n",
             "| ID | Loại | Status | #search | Câu trả lời | Citation (trang) | Context có gold | Đáp án kỳ vọng |",
             "|---|---|---|---|---|---|---|---|"]
    for t in traces:
        c = cases[t["query_id"]]
        kind = "có đáp án" if c.get("answerable", True) else "thiếu bằng chứng"
        pages = ", ".join(f"p{cid.split(':p')[1][:3].lstrip('0')}" for cid in t["citations"]) or "—"
        expected = c.get("expected_answer") or f"Từ chối — {c.get('absence_reason', '')}"
        cell = lambda s: str(s).replace("|", "\\|").replace("\n", " ")  # noqa: E731
        lines.append(f"| {t['query_id']} | {kind} | {t['status']} | {t['num_searches']} | "
                     f"{cell(t['answer_labels'] or t['error'])} | {pages} | {t.get('context_hit')} | {cell(expected)} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--retriever", default="hybrid_rerank",
                    choices=["dense", "bm25", "hybrid", "dense_rerank", "hybrid_rerank"])
    ap.add_argument("--max-searches", type=int, default=4)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--budget", type=int, default=4000, help="ký tự nội dung chunk trong context")
    ap.add_argument("--prompt", choices=["v1", "v2"], default="v1")
    ap.add_argument("--no-district-guard", action="store_true",
                    help="thử nghiệm B: áp dụng mọi district model đề xuất")
    ap.add_argument("--name", default=None, help="tên thư mục kết quả")
    ap.add_argument("--ids", nargs="*", help="chỉ chạy các ID này")
    args = ap.parse_args()
    if args.split == "test" and not (ROOT / "results" / "selection.json").exists():
        sys.exit("Test bị khóa: chốt cấu hình trên dev và ghi results/selection.json trước.")

    chunks = json.loads((DATA / "chunks.json").read_text(encoding="utf-8"))
    chunks_by_id = {c["id"]: c for c in chunks}
    all_cases = [c for c in json.loads((DATA / "cases.json").read_text(encoding="utf-8"))
                 if c["split"] == args.split and (not args.ids or c["id"] in args.ids)]
    cases = {c["id"]: c for c in all_cases}
    districts = sorted({c["metadata"]["district"] for c in chunks if c["metadata"].get("district")})

    client, llm_cfg = get_client()
    t = time.perf_counter()
    retriever = build_retriever(args.retriever, chunks)
    print(f"Model {llm_cfg.model} ({llm_cfg.provider}) | retriever {args.retriever} | "
          f"nạp {time.perf_counter() - t:.1f}s")
    cfg = AgentConfig(model=llm_cfg.model, max_searches=args.max_searches, top_k=args.top_k,
                      context_budget=args.budget, retriever_name=args.retriever,
                      prompt_version=args.prompt, district_guard=not args.no_district_guard)
    agent = Agent(client, retriever, cfg, districts)

    traces = []
    for case in all_cases:
        tr = agent.run(case)          # agent chỉ thấy {id, question, filter}
        tr["split"] = args.split
        score_context(tr, case, chunks_by_id)   # chấm sau khi chạy
        traces.append(tr)
        print(f"{tr['query_id']:<4} {tr['status']:<9} search={tr['num_searches']} "
              f"ctx_gold={str(tr['context_hit']):<5} ${tr['cost_usd']:.5f} {tr['elapsed_ms'] / 1000:.1f}s | "
              f"{(tr['answer_labels'] or tr['error'] or '')[:90]}")

    name = args.name or f"{args.retriever}-s{args.max_searches}-k{args.top_k}-b{args.budget}"
    out = ROOT / "results" / "agent" / args.split / name
    out.mkdir(parents=True, exist_ok=True)
    summary = {"run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "split": args.split,
               "model": llm_cfg.model, "provider": llm_cfg.provider, "config": cfg.__dict__,
               **summarize(traces, cases)}
    (out / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    write_review(out / "review.md", traces, cases)
    template = [{"query_id": t["query_id"], "citation_support": [None] * len(t["citations"]),
                 **({"answer_grade": None} if cases[t["query_id"]].get("answerable", True)
                    else {"abstention_correct": None}), "reason": ""}
                for t in traces if t["status"] != "error"]
    (out / "reviews_template.json").write_text(json.dumps(template, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n" + json.dumps({k: v for k, v in summary.items() if k not in ("config",)},
                            ensure_ascii=False, indent=1))
    print(f"→ {out}")


if __name__ == "__main__":
    main()
