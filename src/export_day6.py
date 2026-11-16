"""Xuất chunk từ vector store Day 6 và chuyển benchmark Day 6 sang định dạng Day 7.

Đầu ra (data/day6/):
  chunks.json     toàn bộ chunk của DAY6_STRATEGY (cùng corpus cho dense, BM25, TF-IDF)
  questions.json  20 câu CÓ đáp án (Day 6, gold = danh sách evidence thay thế) — dùng cho retrieval metrics
  cases.json      toàn bộ 24 câu (20 có đáp án + 4 thiếu bằng chứng từ data/unanswerable.json) — dùng cho agent

Chạy:  .venv\\Scripts\\python src/export_day6.py --day6 <thư mục Day 6>
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from day6_store import Day6Store  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DAY6 = ROOT.parent / "K5-L3-Day06-VuDinhThang-2A202603753-DataIngestionEngineering"


def convert_question(q: dict) -> dict:
    gold = [{"source_id": q["gold_source_id"], "page": q["gold_page"], "evidence": q["evidence_quote"]}]
    gold += [{"source_id": a["source_id"], "page": a["page"], "evidence": a["evidence_quote"]}
             for a in q.get("alt_gold", [])]
    out = {"id": q["id"], "split": q["split"], "question": q["question"],
           "expected_answer": q["expected_answer"], "gold": gold,
           "type": q.get("type"), "grounding_requirements": q.get("grounding_requirements", [])}
    if q.get("filter"):
        out["filter"] = q["filter"]  # do người hỏi nêu trong câu hỏi, không lấy từ gold
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--day6", type=Path, default=DEFAULT_DAY6)
    ap.add_argument("--output", type=Path, default=ROOT / "data" / "day6")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    chunks = Day6Store().all_chunks()
    by_key: dict[tuple, list[str]] = {}
    for c in chunks:
        by_key.setdefault((c["source_id"], c["page"]), []).append(c["text"])

    qpath = args.day6 / "benchmark" / "questions.jsonl"
    questions = [convert_question(json.loads(line))
                 for line in qpath.read_text(encoding="utf-8").splitlines() if line.strip()]

    # Kiểm tra: mỗi câu phải có ít nhất một evidence nằm trọn trong một chunk đúng nguồn/trang
    problems = []
    for q in questions:
        ok = [g for g in q["gold"]
              if any(g["evidence"] in t for t in by_key.get((g["source_id"], g["page"]), []))]
        if not ok:
            problems.append(q["id"])

    # Câu thiếu bằng chứng (soạn tay): kiểm tra lại trên toàn bộ chunk, mọi pattern phải 0 hit
    upath = ROOT / "data" / "unanswerable.json"
    unanswerable = json.loads(upath.read_text(encoding="utf-8")) if upath.exists() else []
    for u in unanswerable:
        for check in u["absence_check"]:
            hits = [c for c in chunks if re.search(check["pattern"], c["text"], re.I)
                    and check["district"] in (None, c["metadata"].get("district"))]
            if hits:
                problems.append(f"{u['id']}: /{check['pattern']}/ có {len(hits)} hit "
                                f"(trang {sorted({h['page'] for h in hits})})")
    for q in questions:
        q["answerable"] = True
    cases = questions + [{k: v for k, v in u.items() if k != "absence_check"} for u in unanswerable]

    (args.output / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=1), encoding="utf-8")
    (args.output / "questions.json").write_text(json.dumps(questions, ensure_ascii=False, indent=1), encoding="utf-8")
    (args.output / "cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"chunks: {len(chunks)} → {args.output / 'chunks.json'}")
    print(f"questions: {len(questions)} câu có đáp án → {args.output / 'questions.json'}")
    print(f"cases:     {len(cases)} câu → {args.output / 'cases.json'}")
    for s in ("dev", "test"):
        ans = sum(q["split"] == s and q["answerable"] for q in cases)
        un = sum(q["split"] == s and not q["answerable"] for q in cases)
        print(f"  {s}: {ans} có đáp án + {un} thiếu bằng chứng")
    if problems:
        print("CẢNH BÁO — cần xem lại:")
        for p in problems:
            print(f"  - {p}")
    else:
        print("OK: evidence của mọi câu có đáp án nằm trọn trong một chunk đúng nguồn/trang; "
              "mọi kiểm tra vắng mặt của câu thiếu bằng chứng đều 0 hit.")


if __name__ == "__main__":
    main()
