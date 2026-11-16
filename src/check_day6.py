"""Kiểm tra kết nối Day 6: database, số chunk, số chiều, embedding và một truy vấn mẫu.

Chạy:  .venv\\Scripts\\python src/check_day6.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from day6_store import Day6Embedder, Day6Store, DenseRetriever  # noqa: E402

store = Day6Store()
try:
    n, dim = store.count(), store.dimensions()
except Exception as exc:
    sys.exit(f"Không kết nối được database: {type(exc).__name__}: {exc}\n"
             "→ Bật Docker Desktop và chạy `docker compose up -d` trong thư mục starter của Day 6.")
print(f"[db]    strategy={store.strategy}  chunks={n}  vector({dim})")
if n == 0:
    sys.exit("Strategy không có chunk nào — kiểm tra DAY6_STRATEGY hoặc chạy lại ingest Day 6.")

t = time.perf_counter()
embedder = Day6Embedder()
print(f"[embed] {embedder.model_name}  dim={embedder.dimensions}  load={time.perf_counter() - t:.1f}s")

retriever = DenseRetriever(store, embedder)
for query, filters in [("How did Canadian tourists affect tourism?", None),
                       ("How did Canadian tourists affect tourism?", {"district": "Boston"})]:
    t = time.perf_counter()
    hits = retriever.search(query, filters=filters, top_k=3)
    ms = (time.perf_counter() - t) * 1000
    print(f"\n[search] {query!r} filters={filters}  ({ms:.0f} ms)")
    for h in hits:
        print(f"  #{h['rank']} score={h['score']:.4f} p{h['page']} "
              f"{h['metadata'].get('district')} | {h['text'][:90]!r}")
print("\nKết nối Day 6 OK.")
