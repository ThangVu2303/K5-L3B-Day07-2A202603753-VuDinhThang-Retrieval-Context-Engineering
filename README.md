# Day 7 — Agentic RAG

> ## Bài làm: Vũ Đình Thắng · 2A202603753
>
> **Báo cáo:** [report/individual/2A202603753_VuDinhThang.md](report/individual/2A202603753_VuDinhThang.md) · **Checklist:** [TODO.md](TODO.md)
>
> **Kết quả chính.** Corpus: Beige Book 10/2026 (Fed), 329 chunk Day 6, pgvector + MiniLM-L12. Agent: GPT-4o-mini với tool `search` (hybrid RRF + cross-encoder rerank).
>
> | | Dev (15 + 2) | Test (5 + 2) |
> |---|---|---|
> | Retrieval Evidence Hit@3, cấu hình chọn / dense baseline | 15/15 / 13/15 | 5/5 / 3/5 |
> | Agent grounded (chấm tay), cấu hình chọn / dense baseline | 2.00 / 1.67 | 2.00 / 1.60 |
> | Từ chối đúng câu thiếu bằng chứng (cấu hình chọn) | 2/2 | 2/2 |
>
> Chi tiết cách chạy ở mục [Cách chạy bài làm](#cách-chạy-bài-làm) bên dưới; phần sau mục đó là đề bài gốc.

## Cách chạy bài làm

**Yêu cầu:** Windows/macOS/Linux, Python 3.11+, Docker Desktop, API key OpenAI **hoặc** OpenRouter (model hỗ trợ tool calling). Dữ liệu và database đến từ repo Day 6 (`K5-L3-Day06-...-DataIngestionEngineering`), đặt cạnh repo này.

### 1. Cài đặt
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   ·   macOS/Linux: source .venv/bin/activate
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
cp .env.example .env      # Windows PowerShell: Copy-Item .env.example .env
```
Điền `.env` (**không commit file này**):
```
PROVIDER=openrouter            # hoặc openai
MODEL=openai/gpt-4o-mini       # với PROVIDER=openai: gpt-4o-mini
OPENAI_API_KEY=<key>
DATABASE_URL=postgresql://day6:day6_local_only@127.0.0.1:55433/day6_lab
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
DAY6_STRATEGY=B                # Day 6 strategy C = chunk B + filter district
```
Trên Windows nên đặt `PYTHONUTF8=1` (`setx PYTHONUTF8 1`); một test của starter đọc file không chỉ định encoding.

### 2. Bật vector store Day 6 và kiểm tra
```bash
cd ../K5-L3-Day06-VuDinhThang-2A202603753-DataIngestionEngineering/starter
docker compose up -d                 # PostgreSQL 16 + pgvector, volume giữ dữ liệu Day 6
cd -
python src/check_env.py              # gọi thử LLM: chat + tool calling
python src/check_day6.py             # 329 chunk, vector(384), truy vấn mẫu
```
Nếu volume Day 6 đã bị xóa: chạy lại `python scripts/run_ingest.py` trong `starter/` của Day 6 (embedding local, $0).

### 3. Dữ liệu và benchmark
```bash
python src/export_day6.py
```
Lệnh này tạo `data/day6/chunks.json` (329 chunk), `questions.json` (20 câu có đáp án) và `cases.json` (24 câu, gồm 4 câu thiếu bằng chứng từ [data/unanswerable.json](data/unanswerable.json)). Script tự kiểm tra evidence nằm trọn trong chunk, và các regex kiểm tra vắng mặt của câu thiếu bằng chứng đều 0 hit. Các file trong `data/day6/` đã được commit sẵn (tài liệu public domain).

### 4. Benchmark retrieval (6 phương pháp × có/không filter)
```bash
python src/eval_retrieval.py --split dev     # → results/retrieval/dev/{summary.json,traces.jsonl}
python src/eval_retrieval.py --split test    # chỉ chạy được khi có results/selection.json
```

### 5. Agent
```bash
python src/run_agent.py --split dev                                   # cấu hình chọn: hybrid_rerank, prompt v1, có chốt district
python src/run_agent.py --split dev --retriever dense --name A-dense-v1         # baseline
python src/run_agent.py --split dev --no-district-guard --name B-hybrid_rerank-noguard-v1
python src/run_agent.py --split dev --prompt v2 --name C-hybrid_rerank-v2
python src/run_agent.py --split test                                  # sau khi chốt
python src/run_agent.py --split test --retriever dense --name A-dense-v1
```
Mỗi lần chạy ghi vào `results/agent/<split>/<name>/`: `traces.json`, `summary.json`, `review.md`, `reviews_template.json`. Chi phí khoảng $0.005 cho 17 câu dev. LLM ở temperature 0 nhưng qua API nên câu chữ có thể khác nhẹ giữa các lần chạy; nhãn chấm tay ứng với trace đã commit.

### 6. Tổng hợp nhãn chấm tay
```bash
python starter/agent_metrics.py --cases data/day6/cases.json \
  --traces  results/agent/test/hybrid_rerank-s4-k5-b4000/traces.json \
  --reviews results/agent/test/hybrid_rerank-s4-k5-b4000/reviews.json \
  --split test --output results/agent/test/hybrid_rerank-s4-k5-b4000/agent_metrics.json
```

### 7. Unit test
```bash
python -m unittest discover -s tests -v      # 36 test, mock LLM/retriever, không gọi API
```

### Cấu trúc bài làm
| Đường dẫn | Nội dung |
|---|---|
| [src/llm.py](src/llm.py) | Cấu hình LLM (`PROVIDER`, `MODEL`, `OPENAI_API_KEY`) |
| [src/day6_store.py](src/day6_store.py) | Kết nối pgvector + embedding Day 6, `DenseRetriever` |
| [src/retrievers.py](src/retrievers.py) | BM25, TF-IDF, Hybrid RRF, Cross-encoder rerank |
| [src/eval_retrieval.py](src/eval_retrieval.py) | Benchmark retrieval: Hit@1/3/10, MRR, Context Evidence Rate |
| [src/agent.py](src/agent.py) | Agent tool calling: `search`, `submit_answer`, citation, context budget, chốt district |
| [src/run_agent.py](src/run_agent.py) | Chạy agent, trace, chấm context sau khi chạy |
| [src/export_day6.py](src/export_day6.py) | Xuất chunk + benchmark, kiểm tra evidence |
| `data/day6/`, [data/unanswerable.json](data/unanswerable.json) | Corpus, benchmark 24 câu, manifest nguồn |
| `results/` | Trace, summary, nhãn chấm tay (`reviews.json`, `context_grades.json`), [selection_criteria.json](results/selection_criteria.json), [selection.json](results/selection.json) |
| `tests/` | Test starter + `test_retrievers.py`, `test_agent.py` (gồm test chống rò rỉ gold) |

---

# Đề bài

**Bài cá nhân · Python 3.11+ · tiếp nối Day 6**

Dùng embedding và vector store từ Day 6 để xây một agent biết tìm bằng chứng, trả lời có nguồn và nhận biết khi không đủ thông tin. Bạn tự chọn kiến trúc, số tool và cách tổ chức retrieval.

> **Tùy chọn:** [Gợi ý dùng AI để lập trình](AGENT_SETUP.md) giúp bạn làm lab; phần này không bắt buộc và không tính điểm.

## Bài cần làm

![Agent tìm tài liệu trong kho Day 6 và trả lời có nguồn](assets/retrieval-pipeline.svg)

1. Kết nối dữ liệu, embedding và vector store Day 6.
2. Xây agent dùng model thật và ít nhất một khả năng truy xuất dữ liệu. Agent có thể dùng nhiều tool và nhiều lượt gọi.
3. Thử **ít nhất 3 phương pháp retrieval khác nhau**, trong đó có dense retrieval dùng vector store Day 6. Tự chọn các phương pháp còn lại; xem [gợi ý](GUIDE.md#2-thử-các-phương-pháp-retrieval).
4. Thử nhiều cấu hình, benchmark, phân tích lỗi và chọn cấu hình tốt trên **dev**; sau đó đánh giá trên **test**.

Benchmark tối thiểu có **24 câu**: dev gồm 15 câu có đáp án và 2 câu thiếu bằng chứng; test gồm 5 câu có đáp án và 2 câu thiếu bằng chứng. Xem [benchmark](BENCHMARK.md) và [rubric](RUBRIC.md).

## Thử thách nâng cao · +15 điểm

Index một tài liệu thực tế từ **100 trang** có bảng, ảnh và biểu đồ; xây agent truy xuất và trả lời các câu hỏi về nội dung đó. Hoàn thành [thử thách nâng cao](ADVANCED.md) và đạt **ít nhất 90% benchmark test** để nhận 15 điểm bonus. Phần cơ bản tối đa 85 điểm; toàn bài tối đa 100 điểm.

## Bắt đầu

1. Đọc [hướng dẫn](GUIDE.md), [rubric](RUBRIC.md) và [quy tắc](RULES.md).
2. Tạo repo cá nhân theo [cách nộp bài](SUBMISSION.md), chuẩn bị đầu ra Day 6 và model hỗ trợ tool calling.
3. Dùng hoặc thay thế [starter](starter/README.md); bạn cần kết nối model và vector store thật.
4. Chạy `python -m unittest discover -s tests -v`. Test mock chỉ kiểm tra code, không phải benchmark agent thật.

![Các bước thử nghiệm trên dev và đánh giá trên test](assets/experiment-loop.svg)

Nộp code, cách chạy, cấu hình, benchmark, trace và [báo cáo](report/individual/MSSV_HoTen.md). Thời gian ở [checkpoints](CHECKPOINTS.md) là gợi ý, có thể điều chỉnh theo lớp và phạm vi thử nghiệm.
