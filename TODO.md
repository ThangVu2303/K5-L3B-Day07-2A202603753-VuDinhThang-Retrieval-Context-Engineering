# TODO — Day 7 Agentic RAG

> Checklist công việc tổng hợp từ README, GUIDE, BENCHMARK, RUBRIC, SUBMISSION, ADVANCED.
> Đánh dấu `[x]` khi hoàn thành. Điểm cơ bản tối đa 85, nâng cao +15.

---

## 0. Chuẩn bị môi trường

- [x] Kiểm tra Python 3.11+ (3.11.9, venv tại `.venv/`)
- [x] Chạy `python -m unittest discover -s tests -v` (15/15 pass; trên Windows cần `PYTHONUTF8=1`)
- [x] Tạo `requirements.txt` (openai, python-dotenv, tiktoken, numpy, rank-bm25) — còn thiếu thư viện vector store Day 6
- [x] Tạo `.env` (`PROVIDER`, `MODEL`, `OPENAI_API_KEY`) và `.env.example`; `.env` đã nằm trong `.gitignore`
- [x] Chọn LLM hỗ trợ tool calling cho agent: `openai/gpt-4o-mini` qua OpenRouter
- [x] Chạy `src/check_env.py` thành công (chat + tool calling)
- [ ] Đẩy repo lên GitHub cá nhân theo [SUBMISSION.md](SUBMISSION.md), không push vào repo đề

## 1. Dữ liệu & kết nối Day 6 — 13 điểm

### 1.1 Kết nối Day 6 (6 điểm)
- [x] Vector store Day 6: PostgreSQL 16 + pgvector 0.8.7 (Docker, `127.0.0.1:55433`, bảng `chunks`)
- [x] Embedding Day 6: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, 384 chiều, CPU, $0
- [x] Code kết nối: [src/day6_store.py](src/day6_store.py) (`DenseRetriever`), kiểm tra: [src/check_day6.py](src/check_day6.py) — kết quả khớp smoke test Day 6
- [x] Xuất chunk + benchmark: [src/export_day6.py](src/export_day6.py) → `data/day6/chunks.json` (329 chunk, strategy B) và `data/day6/questions.json` (15 dev / 5 test)
- [x] Evidence của cả 20 câu nằm trọn trong một chunk đúng nguồn/trang
- [x] TF-IDF baseline chạy trên dev: Hit@1 11/15, Hit@3 12/15
- [x] Ghi nguồn & quyền sử dụng tài liệu vào báo cáo (manifest sao chép sang `data/day6/manifest.json`)
- [x] Ghi chú báo cáo: strategy C Day 6 = chunk B + filter `district`; test Day 6 đã chạy một lần sau khi chốt

### 1.2 Benchmark (7 điểm) — tối thiểu 24 câu
- [x] **dev**: ≥ 15 câu có đáp án (q01–q15 từ Day 6)
- [x] **dev**: ≥ 2 câu thiếu bằng chứng (u01 Boston unemployment rate, u02 Atlanta diesel $/gallon)
- [x] **test**: ≥ 5 câu có đáp án (q16–q20 từ Day 6)
- [x] **test**: ≥ 2 câu thiếu bằng chứng (u03 Dallas số permit data center, u04 SF % giá nhà)
- [x] Câu có đáp án: `id, split, question, expected_answer, gold: [{source_id, page, evidence}]` → `data/day6/questions.json`
- [x] Evidence nằm trọn trong một chunk đúng nguồn/trang; ghi evidence thay thế nếu có
- [x] Câu thiếu bằng chứng: `answerable: false, expected_behavior: "search_and_abstain", absence_reason`, `hard_negatives` — soạn ở [data/unanswerable.json](data/unanswerable.json), `export_day6.py` tự kiểm tra regex 0 hit trên 329 chunk → `data/day6/cases.json` (24 câu)
- [ ] (Tùy chọn — bỏ qua trừ khi làm phương pháp multi-hop) câu multi-hop: ghi rõ nhóm evidence bắt buộc và cách chấm
- [x] Đảm bảo `gold` **không** được đưa vào agent/retrieval — `prepare_input` chỉ giữ `{id, question, filter}`; chấm ở `score_context` sau khi chạy; test `test_gold_never_reaches_llm_or_retriever`

## 2. Retrieval — 17 điểm

### 2.1 Ít nhất 3 phương pháp khác nhau (10 điểm)
- [x] **Phương pháp 1 — Dense retrieval** trên vector store Day 6 (dev: Hit@1 7/15, Hit@3 13/15 — khớp Day 6 C)
- [x] **Phương pháp 2 — BM25** (dev: 11/15, 14/15); TF-IDF starter chạy kèm làm tham chiếu
- [x] **Phương pháp 3 — Hybrid RRF** dense + BM25 (dev: 10/15, 14/15)
- [x] **Phương pháp 4 — Rerank** cross-encoder trên dense / hybrid (dev: hybrid_rerank 14/15, 15/15)
- [x] Code: [src/retrievers.py](src/retrievers.py), benchmark: [src/eval_retrieval.py](src/eval_retrieval.py) → `results/retrieval/dev/`; test: [tests/test_retrievers.py](tests/test_retrievers.py)

### 2.2 Cấu hình, nguồn, score, lỗi (7 điểm)
- [x] Mỗi kết quả trả về: `id, text, score, rank, source_id, source, page, metadata`
- [x] Ghi cấu hình từng phương pháp (top-k, filter, fusion, reranker...) — trong `summary.json` và báo cáo
- [x] Filter chỉ dùng `filter` do người hỏi nêu trong câu (district), không lấy từ gold
- [x] Lỗi retriever được ghi vào trace (`error`), không thay bằng kết quả giả; test bị khóa đến khi có `results/selection.json`

## 3. Agent & context — 17 điểm

### 3.1 Agent dùng model thật (9 điểm)
- [x] Agent riêng trong [src/agent.py](src/agent.py): GPT-4o-mini + tool calling gốc OpenAI (thay cho `Controller` của starter)
- [x] Retriever thật: hybrid + rerank trên pgvector Day 6 + BM25
- [x] Tool: `search(query, district?)` + `submit_answer(answer, citations, abstain)`
- [x] Điều kiện dừng: `submit_answer` hoặc `max_searches = 4` (buộc trả lời); thực tế dev: 1 lượt/câu
- [x] Trace mỗi câu (events, query, filter, kết quả, context, answer, citation, lỗi, token, chi phí, thời gian) → `results/agent/dev/hybrid_rerank-s4-k5-b4000/`
- [x] Chạy dev 17 câu: 0 lỗi, 15 answered + 2 abstain, Context Evidence Rate 15/15, $0.0049
- [x] Unit test agent với LLM giả: [tests/test_agent.py](tests/test_agent.py) (tổng 34 test pass)

### 3.2 Context, citation, thiếu evidence (8 điểm)
- [x] Chọn context: nhãn S#, dedup giữa các lượt search, giữ thứ tự rank
- [x] Context budget 4000 ký tự nội dung chunk (chunk vượt budget ghi `dropped_by_budget`)
- [x] Citation `[S#]` → đổi sang `[chunk-id]`; nhãn sai được sửa 1 lần rồi từ chối; tự gắn citation nếu thiếu trong câu
- [x] `submit_answer(abstain=true)` khi thiếu thông tin (dev: u01, u02 từ chối)
- [x] Tài liệu là dữ liệu: bọc `<documents>` + quy tắc trong system prompt
- [x] Filter do model đề xuất chỉ áp dụng khi District có trong câu hỏi (sửa lỗi q03)

## 4. Đánh giá — 17 điểm

### 4.1 Retrieval / context metrics (9 điểm)
- [x] Chạy 6 phương pháp × có/không filter trên **cùng** corpus, index, bộ câu hỏi dev (và test sau khi chốt)
- [x] Báo **Hit@1** (n/N và %)
- [x] Báo **Evidence Hit@3** (n/N và %)
- [x] Báo **Context Evidence Rate** (n/N và %) — retrieval và agent
- [x] Không đưa câu thiếu bằng chứng vào mẫu số retrieval metrics
- [x] Báo thời gian & chi phí từng cấu hình
- [x] Query rewrite / multi-hop: không áp dụng (query gốc); query agent tự viết được ghi trong trace
- [x] Chấm thủ công context 5 câu test có đáp án: chọn 0/0/5, baseline 1/0/4 → `results/agent/test/*/context_grades.json`

### 4.2 Câu trả lời, citation, abstention, tool use (8 điểm)
- [x] **Tool use** (dev): 17/17; 1 search/câu
- [x] **Grounded answer** (dev, chấm tay): 0/0/15, trung bình 2.00/2
- [x] **Citation hợp lệ** (dev, chấm tay): 17/17; ghi chú q12 thiếu citation cho ý "competitors"
- [x] **Abstention** (dev): 2/2 từ chối đúng; 0/15 từ chối nhầm
- [x] `reviews.json` + [starter/agent_metrics.py](starter/agent_metrics.py) → `results/agent/dev/hybrid_rerank-s4-k5-b4000/agent_metrics.json`
- [x] Nhãn dev đã được duyệt và chốt (nháp do Claude soạn, sinh viên chấp nhận toàn bộ)
- [x] Chạy test baseline (dense) + cấu hình chọn; chấm tay 5 câu test có đáp án + 2 câu thiếu evidence (chọn 2.00, baseline 1.60); context 0/1/2: chọn 0/0/5, baseline 1/0/4
- [ ] **Bạn duyệt nhãn** dev A/B/C và test (xem mục "Cần bạn duyệt" trong báo cáo Claude)

## 5. Thử nghiệm & cải thiện — 13 điểm

- [x] **Cải thiện 1**: rerank (dev Hit@1 7→14 dense, 10→14 hybrid)
- [x] **Cải thiện 2**: chốt district của agent (B không chốt: q03 từ chối nhầm; grounded 1.80 → 2.00)
- [x] **Cải thiện 3** (không cải thiện): prompt v2 (grounded 1.93, q12 xấu đi)
- [x] Phân tích **lỗi dev #1**: q03 chunking (retrieval) + q03 filter About (agent)
- [x] Phân tích **lỗi dev #2**: q05/q09 dense nhầm District; A-dense q03 hallucination, q04 sai số
- [x] Công bố tiêu chí chọn **trước**: `results/selection_criteria.json`
- [x] **Chốt cấu hình trên dev**: `results/selection.json` (hybrid_rerank, v1, có chốt)
- [x] Chạy **baseline (dense) + cấu hình đã chọn** trên **test** (một lần)
- [x] Phân tích đánh đổi chất lượng – thời gian – chi phí
- [x] Ghi cả các thử nghiệm không cải thiện (hybrid không rerank, prompt v2, Hit@1 test của hybrid_rerank)

## 6. Tái lập & báo cáo — 8 điểm

- [x] README mục "Cách chạy bài làm": cài đặt, Docker, export, retrieval, agent, tổng hợp nhãn, test
- [x] Ghi model LLM, embedding, vector store, cách tái tạo dữ liệu (README + báo cáo)
- [x] Kết quả/trace lưu ở `results/` (không bị gitignore) và được commit
- [x] Sao chép report → [2A202603753_VuDinhThang.md](report/individual/2A202603753_VuDinhThang.md)
- [x] Bảng so sánh retrieval (Baseline, PP2, PP3, Cải thiện 1–3, Chọn-test, Baseline-test)
- [x] Bảng đánh giá agent (tool use, grounded, citation, abstention) dev + test
- [x] Ví dụ trình bày: q17 có bằng chứng, u04 thiếu bằng chứng, q19 baseline lỗi
- [x] Kết luận, nguồn, giới hạn (mục cuối báo cáo)
- [ ] Kiểm tra không commit `.env`, API key, token, dữ liệu mật
- [ ] Cấp quyền xem repo cho người chấm, nộp link

---

## 7. (Tùy chọn) Thử thách nâng cao — +15 điểm

- [ ] Chọn tài liệu thật ≥ 100 trang có text, bảng, ảnh, biểu đồ; ghi nguồn, phiên bản, số trang, quyền sử dụng
- [ ] Pipeline xử lý: OCR / layout / trích bảng / mô tả ảnh / vision model / multimodal embedding
- [ ] Index toàn bộ nội dung hữu ích; giữ liên kết về trang gốc và vùng trang/ảnh crop
- [ ] Ghi các trang/phần không xử lý được và ảnh hưởng
- [ ] Kiểm tra nội dung trích xuất với bản gốc
- [ ] Benchmark **dev**: ≥ 15 câu có đáp án + 3 câu thiếu bằng chứng
- [ ] Benchmark **test** ≥ 30 câu: Bảng ≥ 6, Ảnh/sơ đồ ≥ 5, Biểu đồ ≥ 5, Multi-hop ≥ 4, Text ≥ 4, Thiếu bằng chứng ≥ 6
- [ ] Mỗi câu: ID, nhóm chính, câu hỏi, đáp án / hành vi từ chối, nguồn, evidence (trang, ô/hàng/cột, vùng hình), đơn vị, phép tính
- [ ] Chốt pipeline trên dev, chạy toàn bộ test, lưu trace
- [ ] Chấm thủ công từng câu đạt/không đạt kèm lý do
- [ ] Đạt **≥ 90%** (≥ 27/30), báo n/N toàn bộ và theo nhóm, lỗi, thời gian, chi phí

---

## Lịch gợi ý (4 giờ)

| Mốc | Việc | Mục liên quan |
|---|---|---|
| 0:00–1:00 | Kết nối Day 6, chuẩn bị benchmark | 0, 1 |
| 1:00–2:00 | Agent + retrieval đầu tiên | 2, 3 |
| 2:00–3:30 | ≥ 3 phương pháp, thử cấu hình dev, phân tích lỗi | 2, 4, 5 |
| 3:30–4:00 | Chạy test, tổng hợp, báo cáo | 5, 6 |
