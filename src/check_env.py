"""Kiểm tra môi trường: đọc cấu hình và gọi thử LLM (chat + tool calling).

Chạy:  .venv\\Scripts\\python src/check_env.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from llm import get_client  # noqa: E402

try:
    client, cfg = get_client()
except ValueError as exc:
    sys.exit(f"Lỗi cấu hình: {exc}")

print(f"Python:   {sys.version.split()[0]}")
print(f"PROVIDER: {cfg.provider}")
print(f"MODEL:    {cfg.model}")
print(f"BASE_URL: {cfg.base_url}")

# 1) Chat đơn giản
try:
    resp = client.chat.completions.create(
        model=cfg.model,
        messages=[{"role": "user", "content": "Trả lời đúng một từ: OK"}],
        max_tokens=5,
    )
except Exception as exc:  # báo lỗi thật, không giả kết quả
    sys.exit(f"Gọi LLM thất bại: {type(exc).__name__}: {exc}")
print(f"[chat] phản hồi={resp.choices[0].message.content!r}, token={resp.usage.total_tokens}")

# 2) Tool calling (agent cần khả năng này)
tools = [{
    "type": "function",
    "function": {
        "name": "search",
        "description": "Tìm tài liệu liên quan tới câu hỏi.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}]
try:
    resp = client.chat.completions.create(
        model=cfg.model,
        messages=[{"role": "user", "content": "Hãy tìm tài liệu về chính sách hoàn tiền."}],
        tools=tools,
        tool_choice="required",
    )
except Exception as exc:
    sys.exit(f"Tool calling thất bại: {type(exc).__name__}: {exc}")

calls = resp.choices[0].message.tool_calls or []
if not calls:
    sys.exit("Model không trả về tool call — chọn model khác hỗ trợ tool calling.")
print(f"[tool]  {calls[0].function.name}({json.loads(calls[0].function.arguments)})")
print("Môi trường OK.")
