"""Cấu hình LLM dùng chung, đọc từ biến môi trường hoặc file .env.

Biến:
    PROVIDER         openai | openrouter (mặc định: openai)
    MODEL            tên model hỗ trợ tool calling, ví dụ gpt-4o-mini
                     (với OpenRouter: openai/gpt-4o-mini)
    OPENAI_API_KEY   API key của provider tương ứng
    OPENAI_BASE_URL  (tùy chọn) ghi đè endpoint
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI

# Biến môi trường đã export trong shell được ưu tiên hơn .env
load_dotenv(override=False)

BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    api_key: str
    base_url: str


def load_config() -> LLMConfig:
    provider = os.getenv("PROVIDER", "openai").strip().lower()
    if provider not in BASE_URLS:
        raise ValueError(f"PROVIDER không hỗ trợ: {provider!r} (chọn: {', '.join(BASE_URLS)})")

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or api_key == "YOUR_KEY":
        raise ValueError("Chưa có OPENAI_API_KEY (export hoặc điền vào .env).")
    if provider == "openai" and api_key.startswith("sk-or-"):
        raise ValueError("Key dạng sk-or-... là của OpenRouter: đặt PROVIDER=openrouter.")

    model = os.getenv("MODEL", "").strip() or "gpt-4o-mini"
    if model == "YOUR_TOOL_CALLING_MODEL":
        raise ValueError("Chưa đặt MODEL.")
    if provider == "openrouter" and "/" not in model:
        model = f"openai/{model}"  # OpenRouter cần tiền tố nhà cung cấp

    base_url = os.getenv("OPENAI_BASE_URL", "").strip() or BASE_URLS[provider]
    return LLMConfig(provider, model, api_key, base_url)


def get_client(cfg: LLMConfig | None = None) -> tuple[OpenAI, LLMConfig]:
    cfg = cfg or load_config()
    return OpenAI(api_key=cfg.api_key, base_url=cfg.base_url), cfg
