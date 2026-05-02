from __future__ import annotations

from ..config import Settings
from .base import LLM
from .ollama import OllamaLLM
from .vllm import VLLMClient


def build_llm(
    settings: Settings | None = None,
    *,
    backend: str | None = None,
    model: str | None = None,
) -> LLM:
    """Construct an LLM client. Auto-selects vLLM when VLLM_BASE_URL is set, else Ollama."""
    settings = settings or Settings()
    backend = backend or ("vllm" if settings.vllm_base_url else "ollama")

    if backend == "ollama":
        return OllamaLLM(
            model=model or settings.ollama_model_llm,
            host=settings.ollama_host,
        )
    if backend == "vllm":
        if not settings.vllm_base_url:
            raise ValueError("VLLM_BASE_URL must be set in .env to use vLLM backend")
        return VLLMClient(
            base_url=settings.vllm_base_url,
            model=model or settings.vllm_model,
            api_key=settings.vllm_api_key,
        )
    raise ValueError(f"Unknown LLM backend: {backend!r}")


__all__ = ["LLM", "OllamaLLM", "VLLMClient", "build_llm"]
