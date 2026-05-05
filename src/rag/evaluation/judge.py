"""Build the LangChain LLM that Ragas uses as judge.

Generation pipelines (BaselineRAG / GraphRAG) keep using `rag.llm`. This module
exists so that *evaluation* can swap to a stronger judge (Anthropic/OpenAI)
without touching the generation side. Phase 1 demos showed `faithfulness=NaN`
when llama3.1:8b judged itself.
"""

from __future__ import annotations

from ..config import Settings


def build_judge_llm(settings: Settings | None = None):
    """Return a `ragas.llms.LangchainLLMWrapper` wrapping the configured backend."""
    s = settings or Settings()
    from ragas.llms import LangchainLLMWrapper

    backend = s.ragas_judge_backend

    if backend == "ollama":
        from langchain_ollama import ChatOllama

        model = s.ragas_judge_model or s.ollama_model_llm
        return LangchainLLMWrapper(ChatOllama(model=model, base_url=s.ollama_host, temperature=0.0))

    if backend == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as e:
            raise ImportError(
                "RAGAS_JUDGE_BACKEND=anthropic requires the `judge-anthropic` "
                "extra. Install with: uv sync --extra judge-anthropic"
            ) from e
        if not s.anthropic_api_key:
            raise ValueError("RAGAS_JUDGE_BACKEND=anthropic requires ANTHROPIC_API_KEY in .env")
        model = s.ragas_judge_model or "claude-haiku-4-5-20251001"
        return LangchainLLMWrapper(
            ChatAnthropic(
                model=model,
                api_key=s.anthropic_api_key,
                temperature=0.0,
                max_tokens=1024,
            )
        )

    if backend == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as e:
            raise ImportError(
                "RAGAS_JUDGE_BACKEND=openai requires the `judge-openai` extra. "
                "Install with: uv sync --extra judge-openai"
            ) from e
        if not s.openai_api_key:
            raise ValueError("RAGAS_JUDGE_BACKEND=openai requires OPENAI_API_KEY in .env")
        model = s.ragas_judge_model or "gpt-4o-mini"
        return LangchainLLMWrapper(
            ChatOpenAI(model=model, api_key=s.openai_api_key, temperature=0.0)
        )

    raise ValueError(f"Unknown RAGAS_JUDGE_BACKEND: {backend!r}")
