from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LLM(Protocol):
    """Minimal LLM surface used across pipelines and techniques."""

    name: str

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
        **kwargs: Any,
    ) -> str: ...

    async def acomplete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
        **kwargs: Any,
    ) -> str: ...
