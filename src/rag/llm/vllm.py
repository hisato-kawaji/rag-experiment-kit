from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI, OpenAI


class VLLMClient:
    """Talk to a vLLM (or any OpenAI-compatible) server."""

    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY") -> None:
        self.name = f"vllm/{model}"
        self.model = model
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._aclient = AsyncOpenAI(base_url=base_url, api_key=api_key)

    def _kwargs(self, max_tokens: int | None, json_mode: bool) -> dict[str, Any]:
        kw: dict[str, Any] = {}
        if max_tokens is not None:
            kw["max_tokens"] = max_tokens
        if json_mode:
            kw["response_format"] = {"type": "json_object"}
        return kw

    def _messages(self, prompt: str, system: str | None) -> list[dict[str, str]]:
        msgs: list[dict[str, str]] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        return msgs

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
        **kwargs: Any,
    ) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=self._messages(prompt, system),
            temperature=temperature,
            **self._kwargs(max_tokens, json_mode),
        )
        return resp.choices[0].message.content or ""

    async def acomplete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
        **kwargs: Any,
    ) -> str:
        resp = await self._aclient.chat.completions.create(
            model=self.model,
            messages=self._messages(prompt, system),
            temperature=temperature,
            **self._kwargs(max_tokens, json_mode),
        )
        return resp.choices[0].message.content or ""
