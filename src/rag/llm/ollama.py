from __future__ import annotations

from typing import Any

import ollama


class OllamaLLM:
    def __init__(self, model: str, host: str = "http://localhost:11434") -> None:
        self.name = f"ollama/{model}"
        self.model = model
        self._client = ollama.Client(host=host)
        self._aclient = ollama.AsyncClient(host=host)

    def _build(
        self,
        prompt: str,
        system: str | None,
        temperature: float,
        max_tokens: int | None,
        json_mode: bool,
    ) -> tuple[list[dict[str, str]], dict[str, Any], str]:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        options: dict[str, Any] = {"temperature": temperature}
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        fmt = "json" if json_mode else ""
        return messages, options, fmt

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
        messages, options, fmt = self._build(prompt, system, temperature, max_tokens, json_mode)
        resp = self._client.chat(model=self.model, messages=messages, options=options, format=fmt)
        return resp["message"]["content"]

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
        messages, options, fmt = self._build(prompt, system, temperature, max_tokens, json_mode)
        resp = await self._aclient.chat(
            model=self.model, messages=messages, options=options, format=fmt
        )
        return resp["message"]["content"]
