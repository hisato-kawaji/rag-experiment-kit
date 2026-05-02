from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedding(Protocol):
    name: str

    @property
    def dim(self) -> int: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...

    async def aembed(self, texts: list[str]) -> list[list[float]]: ...
