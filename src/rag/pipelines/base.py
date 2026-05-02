from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..types import Answer


@runtime_checkable
class Pipeline(Protocol):
    name: str

    def answer(self, query: str) -> Answer: ...
