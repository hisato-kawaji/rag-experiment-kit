from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from ...types import Document


@runtime_checkable
class Source(Protocol):
    name: str

    def fetch(self, **kwargs: object) -> Iterator[Document]: ...
