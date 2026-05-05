from __future__ import annotations

from .base import Source
from .web import WebSource
from .wikipedia import WikipediaSource

SOURCES: dict[str, type] = {
    "wikipedia": WikipediaSource,
    "web": WebSource,
}


def build_source(name: str, **init_kwargs: object) -> Source:
    if name not in SOURCES:
        raise ValueError(f"Unknown source: {name!r}. Known: {sorted(SOURCES)}")
    return SOURCES[name](**init_kwargs)


__all__ = ["SOURCES", "Source", "WebSource", "WikipediaSource", "build_source"]
