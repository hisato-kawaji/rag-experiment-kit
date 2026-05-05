from __future__ import annotations

from collections.abc import Iterator

import wikipediaapi

from ...embedding.base import Embedding
from ...logging import log
from ...types import Document

_USER_AGENT = (
    "rag-experiment-kit/0.1 (research; https://github.com/hisato-kawaji/rag-experiment-kit)"
)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class WikipediaSource:
    """Pull Wikipedia articles by title query, optionally traversing outgoing links.

    When `related_only=True`, link targets are filtered by cosine similarity
    between the seed page text and each candidate's summary. Without an
    Embedding injected this filter is a no-op (the constructor warns).
    """

    name = "wikipedia"

    def __init__(
        self,
        language: str = "en",
        traverse_depth: int = 0,
        *,
        related_only: bool = False,
        relevance_threshold: float = 0.4,
        embedding: Embedding | None = None,
    ) -> None:
        self._wiki = wikipediaapi.Wikipedia(user_agent=_USER_AGENT, language=language)
        self.traverse_depth = traverse_depth
        self.related_only = related_only
        self.relevance_threshold = relevance_threshold
        self._embedding = embedding
        if related_only and embedding is None:
            log.warning("wikipedia.related_only.no_embedding")
            self.related_only = False
        self._seed_vec: list[float] | None = None

    def fetch(self, *, query: str, limit: int = 20) -> Iterator[Document]:
        seed = self._wiki.page(query)
        if not seed.exists():
            log.warning("wikipedia.seed.missing", query=query)
            return
        if self.related_only and self._embedding is not None:
            seed_text = (seed.summary or seed.text or query)[:4000]
            self._seed_vec = self._embedding.embed([seed_text])[0]
        seen: set[str] = set()
        for i, doc in enumerate(self._walk(seed, depth=self.traverse_depth, seen=seen), start=1):
            yield doc
            if i >= limit:
                return

    def _is_related(self, page: wikipediaapi.WikipediaPage) -> bool:
        if not self.related_only or self._embedding is None or self._seed_vec is None:
            return True
        candidate = (page.summary or "").strip()
        if not candidate:
            return False
        vec = self._embedding.embed([candidate[:2000]])[0]
        score = _cosine(self._seed_vec, vec)
        keep = score >= self.relevance_threshold
        log.info(
            "wikipedia.related.score",
            title=page.title,
            score=round(score, 3),
            keep=keep,
        )
        return keep

    def _walk(
        self,
        page: wikipediaapi.WikipediaPage,
        *,
        depth: int,
        seen: set[str],
    ) -> Iterator[Document]:
        if page.title in seen:
            return
        seen.add(page.title)
        text = (page.text or "").strip()
        if text:
            yield Document(
                id=page.fullurl or f"wiki:{page.title}",
                text=text,
                metadata={
                    "title": page.title,
                    "url": page.fullurl,
                    "source": self.name,
                },
            )
        if depth <= 0:
            return
        for link_title, link_page in page.links.items():
            if link_title in seen:
                continue
            try:
                if not link_page.exists():
                    continue
            except Exception as e:
                log.warning("wikipedia.link.error", title=link_title, error=str(e))
                continue
            if not self._is_related(link_page):
                continue
            yield from self._walk(link_page, depth=depth - 1, seen=seen)
