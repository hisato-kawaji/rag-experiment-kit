from __future__ import annotations

from collections.abc import Iterator

import wikipediaapi

from ...logging import log
from ...types import Document

_USER_AGENT = (
    "rag-experiment-kit/0.1 (research; https://github.com/hisato-kawaji/rag-experiment-kit)"
)


class WikipediaSource:
    """Pull Wikipedia articles by title query, optionally traversing outgoing links.

    `_traverse` deliberately uses a flat link iteration here; relevance-filtered
    traversal lands in issue #3 on top of this interface.
    """

    name = "wikipedia"

    def __init__(self, language: str = "en", traverse_depth: int = 0) -> None:
        self._wiki = wikipediaapi.Wikipedia(user_agent=_USER_AGENT, language=language)
        self.traverse_depth = traverse_depth

    def fetch(self, *, query: str, limit: int = 20) -> Iterator[Document]:
        seed = self._wiki.page(query)
        if not seed.exists():
            log.warning("wikipedia.seed.missing", query=query)
            return
        seen: set[str] = set()
        for i, doc in enumerate(self._walk(seed, depth=self.traverse_depth, seen=seen), start=1):
            yield doc
            if i >= limit:
                return

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
            yield from self._walk(link_page, depth=depth - 1, seen=seen)
