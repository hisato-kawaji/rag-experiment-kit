from __future__ import annotations

from collections.abc import Iterator

import trafilatura

from ...logging import log
from ...types import Document


class WebSource:
    """Fetch arbitrary URLs and extract main text via trafilatura."""

    name = "web"

    def fetch(self, *, urls: list[str]) -> Iterator[Document]:
        for url in urls:
            html = trafilatura.fetch_url(url)
            if not html:
                log.warning("web.fetch.failed", url=url)
                continue
            text = trafilatura.extract(html) or ""
            text = text.strip()
            if not text:
                log.warning("web.extract.empty", url=url)
                continue
            yield Document(
                id=url,
                text=text,
                metadata={"url": url, "source": self.name},
            )
