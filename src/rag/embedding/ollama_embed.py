from __future__ import annotations

import ollama

_KNOWN_DIMS: dict[str, int] = {
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "bge-m3": 1024,
    "snowflake-arctic-embed": 1024,
}


class OllamaEmbedding:
    def __init__(self, model: str, host: str = "http://localhost:11434") -> None:
        self.name = f"ollama/{model}"
        self.model = model
        self._client = ollama.Client(host=host)
        self._aclient = ollama.AsyncClient(host=host)
        self._dim: int | None = _KNOWN_DIMS.get(model.split(":")[0])

    @property
    def dim(self) -> int:
        if self._dim is None:
            v = self.embed(["dim-probe"])[0]
            self._dim = len(v)
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = self._client.embed(model=self.model, input=texts)
        return list(resp["embeddings"])

    async def aembed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = await self._aclient.embed(model=self.model, input=texts)
        return list(resp["embeddings"])
