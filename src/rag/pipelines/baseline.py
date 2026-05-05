from __future__ import annotations

from ..embedding import Embedding, build_embedding
from ..llm import LLM, build_llm
from ..tracing import traced_span
from ..types import Answer, RetrievedChunk
from ..vectorstores import VectorStore, build_vectorstore

ANSWER_PROMPT = """You are a precise assistant. Answer the user's question using ONLY the context below.
If the answer is not contained in the context, say "I don't know based on the provided sources."
Cite the source doc_ids inline as [doc_id].

Context:
{context}

Question: {question}

Answer:"""


class BaselineRAG:
    """Dense-retrieval RAG: embed query → top-k from Qdrant → stuff into prompt → LLM."""

    name = "baseline"

    def __init__(
        self,
        *,
        top_k: int = 5,
        llm: LLM | None = None,
        embedding: Embedding | None = None,
        vectorstore: VectorStore | None = None,
    ) -> None:
        self.top_k = top_k
        self.llm = llm or build_llm()
        self.embedding = embedding or build_embedding()
        self.vectorstore = vectorstore or build_vectorstore()

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        qv = self.embedding.embed([query])[0]
        return self.vectorstore.search(qv, k=self.top_k)

    def answer(self, query: str) -> Answer:
        with traced_span("baseline.answer", top_k=self.top_k, query_chars=len(query)):
            contexts = self.retrieve(query)
            context_block = "\n\n".join(f"[{c.chunk.doc_id}] {c.chunk.text}" for c in contexts)
            prompt = ANSWER_PROMPT.format(context=context_block, question=query)
            text = self.llm.complete(prompt, temperature=0.0, max_tokens=512)
            return Answer(
                text=text,
                contexts=contexts,
                metadata={"pipeline": self.name, "top_k": self.top_k, "llm": self.llm.name},
            )
