from __future__ import annotations

from typing import Any

from ..config import Settings
from ..logging import log


def evaluate_ragas(
    rows: list[dict[str, Any]], settings: Settings | None = None
) -> dict[str, float]:
    """Run Ragas faithfulness / answer relevancy / context precision+recall.

    rows schema: question, answer, contexts (list[str]), ground_truth (optional).
    """
    s = settings or Settings()
    try:
        from datasets import Dataset
        from langchain_ollama import OllamaEmbeddings
        from ragas import evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.metrics import Faithfulness, ResponseRelevancy

        extra_metrics = []
        try:
            from ragas.metrics import (
                LLMContextPrecisionWithReference,
                LLMContextRecall,
            )

            extra_metrics = [LLMContextPrecisionWithReference(), LLMContextRecall()]
        except ImportError:
            pass
    except ImportError as e:
        log.error("ragas.import-error", error=str(e))
        return {}

    from .judge import build_judge_llm

    try:
        llm = build_judge_llm(s)
    except (ImportError, ValueError) as e:
        log.error("ragas.judge.build-failed", backend=s.ragas_judge_backend, error=str(e))
        return {}
    log.info(
        "ragas.judge",
        backend=s.ragas_judge_backend,
        model=s.ragas_judge_model or "<default>",
    )

    emb = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(model=s.ollama_model_embed, base_url=s.ollama_host)
    )

    ragas_rows = [
        {
            "user_input": r["question"],
            "response": r["answer"],
            "retrieved_contexts": r.get("contexts", []),
            "reference": r.get("ground_truth", ""),
        }
        for r in rows
    ]
    ds = Dataset.from_list(ragas_rows)

    metrics = [Faithfulness(), ResponseRelevancy(), *extra_metrics]
    log.info("ragas.evaluate.start", n=len(rows), metrics=[m.name for m in metrics])
    try:
        result = evaluate(ds, metrics=metrics, llm=llm, embeddings=emb)
    except Exception as e:
        log.warning("ragas.evaluate.failed", error=str(e))
        return {}

    try:
        df = result.to_pandas()
    except Exception as e:
        log.warning("ragas.to_pandas.failed", error=str(e))
        return {}

    out: dict[str, float] = {}
    name_map = {
        "faithfulness": "faithfulness",
        "answer_relevancy": "answer_relevancy",
        "response_relevancy": "answer_relevancy",
        "context_precision": "context_precision",
        "llm_context_precision_with_reference": "context_precision",
        "context_recall": "context_recall",
        "llm_context_recall": "context_recall",
    }
    for col, key in name_map.items():
        if col in df.columns:
            try:
                out[key] = float(df[col].mean(skipna=True))
            except Exception:
                continue
    log.info("ragas.evaluate.done", **out)
    return out
