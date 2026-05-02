"""Streamlit query inspector — pick pipeline, see answer + contexts side-by-side."""

from __future__ import annotations

import streamlit as st

from rag.config import Settings
from rag.pipelines import build_pipeline

st.set_page_config(page_title="RAG dev env", layout="wide")

PIPELINES = ["baseline", "graphrag/global", "graphrag/local"]


@st.cache_resource(show_spinner=False)
def _build(name: str, top_k: int):
    if name == "baseline":
        return build_pipeline("baseline", top_k=top_k)
    if name == "graphrag/global":
        return build_pipeline("graphrag", mode="global")
    if name == "graphrag/local":
        return build_pipeline("graphrag", mode="local", top_k=top_k)
    raise ValueError(name)


def render_answer(container, name: str, query: str, top_k: int) -> None:
    with container:
        st.markdown(f"### `{name}`")
        if not query:
            return
        try:
            pipe = _build(name, top_k)
        except Exception as e:
            st.error(f"failed to load `{name}`: {e}")
            st.caption("→ For graphrag, run `task build-graph` first.")
            return
        with st.spinner(f"running {name}…"):
            try:
                ans = pipe.answer(query)
            except Exception as e:
                st.error(f"answer failed: {e}")
                return
        st.markdown("**Answer**")
        st.write(ans.text)
        st.markdown(f"**Contexts** — {len(ans.contexts)}")
        for i, c in enumerate(ans.contexts, 1):
            with st.expander(
                f"{i}. {c.chunk.doc_id}  ·  score={c.score:.3f}", expanded=(i == 1)
            ):
                meta = {
                    k: v
                    for k, v in (c.chunk.metadata or {}).items()
                    if k not in {"text"} and not isinstance(v, list)
                }
                if meta:
                    st.caption(" · ".join(f"{k}={v}" for k, v in meta.items()))
                st.write(
                    c.chunk.text[:2000]
                    + ("…" if len(c.chunk.text) > 2000 else "")
                )
                findings = (c.chunk.metadata or {}).get("findings")
                if findings:
                    st.markdown("**Findings:**")
                    for f in findings:
                        st.markdown(f"- {f}")
        with st.expander("Pipeline metadata", expanded=False):
            st.json(ans.metadata)


def main() -> None:
    st.title("RAG dev env — query inspector")
    s = Settings()
    st.caption(
        f"LLM: {s.ollama_model_llm}  ·  Embed: {s.ollama_model_embed}  ·  "
        f"Qdrant: {s.qdrant_collection} @ {s.qdrant_url}"
    )

    with st.sidebar:
        st.header("Run")
        top_k = st.slider("top_k", 1, 20, 5)
        compare = st.checkbox("Side-by-side compare", value=True)
        if compare:
            left = st.selectbox("left pipeline", PIPELINES, index=0)
            right = st.selectbox(
                "right pipeline",
                PIPELINES,
                index=1 if len(PIPELINES) > 1 else 0,
            )
        else:
            single = st.selectbox("pipeline", PIPELINES, index=0)
        st.divider()
        if st.button("Clear pipeline cache"):
            _build.clear()
            st.rerun()

    query = st.text_area("Question", placeholder="Ask something…", height=80)
    submit = st.button("Run", type="primary")

    if not submit:
        st.info(
            "Enter a question and hit **Run**.  "
            "Prereqs: `task ingest -- ...` for baseline; `task build-graph` for graphrag."
        )
        return

    if compare:
        c1, c2 = st.columns(2, gap="large")
        render_answer(c1, left, query, top_k)
        render_answer(c2, right, query, top_k)
    else:
        render_answer(st.container(), single, query, top_k)


if __name__ == "__main__":
    main()
