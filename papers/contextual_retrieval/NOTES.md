# Contextual Retrieval (Anthropic, 2024)

Source: https://www.anthropic.com/news/contextual-retrieval

## Algorithm

For each chunk in a document:

1. Build the prompt:

       <document>{full_document}</document>

       Here is the chunk we want to situate within the whole document:
       <chunk>{chunk_text}</chunk>

       Please give a SHORT succinct context (1-2 sentences) to situate this
       chunk within the overall document for the purposes of improving
       search retrieval. Answer ONLY with the succinct context.

2. Call the LLM. Take the 1-2 sentence response.
3. Prepend the response (with a blank line separator) to the chunk text.
4. Embed the enriched text and store it in the vector DB.

At query time: dense retrieval as usual. The "magic" is purely indexing-side.

## Why it works

A naked chunk like "It was 14% faster than the previous version" has no
referent. After contextualization the embedded text reads "This chunk is from
the Q3 earnings report and refers to the new chip release. ... It was 14%
faster than the previous version" — far easier for the embedder to place
near the right query.

## Cost / caveats

- One extra LLM call per chunk at indexing time. With Ollama, that's just
  local compute. With Anthropic, prompt caching on the `<document>` block
  makes the doc portion ~free for chunks 2..N of the same doc.
- Increases chunk text length by ~120 tokens — within most embedder limits.
- We do *not* add prompt caching here; it lives in the LLM adapter (future
  work). For now each chunk re-sends the document.

## Use it

```bash
QDRANT_COLLECTION=rag_contextual task ingest -- --reset --chunker contextual \
    --source wikipedia --query "Quantum computing" --limit 20
QDRANT_COLLECTION=rag_contextual task run -- --pipeline contextual_retrieval
```

Compare side-by-side with the default collection:

```bash
task compare
```
