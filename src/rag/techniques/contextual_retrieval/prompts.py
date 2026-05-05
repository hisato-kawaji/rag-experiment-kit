"""Prompt for Anthropic's Contextual Retrieval (2024)."""

CONTEXT_PROMPT = """<document>
{doc}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk}
</chunk>

Please give a SHORT succinct context (1-2 sentences) to situate this chunk
within the overall document for the purposes of improving search retrieval.
Answer ONLY with the succinct context and nothing else.
"""
