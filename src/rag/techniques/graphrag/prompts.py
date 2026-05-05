"""Prompts for the self-built GraphRAG (Edge et al. arXiv:2404.16130).

Kept in one file so they're easy to A/B by editing.
"""

EXTRACTION_SYSTEM = "You are an expert information extraction system."

EXTRACTION_PROMPT = """Extract all named entities and the relationships between them from the text below.

For each entity, output:
- name: the entity name (TitleCase, no leading articles)
- type: ONE of [PERSON, ORGANIZATION, LOCATION, EVENT, CONCEPT, PRODUCT, OTHER]
- description: a 1-2 sentence description grounded in the text

For each relationship, output:
- source: source entity name (must appear in entities)
- target: target entity name (must appear in entities)
- description: a 1-2 sentence description of the relationship grounded in the text
- weight: an integer 1-10 indicating the strength/importance of the relationship

Return STRICT JSON in this exact schema (no commentary, no markdown fences):
{{
  "entities": [{{"name": "...", "type": "...", "description": "..."}}],
  "relationships": [{{"source": "...", "target": "...", "description": "...", "weight": 5}}]
}}

Text:
\"\"\"
{text}
\"\"\"
"""

GLEANING_PROMPT = """You previously extracted entities and relationships from a text. Inspect the SAME text again and report ONLY items that were MISSED.

Already-extracted entities (do NOT repeat these):
{prior_entities}

Already-extracted relationships (do NOT repeat these):
{prior_relationships}

Text:
\"\"\"
{text}
\"\"\"

Rules:
- Output a NEW entity only if it is mentioned in the text and not in the list above.
- Output a NEW relationship only if both endpoints appear in the text and the pair (or its description) is not in the list above.
- If nothing was missed, return empty arrays.

Schema (no commentary, no markdown fences):
{{
  "entities": [{{"name": "...", "type": "...", "description": "..."}}],
  "relationships": [{{"source": "...", "target": "...", "description": "...", "weight": 5}}]
}}
"""

COMMUNITY_SUMMARY_PROMPT = """You are an analyst writing a report on a knowledge-graph community.

Entities in this community:
{entities}

Relationships among them:
{relationships}

Write a structured report with:
1. TITLE: a short evocative name for the community
2. SUMMARY: 3-5 sentences describing the community's overall theme
3. FINDINGS: 3-5 bullet points highlighting the most important facts/insights

Return STRICT JSON in this exact schema (no commentary):
{{"title": "...", "summary": "...", "findings": ["...", "...", "..."]}}
"""

GLOBAL_MAP_PROMPT = """You are answering a user question by reasoning over ONE community report.

Community report:
{report}

User question: {question}

Decide whether this community is helpful and, if so, draft a partial answer.

Return STRICT JSON in this exact schema (no commentary):
{{"helpfulness": 0-100, "partial_answer": "..."}}

If unhelpful, set helpfulness to 0 and partial_answer to "".
"""

GLOBAL_REDUCE_PROMPT = """Synthesize a final answer from multiple partial answers, each derived from a different community of a knowledge graph.

User question: {question}

Partial answers (sorted by helpfulness, highest first):
{partials}

Write the final answer. Be specific. Cite community findings inline as [community: <title>].
If the partial answers do not contain the information, say "I don't know based on the provided sources."

Final answer:
"""

LOCAL_PROMPT = """Answer the user question using a local knowledge-graph context.

Anchored entities (matching the query):
{entities}

Their direct neighborhood (related entities + relationships):
{neighborhood}

Supporting text chunks:
{chunks}

User question: {question}

Write a precise answer grounded in the context above. Cite chunks inline as [doc_id].
If the context does not contain the answer, say "I don't know based on the provided sources."

Answer:
"""
