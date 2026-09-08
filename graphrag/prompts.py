"""Prompt templates.

Extraction uses the sustainability knowledge-graph ontology defined in
`sustainability_knowledge_graph_construction_prompt.txt` (repo root) — a
controlled vocabulary of communities / entity types / relationships, JSON output,
with per-fact provenance and confidence.
"""
from __future__ import annotations

from .config import ROOT

_EXTRACTION_SYSTEM_FILE = ROOT / "sustainability_knowledge_graph_construction_prompt.txt"

try:
    EXTRACTION_SYSTEM = _EXTRACTION_SYSTEM_FILE.read_text(encoding="utf-8")
except FileNotFoundError:  # pragma: no cover
    EXTRACTION_SYSTEM = (
        "You are a high-precision Sustainability Knowledge Graph Extraction Engine. "
        "Extract entities and relationships as JSON with keys 'entities' and "
        "'relationships'. Do not hallucinate."
    )

# the per-chunk user message
EXTRACTION_USER = """Extract the sustainability knowledge graph from the DOCUMENT CHUNK below.

- Follow every rule in the system instructions.
- Use ONLY the approved communities, entity types and relationship types.
- Fill provenance.source_text with the exact supporting text from this chunk.
- If the chunk contains "===== PAGE N =====" markers, use N as provenance.page_number.
- Return ONLY the JSON object (keys: document, entities, relationships). No prose.

DOCUMENT: {doc_id}

DOCUMENT CHUNK:
\"\"\"
{input_text}
\"\"\"
"""

EXTRACTION_GLEANING = """Some entities or relationships in that chunk were likely missed
(look especially for measurements in tables, targets, policies, frameworks,
disclosures, committees, locations).

Return a JSON object with the SAME structure containing ONLY the additional
entities and relationships. If nothing was missed, return {"entities": [], "relationships": []}.
Return ONLY JSON."""

# ---------------------------------------------------------------------------
# merge / community / retrieval prompts
# ---------------------------------------------------------------------------

DESCRIPTION_SUMMARIZATION = """You are consolidating descriptions for a knowledge-graph {kind}.
Given the {kind} name and a list of descriptions collected from different parts of
the source document(s), write ONE factual third-person description that resolves
overlaps. Do not add information that is not in the descriptions. Keep it under
150 words.

{kind}: {name}
Descriptions:
{descriptions}

Consolidated description:"""

COMMUNITY_REPORT = """You are an analyst writing a report about a cluster of related entities
from a sustainability / ESG knowledge graph. Given the entities and relationships
below, produce a JSON object:

{{
  "title": "<short descriptive name for this cluster>",
  "summary": "<what this cluster covers: the organisation(s), reports, measurements, targets, policies and how they connect>",
  "rating": <float 0-10 — how significant this cluster is to understanding the company's sustainability performance>,
  "rating_explanation": "<one sentence>",
  "findings": [
    {{"summary": "<insight headline>", "explanation": "<1-3 sentences, cite specific values / periods / frameworks where present>"}}
  ]
}}

Return 3-7 findings. Only use facts present in the data below.

######################
Cluster data:
{input_text}
######################
Output (JSON only):"""

LOCAL_ANSWER = """You are answering a question using the sustainability knowledge-graph
context below (entities with typed properties, relationships, community reports,
and verbatim source text).

- Use only information present in the context.
- Quote exact values, units and reporting periods where available.
- If the context is insufficient, say so plainly.
- Cite the entities or source documents you relied on.

######################
Context:
{context}
######################
Question: {question}

Answer:"""

GLOBAL_MAP = """You are a sustainability analyst. The context below is a batch of community
reports from a knowledge graph. Extract the points from these reports relevant to
the user's question.

Return JSON: {{"points": [{{"description": "<point, with values/periods if given>", "score": <int 0-100>}}]}}
Only include points supported by the context. If nothing is relevant: {{"points": []}}.

######################
Reports:
{context}
######################
Question: {question}

Output (JSON only):"""

GLOBAL_REDUCE = """You are a sustainability analyst. Below are analyst points (with importance
scores) gathered from many community reports. Synthesise them into one
comprehensive answer to the user's question. Merge duplicates, drop irrelevant
points, keep specific values / periods / framework names. Note if the points are
insufficient.

######################
Analyst points:
{context}
######################
Question: {question}

Answer:"""
