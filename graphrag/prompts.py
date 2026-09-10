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
- If the chunk contains "<!-- Page number: N -->" markers, use N as provenance.page_number.
- Return ONLY the JSON object (keys: document, entities, relationships). No prose.

REPORTING PERIOD
This document's reporting period is: {doc_period}

Every entity carrying a figure, count, rate, amount or percentage MUST have a
`reporting_period`. Decide it in this STRICT order of precedence:

1. AN EXPLICIT PERIOD ATTACHED TO THAT FIGURE ALWAYS WINS.
   These tables report several years side by side, and the text below may contain
   pre-flattened fact lines of the form
       - <table caption> | <row label> | <column label> = <value>
   When the column label is a period (e.g. "FY 2024-25*", "FY 2023-24^"), that
   column label IS the reporting period for that value. Use it verbatim, with any
   trailing footnote marks (* ^ @) removed.
   NEVER overwrite such a label with the document's reporting period. Most tables
   here show the current year AND a prior year, so a figure sitting in the
   prior-year column belongs to the PRIOR year even though this document's own
   period is {doc_period}.
2. A period stated in the surrounding prose for that specific figure.
3. ONLY IF neither of the above gives a period, fall back to the document's
   reporting period, {doc_period}. This is propagation of a stated fact, not
   inference, so it does not violate the "do not infer reporting periods" rule.
4. Use null when the figure is genuinely period-independent (a policy name, a
   framework, a location).

Emit ONE entity per (row, period) pair. A row with two year columns produces TWO
entities with different `reporting_period` values and different `value`s — never
one merged entity, and never two entities sharing the same period.

Also keep the ROW LABEL exact: "Permanent Employees" and "Total Employees" are
different rows with different values; do not conflate a sub-total with a total.

DOCUMENT: {doc_id}

DOCUMENT CHUNK:
\"\"\"
{input_text}
\"\"\"
"""

EXTRACTION_GLEANING = """Some entities or relationships in that chunk were likely missed
(look especially for measurements in tables, targets, policies, frameworks,
disclosures, committees, locations).

Also check the already-extracted items above: any entity carrying a figure, count,
rate, amount or percentage that is MISSING a `reporting_period` should be re-emitted
here with the correct period filled in (per the REPORTING PERIOD rules above).

Return a JSON object with the SAME structure containing ONLY the additional or
corrected entities and relationships. If nothing was missed, return
{"entities": [], "relationships": []}. Return ONLY JSON."""

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

The context deliberately contains MANY near-identical sibling entities — the same
metric for different reporting periods, for male/female splits, for permanent vs
contract staff, for different penalties or incidents. Picking the wrong sibling is
the most common way to get this wrong, so select deliberately:

1. SELECT BY NAME FIRST. Work out which entity the question is asking about by
   matching the question's reporting period, gender/category and scope against the
   entity NAME and its properties. Do not grab the first nearby number.
2. QUOTE VERBATIM. Reproduce values, units and periods exactly as written. Never
   round, rescale, average, sum or reconcile figures that disagree.
3. NAME YOUR SOURCE. For every figure you state, name the exact entity or chunk id
   it came from, e.g. "PERMANENT EMPLOYEES FEMALE (FY 2024-25) = 6,754".
4. IF SEVERAL SIBLINGS COULD MATCH, do not silently choose one. State each
   candidate with its full name and value, and say which best fits the question
   and why.
5. IF THE EXACT ITEM ASKED FOR IS ABSENT, say so plainly. Do not substitute a
   related figure. If a closely related figure exists, you may mention it, but
   label it clearly as not being what was asked.
6. ANSWER EVERY PART. If the question asks for several quantities, give all of
   them; mark any you cannot find as not found rather than omitting them.

######################
Context:
{context}
######################
Question: {question}

Answer:"""

GLOBAL_MAP = """You are a sustainability analyst performing the MAP stage of a global
GraphRAG search.

The context below contains a batch of community reports extracted from the
knowledge graph of a larger sustainability report.

Your job is to identify the important THEMES and TOPICS represented across
the entire batch.

IMPORTANT:

- Do NOT try to find only one point that is most relevant to the question.
- Extract multiple distinct themes when they are present.
- A community may contain a narrow topic such as water consumption,
  employee count, emissions, complaints, or energy usage. These are
  individual topics, NOT automatically the overall theme of the report.
- For questions asking about the overall theme, purpose, scope, or major
  areas of the report, identify broad themes represented by the batch.
- For factual questions asking for a specific number, value, date, or
  entity, extract the relevant factual evidence instead.
- Every theme must be supported by the supplied community reports.
- Do not invent themes that are not supported by the context.
- Preserve important terminology from the reports.
- Return multiple themes when the context supports them.

Return JSON only:

{{
  "themes": [
    {{
      "theme": "<broad theme or topic>",
      "description": "<what the reports say about this theme>",
      "importance": <int 0-100>
    }}
  ]
}}

If the batch contains no meaningful themes or evidence relevant to the
question, return:

{{"themes": []}}

######################
Community Reports:
{context}
######################

Question:
{question}

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
