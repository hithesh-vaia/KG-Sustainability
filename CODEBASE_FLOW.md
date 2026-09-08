# graphrag-lite — Codebase Flow

A small, readable **Graph RAG** system: turn documents (IDBI Bank BRSR / sustainability
PDFs) into a typed knowledge graph in Neo4j, then answer questions by walking that graph.

- **LLM:** OpenRouter (OpenAI-compatible), default `openai/gpt-4o-mini` — `graphrag/llm.py`
- **Embeddings:** local `fastembed` (`BAAI/bge-small-en-v1.5`, 384-dim) — `graphrag/embeddings.py`
- **Graph store:** Neo4j 5.11+ with native vector indexes — `graphrag/graphdb.py`
- **Entry point:** `cli.py` (Typer) — commands `ingest`, `query`, `stats`

---

## 1. Big picture

```
                 ┌─────────────────── INGEST (cli.py ingest) ───────────────────┐
 input/*.pdf ──► load + chunk ──► LLM extract ──► merge/dedupe ──► NetworkX graph
 .txt .md .xlsx     (chunking.py)   (extract.py)     (merge.py)       (build.py)
                                                                          │
                                       Leiden communities ◄───────────────┘
                                       + LLM community reports (communities.py)
                                                     │
                                       local embeddings (embeddings.py)
                                                     │
                                 ┌───────────────────┴───────────────────┐
                            Neo4j (nodes + edges + vector indexes)   output/*.parquet
                                       graphdb.py                    (debug artifacts)

                 ┌─────────────────── QUERY (cli.py query) ─────────────────────┐
 question ──► local_search   (entity-centric graph walk)   ──► LLM answer
        └──►  global_search  (map-reduce over community reports) ──► LLM answer
```

Everything LLM-related is **disk-cached** under `output/cache/` (`extract_v2_*`,
`summary_*`, `report_*`), so re-running ingest costs almost nothing.

---

## 2. Configuration — `graphrag/config.py`

`CONFIG` is a single dataclass instance built from environment / `.env`:
OpenRouter key+model, Neo4j URI/creds, embed model+dim, and ingestion tuning
(`CHUNK_TOKENS=1200`, `CHUNK_OVERLAP=100`, `MAX_GLEANINGS=1`, `MAX_CLUSTER_SIZE=10`,
`LLM_CONCURRENCY=4`). Paths: `input/`, `output/`, `output/cache/`.
`CONFIG.validate()` fails fast if `OPENROUTER_API_KEY` is missing.

---

## 3. Ingestion pipeline — `graphrag/ingest/pipeline.py::run_ingest`

Orchestrates the whole build. Steps, in order:

### 3.1 Load + chunk — `graphrag/chunking.py`
- `load_documents()` walks `input/` (recursively). Readers by extension:
  - `.pdf` → `pypdf`, one `===== PAGE N =====` marker per page (kept so the LLM can
    record `page_number` in provenance).
  - `.xlsx/.xls/.csv` → pandas → markdown tables, one block per sheet.
  - `.txt/.md` → raw text.
  - Returns `{doc_id: text}` where `doc_id` is the filename stem.
- `chunk_documents()` uses LangChain `TokenTextSplitter` (1200 tokens, 100 overlap)
  → list of `Chunk(id, doc_id, chunk_index, text)`. `id` = sha256(`doc_id:index`)[:16].

### 3.2 Extract — `graphrag/ingest/extract.py::extract_all`
- Thread pool (`LLM_CONCURRENCY` wide). For each chunk, `extract_chunk`:
  1. Cache hit? load `output/cache/extract_v2_<chunk_id>.json` and skip the LLM.
  2. Otherwise `llm.chat_json()` with `prompts.EXTRACTION_SYSTEM` (the full ontology
     prompt from `sustainability_knowledge_graph_construction_prompt.txt`) +
     `EXTRACTION_USER` (the chunk).
  3. **Gleaning:** up to `MAX_GLEANINGS` follow-up calls asking "what did you miss?"
     (targets table measurements, targets, policies…). Payloads merged.
  4. Write merged JSON to cache.
- `parse_extraction()` turns the raw JSON into `ChunkExtraction(entities, relationships)`:
  - Names normalized to UPPER/whitespace-collapsed (`_norm_name`).
  - `entity_type` / `community` (domain) snapped to the allowed ontology terms
    (`_match` — case/punctuation-insensitive; unknown → `Metric` / `Measurement`).
  - **Period-scoping:** for `PERIOD_SCOPED_TYPES` (emissions, energy, water, waste,
    revenue…) the reporting period is appended to the name, e.g.
    `SCOPE 2 EMISSION (FY2024-25)` — so FY24 and FY25 figures never merge.
  - Relationship `source_id`/`target_id` resolved back to entity names.
  - `description` rendered from properties + a provenance source-text snippet.

### 3.3 Merge / dedupe — `graphrag/ingest/merge.py::merge_extractions`
- Entities keyed by **name**. Across all chunks:
  - properties merged, provenance appended, max confidence, source chunks unioned.
  - `entity_type` and `community` settled by **majority vote** across mentions.
  - entities that appear only as a relationship endpoint get a bare node.
- Relationships keyed by `(source, target, rel_type)`; provenance/chunks unioned.
- **Description consolidation:** entities with >1 distinct description get one LLM
  call (`DESCRIPTION_SUMMARIZATION`) to write a single factual description.
  Concurrent + cached as `summary_*.json`.
- Computes `degree` per entity; relationship `strength = max(confidence*10, 1)`.
- Returns `list[Entity]`, `list[Relationship]`.

### 3.4 Build graph — `graphrag/ingest/build.py`
`build_graph()` → an undirected `networkx.Graph` (nodes = entities, edges =
relationships with weight/rel_type). Purely an input to community detection;
Neo4j is the real store. (`largest_component` helper is available but unused by
the pipeline.)

### 3.5 Communities + reports — `graphrag/ingest/communities.py`
- `detect_communities()` runs **hierarchical Leiden** (`graspologic`) per connected
  component, `max_cluster_size=MAX_CLUSTER_SIZE`, seed 42. Produces `Community(id="L{level}-{cluster}", level, parent, members)`
  at multiple levels (L0 = broad, L1 = finer, …).
- `generate_reports()` — for each community, build a text digest of its member
  entities + internal relationships, then `llm.chat_json(COMMUNITY_REPORT)` →
  `{title, summary, rating 0-10, findings[]}`. Concurrent, cached as `report_*.json`.
  `full_content` = the assembled markdown report.

### 3.6 Embed — `graphrag/embeddings.py`
Local fastembed, three sets:
- chunks (page markers stripped)
- entities (`"<name> [<type> / <community>]: <description>"`)
- communities (`full_content` or `title`)

### 3.7 Write to Neo4j — `graphrag/graphdb.py::Neo4jClient`
- `reset()` (if `--reset`) then `init_schema()`: uniqueness constraints +
  three **vector indexes** (`entity_embedding`, `chunk_embedding`,
  `community_embedding`, cosine, dim = `EMBED_DIM`).
- `upsert_chunks / upsert_entities / upsert_relationships / upsert_communities`
  (batched `UNWIND ... MERGE`). Graph model:

  | node/edge | key properties |
  |---|---|
  | `(:Entity)` | `name` (unique), `entity_type`, `domain`, `description`, `properties`(JSON), `provenance`(JSON), `confidence`, `degree`, `embedding` |
  | `(:Chunk)` | `id`, `text`, `doc_id`, `chunk_index`, `embedding` |
  | `(:Community)` | `id`, `level`, `parent`, `title`, `summary`, `rating`, `full_content`, `embedding` |
  | `(:Entity)-[:RELATED {type, description, strength, confidence, provenance}]-(:Entity)` | all edges share Neo4j type `:RELATED`; the ontology verb is in `r.type` (APOC-free) |
  | `(:Entity)-[:MENTIONED_IN]->(:Chunk)` | provenance to source text |
  | `(:Entity)-[:IN_COMMUNITY]->(:Community)` | Leiden membership |

- Also dumps `output/{chunks,entities,relationships,communities}.parquet` for
  inspection.

---

## 4. The ontology — `graphrag/ontology.py`

Controlled vocabulary mirroring the extraction prompt:
- **13 communities/domains** (`Environmental`, `Social`, `Governance`, `Reporting`, …)
- **~200 entity types** (`Scope2Emission`, `WaterWithdrawal`, `AssuranceProvider`, …)
- **~120 relationship verbs** (`USES_FRAMEWORK`, `HAS_MEASUREMENT`, `ASSURED_BY`, …)
- `PERIOD_SCOPED_TYPES` — never merged across reporting years.
- Defaults for unmatched terms: `Metric` / `Measurement` / `RELATES_TO`.

`extract.py::_match` is what enforces this at parse time.

---

## 5. Query pipeline — `cli.py query`

`query(question, method=local|global|both, level=1, top_k=15, chunk_search=False)`.
Opens `Neo4jClient` + `LLM`, dispatches, prints answer + provenance + token usage.

### 5.1 Local search — `graphrag/retrieve/local_search.py`
Entity-centric, "pure GraphRAG": the **only** vector query is the seed selection.
1. `embed_one(question)` → `db.vector_search_entities(qvec, top_k)` → seed entities.
2. `db.entity_context(seed_names)` — one Cypher query walks out to: the seeds'
   properties, `:RELATED` neighbours + relationships, `:MENTIONED_IN` source
   chunks, `:IN_COMMUNITY` community reports.
3. Optional `--chunk-search`: also `vector_search_chunks(qvec)` — a plain-RAG
   safety net for facts trapped in tables.
4. `context.pack()` greedily fits sections (Entities / Source text / Relationships /
   Community reports) into an ~11k-token budget (tiktoken `cl100k_base`).
5. `llm.chat(LOCAL_ANSWER)` → `LocalResult(answer, entities, communities, chunks, context)`.

Use for: specific facts, numbers, names, "what is X", "how many employees".

### 5.2 Global search — `graphrag/retrieve/global_search.py`
Map-reduce over community reports only (never touches entities/chunks).
1. `db.community_reports(level)` — all `(:Community)` reports at that level
   (falls back to all levels if empty).
2. **Map:** batch reports into ~6k-token groups; per batch `llm.chat_json(GLOBAL_MAP)`
   → scored points (`description`, `score` 0-100).
3. Rank all points, keep top 30.
4. **Reduce:** `llm.chat(GLOBAL_REDUCE)` synthesises the final answer.
   If no points survive, it tells the user to try `--method local`.

Use for: themes, summaries, "how does the bank approach climate risk".

> `level 0` = broader communities, `level 1` (default) = finer-grained.

---

## 6. Supporting modules

| file | role |
|---|---|
| `graphrag/llm.py` | OpenRouter client. `chat()` / `chat_json()` (JSON mode + salvage), `tenacity` retry (5×, exp backoff), thread-safe `Usage` token counter. |
| `graphrag/log.py` | `setup_logging(kind)` — console + `output/logs/<kind>-<timestamp>.log`. |
| `graphrag/retrieve/context.py` | `ntokens()` and `pack()` token-budgeting helper shared by both searches. |
| `viz.py` | Standalone pyvis HTML graph export (`--color domain|type|leiden`, `--communities`). Reads Neo4j, writes `output/graph.html`. |
| `docker-compose.yml` | Neo4j 5.x container (`neo4j` / `password123`, ports 7474/7687). |

---

## 7. Evaluation — `eval/`

| file | role |
|---|---|
| `eval/gold.yaml` | hand-authored Q&A ground-truthed from the BRSR PDFs — `must_include` / `must_not_include` values, per-item `method`, plus negative "must decline" cases. |
| `eval/run_eval.py` | `python -m eval.run_eval` — runs the **live** `local_search` / `global_search` for each gold item and scores it. |
| `eval/normalize.py` | `contains_all` / `contains_any` — substring + numeric matching (1% tolerance, Indian digit grouping). |
| `eval/judge.py` | optional LLM-as-judge (`--judge`) → correct / partial / incorrect. |

Scoring separates **answer** (did the value land in the final answer) from
**context** (was it retrieved at all) → isolates retrieval gaps vs generation gaps.
Results → `eval/results/<timestamp>.json` + `latest.md`; `--compare` shows
FIXED / REGRESSED deltas; `--min-pass` gates CI.

Smoke test: `RUN_EVAL=1 pytest tests/test_eval_smoke.py` (needs Neo4j).
Offline unit tests: `pytest` (chunking, extraction parser, merge/dedupe).

---

## 8. Typical workflow

```bash
docker compose up -d                       # Neo4j
cp .env.example .env                        # set OPENROUTER_API_KEY
pip install -e ".[dev]"

python cli.py ingest --reset                # build the graph (cached after 1st run)
python cli.py stats                         # node/edge counts
python cli.py query "How many employees does IDBI Bank have, by gender?" --method local
python cli.py query "What are IDBI Bank's main sustainability commitments?" --method global
python -m eval.run_eval                     # score against gold.yaml
python viz.py --communities --level 1 --open # visual overview
```
