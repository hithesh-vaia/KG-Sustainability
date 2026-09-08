# graphrag-lite

A small, readable **Graph RAG** pipeline — ingestion + retrieval — modeled on
Microsoft's GraphRAG approach and the
[ALucek/GraphRAG-Breakdown](https://github.com/ALucek/GraphRAG-Breakdown) walkthrough.

- **LLM:** OpenRouter (OpenAI-compatible API) — uses your OpenRouter credits
- **Embeddings:** local, via `fastembed` (`BAAI/bge-small-en-v1.5`) — no extra API key
- **Graph store:** Neo4j (with native vector indexes)
- **Retrieval:** local search (entity-centric) + global search (map-reduce over communities)

## Pipeline

```
documents ──► token chunking (1200/100, PDF pages marked)
          ──► LLM extraction into the sustainability ontology  (JSON, + gleaning, cached)
              · 13 domains · ~200 entity types · ~120 relationship types
              · every entity/relationship carries properties + provenance + confidence
              · measurements/emissions kept period-scoped (FY24 ≠ FY25)
          ──► merge / dedupe (type & domain by majority vote) / LLM description consolidation
          ──► NetworkX graph ──► hierarchical Leiden communities
          ──► LLM community reports
          ──► local embeddings (entities, chunks, reports)
          ──► Neo4j (nodes + RELATED{type} / MENTIONED_IN / IN_COMMUNITY, vector indexes)
              + parquet artifacts in output/
```

The extraction prompt/ontology lives in
`sustainability_knowledge_graph_construction_prompt.txt` (repo root); the allowed
terms are enforced in `graphrag/ontology.py`.

Retrieval:
- **local** — embed query → vector-search seed entities → traverse neighbors,
  relationships, source chunks, community reports → token-budgeted context → answer
- **global** — map over community reports at a level to extract scored points →
  rank → reduce into a final synthesized answer

## Setup

> **Python 3.10–3.12 recommended.** `graspologic` (Leiden) and `fastembed` depend on
> `numba`/`onnxruntime`, which may not yet have wheels for 3.13+.

```bash
# 1. Neo4j (needs 5.11+ for vector indexes)
docker compose up -d
# If Docker Desktop errors with "docker-credential-desktop ... not found", prepend:
#   export PATH="$PATH:/Applications/Docker.app/Contents/Resources/bin"

# 2. Python 3.12 venv  (3.13+ has no graspologic/fastembed wheels)
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 3. Config
cp .env.example .env      # then set OPENROUTER_API_KEY
```

## Usage

```bash
# put source files in input/  (.txt .md .pdf .xlsx .xls .csv), then:

# ---- INGEST: documents -> knowledge graph ----
python cli.py ingest --reset          # --reset wipes the graph first
python cli.py ingest                   # incremental; reuses output/cache/
python cli.py ingest --no-cache        # force re-extraction (spends credits)
python cli.py ingest --input pdfs/     # ingest from a different folder

# ---- QUERY ----
# local  = specific facts / numbers / names / "what is X"   (searches entities + source text)
# global = themes / summaries / "how does the bank approach X" (searches community reports only)
# both   = run both and print each
python cli.py query "How many employees does IDBI Bank have, by gender?" --method local
python cli.py query "..." --method local --chunk-search   # + plain-RAG safety net
python cli.py query "What are IDBI Bank's main sustainability commitments?" --method global
python cli.py query "..." --method both
python cli.py query "..." --method global --level 0   # broader communities

# ---- INSPECT ----
python cli.py stats                    # node/edge counts
```

### Logs

Every run prints progress to the console **and** writes a timestamped file:

```
output/logs/ingest-YYYYMMDD-HHMMSS.log
output/logs/query-YYYYMMDD-HHMMSS.log
```

Run in the background and follow the log:

```bash
nohup python cli.py ingest --reset > /dev/null 2>&1 &
tail -f output/logs/ingest-*.log
```

`ingest` caches every LLM result under `output/cache/`
(`extract_*` chunk extractions, `summary_*` merged descriptions, `report_*`
community reports), so re-runs skip the LLM and cost almost nothing.

### How long ingestion takes

All three LLM phases run `LLM_CONCURRENCY`-wide (default 4) and are cached.
Rough numbers for the two IDBI BRSR PDFs (~90 chunks, ~900 entities):

| phase | LLM calls | first run | cached re-run |
|---|---|---|---|
| extraction | ~90–180 | 3–6 min | instant |
| merge / description summary | ~450 | 4–6 min | instant |
| community reports | ~100–200 | 2–4 min | instant |
| embedding + Neo4j write | 0 (local) | ~1 min | ~1 min |

Raise `LLM_CONCURRENCY` in `.env` (e.g. 8–10) to roughly halve the first-run times.

## Tests

```bash
pytest        # chunking, extraction parser, merge/dedupe — no network needed
```

## Evaluation

`eval/gold.yaml` is a hand-authored Q&A set ground-truthed from the source BRSR
PDFs (emissions, energy, water, headcount, assurance, frameworks, plus negative
"must decline" cases). `eval/run_eval.py` runs the **live retrieval pipeline**
against it and scores each answer.

```bash
python -m eval.run_eval                    # deterministic substring/number scoring
python -m eval.run_eval --judge            # + LLM-as-judge (correct/partial/incorrect)
python -m eval.run_eval --only environmental
python -m eval.run_eval --compare eval/results/<earlier>.json   # FIXED / REGRESSED diff
python -m eval.run_eval --min-pass 0.8     # exit 1 if pass rate below threshold
```

Each item is scored on:
- **answer** — do the required values/units appear in the final answer (numbers
  matched with 1% tolerance; Indian digit grouping handled)
- **context** — were they in the assembled context at all → separates *retrieval*
  gaps from *generation* gaps
- **judge** (optional) — an LLM compares the answer to the reference

Results land in `eval/results/<timestamp>.json` + `eval/results/latest.md`.
Workflow: run a baseline, change the prompt / chunking / retrieval, re-run with
`--compare`, watch the delta. Set `EVAL_JUDGE_MODEL` to grade with a stronger
model than extraction uses.

Opt-in end-to-end smoke test (needs Neo4j): `RUN_EVAL=1 pytest tests/test_eval_smoke.py`.

## Visualize the graph

```bash
python viz.py --open                                   # entities coloured by ESG domain
python viz.py --color type --open                       # coloured by entity_type
python viz.py --color leiden --level 1 --open           # coloured by detected cluster
python viz.py --communities --level 1 --open            # community-level map
```

Each writes a single self-contained HTML file to `output/` (default
`graph.html`). Nodes are sized by degree; only the ~30 most-connected are
labelled (rest on hover); hovering fades everything but a node's neighbours;
there's a legend top-left. `--labels all|top|none` controls how many names are
drawn (default `top` = 60 highest-degree). `--top` / `--min-degree` control how
many nodes are kept. The `--communities` view collapses each Leiden community to one node
(size = member count, colour = impact rating, edges = cross-community links) —
the clearest overview.

Or explore interactively in **Neo4j Browser** (http://localhost:7474,
`neo4j` / `password123`):

```cypher
MATCH (e:Entity) WITH e ORDER BY e.degree DESC LIMIT 40
MATCH (e)-[r:RELATED]-(m:Entity) WHERE m.degree > 3
RETURN e, r, m
```

## How Neo4j retrieval works

Ingestion writes this graph model:

| element | properties |
|---|---|
| `(:Entity)` | `name` (unique), `entity_type`, `domain`, `description`, `properties` (JSON), `provenance` (JSON), `confidence`, `degree`, `embedding` |
| `(:Chunk)` | `id`, `text`, `doc_id`, `chunk_index`, `embedding` |
| `(:Community)` | `id`, `level`, `parent`, `title`, `summary`, `rating`, `full_content`, `embedding` |
| `(:Entity)-[:RELATED {type, description, confidence, strength, provenance}]-(:Entity)` | `type` is the ontology verb (`USES_FRAMEWORK`, `HAS_MEASUREMENT`, …) |
| `(:Entity)-[:MENTIONED_IN]->(:Chunk)` | source chunk |
| `(:Entity)-[:IN_COMMUNITY]->(:Community)` | Leiden membership |

> All relationships share the `:RELATED` Neo4j type (APOC-free) with the ontology
> verb in `r.type`; filter with `WHERE r.type = 'USES_FRAMEWORK'`.

**Local search** is pure GraphRAG: the *only* embedding query picks ~15 seed
entities; everything else (relationships, neighbours, source chunks via
`:MENTIONED_IN`, community reports) is reached by walking the graph. Add
`--chunk-search` to also run a direct query→chunk vector search (plain RAG) as a
safety net for table-bound facts.
```cypher
CALL db.index.vector.queryNodes('entity_embedding', 10, $qvec) YIELD node
MATCH (node)-[r:RELATED]-(nb:Entity)
OPTIONAL MATCH (node)-[:MENTIONED_IN]->(ch:Chunk)
OPTIONAL MATCH (node)-[:IN_COMMUNITY]->(co:Community)
RETURN node, collect(nb), collect(r), collect(ch), collect(co)
```
**Global search** reads `(:Community)` reports at a level and does LLM map-reduce
over them. See `graphrag/graphdb.py` for the exact queries.

> **Pick the right mode.** `global` only ever sees the 120 thematic community
> summaries — it cannot answer "how many X" or "who is Y". Use `local` for any
> concrete fact, number, name, or entity; use `global` for "what are the themes /
> how does the bank approach …"; use `both` when unsure.

## Roadmap / open items (from `requirments.md`)

- **Accuracy (95–100%):** raise `MAX_GLEANINGS`, use a stronger `OPENROUTER_MODEL`,
  add an extraction-verification pass, and curate the entity-type list per domain
  (e.g. sustainability: `emission_scope`, `metric`, `target`, `regulation`, `facility`).
- **LangChain wrapper:** `local_search` / `global_search` can be exposed as a
  LangChain `Retriever` / `Tool` — not yet implemented.
- **Domain tuning:** for IDBI sustainability reports, add a domain prompt preamble
  in `graphrag/prompts.py` and table-aware chunking for Excel disclosures.

## Config (`.env`)

| var | default | notes |
|-----|---------|-------|
| `OPENROUTER_MODEL` | `openai/gpt-4o-mini` | any OpenRouter chat model |
| `EMBED_MODEL` / `EMBED_DIM` | `BAAI/bge-small-en-v1.5` / `384` | must match Neo4j vector index dim |
| `CHUNK_TOKENS` / `CHUNK_OVERLAP` | `1200` / `100` | |
| `MAX_GLEANINGS` | `1` | extra extraction passes per chunk |
| `MAX_CLUSTER_SIZE` | `10` | Leiden max community size |
| `LLM_CONCURRENCY` | `4` | parallel extraction requests |

> Changing `EMBED_DIM` requires dropping the Neo4j vector indexes (or `ingest --reset`
> after `MATCH (n) DETACH DELETE n` and `DROP INDEX ...`).
