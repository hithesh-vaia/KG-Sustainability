"""Neo4j persistence + retrieval queries."""
from __future__ import annotations

from contextlib import contextmanager

from neo4j import GraphDatabase

from .config import CONFIG, Config


class Neo4jClient:
    def __init__(self, config: Config = CONFIG):
        self.config = config
        self.driver = GraphDatabase.driver(
            config.neo4j_uri, auth=(config.neo4j_user, config.neo4j_password)
        )

    def close(self) -> None:
        self.driver.close()

    @contextmanager
    def session(self):
        with self.driver.session() as s:
            yield s

    # ---- schema -----------------------------------------------------------
    def _drop_stale_vector_indexes(self, dim: int) -> None:
        """Vector indexes are fixed-dimension and CREATE ... IF NOT EXISTS will not
        resize one. Switching EMBED_PROVIDER / EMBED_DIM therefore silently keeps the
        old dimension and every write fails, so drop any index that no longer matches.
        """
        from .log import get_logger

        log = get_logger()
        with self.session() as s:
            stale = [
                r["name"]
                for r in s.run(
                    "SHOW INDEXES YIELD name, type, options "
                    "WHERE type = 'VECTOR' RETURN name, options"
                )
                if r["options"]["indexConfig"].get("vector.dimensions") != dim
            ]
            for name in stale:
                log.info("dropping vector index %s (dimension != %d)", name, dim)
                s.run(f"DROP INDEX {name} IF EXISTS")

    def init_schema(self) -> None:
        dim = self.config.embed_dim
        self._drop_stale_vector_indexes(dim)
        stmts = [
            "CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT community_id IF NOT EXISTS FOR (c:Community) REQUIRE c.id IS UNIQUE",
            f"CREATE VECTOR INDEX entity_embedding IF NOT EXISTS FOR (e:Entity) ON (e.embedding) "
            f"OPTIONS {{indexConfig: {{`vector.dimensions`: {dim}, `vector.similarity_function`: 'cosine'}}}}",
            f"CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS FOR (c:Chunk) ON (c.embedding) "
            f"OPTIONS {{indexConfig: {{`vector.dimensions`: {dim}, `vector.similarity_function`: 'cosine'}}}}",
            f"CREATE VECTOR INDEX community_embedding IF NOT EXISTS FOR (c:Community) ON (c.embedding) "
            f"OPTIONS {{indexConfig: {{`vector.dimensions`: {dim}, `vector.similarity_function`: 'cosine'}}}}",
        ]
        with self.session() as s:
            for stmt in stmts:
                s.run(stmt)

    def reset(self) -> None:
        with self.session() as s:
            s.run("MATCH (n) DETACH DELETE n")

    # ---- ingestion upserts ---------------------------------------------------
    def upsert_chunks(self, rows: list[dict]) -> None:
        query = """
        UNWIND $rows AS row
        MERGE (c:Chunk {id: row.id})
        SET c.text = row.text, c.doc_id = row.doc_id,
            c.chunk_index = row.chunk_index, c.embedding = row.embedding
        """
        with self.session() as s:
            s.run(query, rows=rows)

    def upsert_entities(self, rows: list[dict]) -> None:
        query = """
        UNWIND $rows AS row
        MERGE (e:Entity {name: row.name})
        SET e.entity_type = row.entity_type, e.domain = row.community,
            e.description = row.description, e.properties = row.properties_json,
            e.provenance = row.provenance_json, e.confidence = row.confidence,
            e.degree = row.degree, e.embedding = row.embedding
        WITH e, row
        UNWIND row.source_chunks AS cid
        MATCH (c:Chunk {id: cid})
        MERGE (e)-[:MENTIONED_IN]->(c)
        """
        with self.session() as s:
            s.run(query, rows=rows)

    def upsert_relationships(self, rows: list[dict]) -> None:
        # relationship stays :RELATED (APOC-free); the ontology verb is r.type
        query = """
        UNWIND $rows AS row
        MATCH (a:Entity {name: row.source})
        MATCH (b:Entity {name: row.target})
        MERGE (a)-[r:RELATED {key: row.key}]->(b)
        SET r.type = row.rel_type, r.description = row.description,
            r.strength = row.strength, r.confidence = row.confidence,
            r.provenance = row.provenance_json,
            r.derived = coalesce(row.derived, false)
        """
        with self.session() as s:
            s.run(query, rows=rows)

    def upsert_communities(self, rows: list[dict]) -> None:
        query = """
        UNWIND $rows AS row
        MERGE (c:Community {id: row.id})
        SET c.level = row.level, c.parent = row.parent, c.title = row.title,
            c.summary = row.summary, c.rating = row.rating,
            c.full_content = row.full_content, c.embedding = row.embedding
        WITH c, row
        UNWIND row.members AS m
        MATCH (e:Entity {name: m})
        MERGE (e)-[:IN_COMMUNITY]->(c)
        """
        with self.session() as s:
            s.run(query, rows=rows)

    # ---- retrieval queries -------------------------------------------------
    def vector_search_entities(self, embedding: list[float], k: int) -> list[dict]:
        query = """
        CALL db.index.vector.queryNodes('entity_embedding', $k, $embedding)
        YIELD node, score
        RETURN node.name AS name, node.entity_type AS type, node.domain AS domain,
               node.description AS description, node.properties AS properties, score
        """
        with self.session() as s:
            return [dict(r) for r in s.run(query, k=k, embedding=embedding)]

    def vector_search_chunks(self, embedding: list[float], k: int) -> list[dict]:
        query = """
        CALL db.index.vector.queryNodes('chunk_embedding', $k, $embedding)
        YIELD node, score
        RETURN node.id AS id, node.text AS text, node.doc_id AS doc_id, score
        """
        with self.session() as s:
            return [dict(r) for r in s.run(query, k=k, embedding=embedding)]

    def entity_context(self, names: list[str], max_chunks: int = 8) -> dict:
        # Derived edges (period siblings, orphan-figure owner edges) are excluded
        # from neighbour expansion: they exist to connect the graph and feed
        # community detection, but pulling every sibling / every figure the org
        # owns into the context inflates the Entities section and evicts source
        # text and community reports from the token budget.
        # Independent CALL subqueries per collection -- chaining OPTIONAL MATCHes
        # instead multiplies rows (neighbours x chunks x communities per seed) and
        # is pathological now that hubs like the reporting org carry hundreds of
        # edges: a single query was taking minutes and collecting thousands of
        # copies of long community reports before DISTINCT.
        query = """
        MATCH (e:Entity) WHERE e.name IN $names
        WITH collect(e) AS seeds
        CALL (seeds) {
          UNWIND seeds AS e
          RETURN collect(DISTINCT {name: e.name, type: e.entity_type, domain: e.domain,
                                   description: e.description, properties: e.properties}) AS entities
        }
        CALL (seeds) {
          UNWIND seeds AS e
          MATCH (e)-[r:RELATED]-(nb:Entity)
          WHERE coalesce(r.derived, false) = false
          WITH r, nb LIMIT 400
          RETURN collect(DISTINCT {source: startNode(r).name, target: endNode(r).name,
                                   rel_type: r.type, description: r.description,
                                   strength: r.strength}) AS relationships,
                 collect(DISTINCT {name: nb.name, type: nb.entity_type,
                                   description: nb.description}) AS neighbors
        }
        CALL (seeds) {
          UNWIND seeds AS e
          MATCH (e)-[:MENTIONED_IN]->(ch:Chunk)
          WITH DISTINCT ch LIMIT $max_chunks
          RETURN collect({id: ch.id, text: ch.text}) AS chunks
        }
        CALL (seeds) {
          UNWIND seeds AS e
          MATCH (e)-[:IN_COMMUNITY]->(co:Community)
          WITH DISTINCT co ORDER BY co.rating DESC LIMIT 12
          RETURN collect({id: co.id, title: co.title, content: co.full_content,
                          rating: co.rating}) AS communities
        }
        RETURN entities, relationships, neighbors, chunks, communities
        """
        with self.session() as s:
            rec = s.run(query, names=names, max_chunks=max_chunks).single()
            return dict(rec) if rec else {}

    def community_reports(self, level: int | None = None) -> list[dict]:
        query = """
        MATCH (c:Community)
        WHERE $level IS NULL OR c.level = $level
        RETURN c.id AS id, c.level AS level, c.title AS title,
               c.rating AS rating, c.full_content AS full_content
        ORDER BY c.rating DESC
        """
        with self.session() as s:
            return [dict(r) for r in s.run(query, level=level)]

    def stats(self) -> dict:
        query = """
        RETURN
          COUNT { (e:Entity) } AS entities,
          COUNT { ()-[r:RELATED]->() } AS relationships,
          COUNT { (c:Chunk) } AS chunks,
          COUNT { (c:Community) } AS communities
        """
        with self.session() as s:
            return dict(s.run(query).single())
