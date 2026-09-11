"""graphrag-lite CLI: ingest documents and query the knowledge graph."""
from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(add_completion=False, help="Graph RAG ingestion + retrieval (OpenRouter + Neo4j)")


@app.command()
def ingest(
    input: Path = typer.Option(None, help="Directory of .txt/.md files (default: ./input)"),
    reset: bool = typer.Option(False, help="Wipe the Neo4j graph before ingesting"),
    no_cache: bool = typer.Option(False, help="Ignore the extraction cache"),
):
    """Build the knowledge graph from documents."""
    from graphrag.ingest.pipeline import run_ingest

    run_ingest(input_dir=input, reset=reset, use_cache=not no_cache)


@app.command()
def query(
    question: str,
    method: str = typer.Option(
        "local",
        help="local (facts/names/numbers) | global (themes/summaries) | both",
    ),
    level: int = typer.Option(1, help="Community level for global search"),
    top_k: int = typer.Option(15, help="Seed entities for local search"),
    chunk_search: bool = typer.Option(
        True, help="local: also do a direct query->chunk vector search (plain RAG safety net)"
    ),
):
    """Ask a question against the knowledge graph."""
    from graphrag.graphdb import Neo4jClient
    from graphrag.llm import LLM
    from graphrag.log import setup_logging

    setup_logging("query")

    if method not in ("local", "global", "both"):
        raise typer.BadParameter("method must be 'local', 'global', or 'both'")

    db = Neo4jClient()
    llm = LLM()
    try:
        if method in ("local", "both"):
            from graphrag.retrieve.local_search import local_search

            res = local_search(question, db, llm, top_k=top_k, chunk_search=chunk_search)
            if method == "both":
                typer.secho("── LOCAL ──────────────────────────────", fg="cyan")
            typer.echo(res.answer)
            typer.secho(
                f"\n[entities: {', '.join(res.entities)}]"
                f"\n[communities: {', '.join(res.communities)}]"
                f"\n[chunks: {', '.join(res.chunks)}]",
                fg="bright_black",
            )
        if method in ("global", "both"):
            from graphrag.retrieve.global_search import global_search

            res = global_search(question, db, llm, level=level)
            if method == "both":
                typer.secho("\n── GLOBAL ─────────────────────────────", fg="cyan")
            typer.echo(res.answer)
            typer.secho(
                f"\n[{len(res.themes)}  from {res.communities_used} community reports]",
                fg="bright_black",
            )
        typer.secho(f"[{llm.usage.summary()}]", fg="bright_black")
    finally:
        db.close()


@app.command()
def stats():
    """Show graph statistics."""
    from graphrag.graphdb import Neo4jClient

    db = Neo4jClient()
    try:
        typer.echo(db.stats())
    finally:
        db.close()


if __name__ == "__main__":
    app()
